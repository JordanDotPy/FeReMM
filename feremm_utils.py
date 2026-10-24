import torch
import threading
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
import pandas as pd
from collections import Counter


@dataclass
class FeReMMStruct:
    data_sizes: defaultdict = field(default_factory=lambda: defaultdict(lambda: None))

    acc_before: defaultdict = field(default_factory=lambda: defaultdict(float))
    auc_before: defaultdict = field(default_factory=lambda: defaultdict(float))
    acc_after: float = 0.0
    auc_after: float = 0.0
    loss_before: defaultdict = field(default_factory=lambda: defaultdict(float))
    loss_after: float = 0.0

    act_tracker: defaultdict = field(default_factory=lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: []))))
    grad_tracker: defaultdict = field(default_factory=lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: []))))
    registered_hooks: defaultdict = field(default_factory=lambda: defaultdict(lambda: []))
    metric_tracker: defaultdict = field(default_factory=lambda: defaultdict(lambda: None))

    model_states: defaultdict = field(default_factory=lambda: defaultdict(lambda: None))
    client_updates: defaultdict = field(default_factory=lambda: defaultdict(lambda: None))

    policy_inputs: defaultdict = field(default_factory=lambda: defaultdict(lambda: defaultdict(dict)))
    policy_outputs: defaultdict = field(default_factory=lambda: defaultdict(lambda: None))
    policy_reward: float | torch.Tensor | None = None
    best_action: defaultdict = field(default_factory=lambda: defaultdict(lambda: None))


def merge_clients(gm, candidate_states, candidate_datasizes):
    total_samples = sum(candidate_datasizes)
    with torch.no_grad():
        merge_state = gm.state_dict()

        for key in merge_state.keys():
            if not merge_state[key].dtype.is_floating_point:
                continue
            avg = sum(candidate_states[i][key] * (candidate_datasizes[i] / total_samples)
                      for i in range(len(candidate_states)))
            merge_state[key] = avg

    gm.load_state_dict(merge_state)
    return gm


def compute_client_update(global_model, client_models, updates, layer_names):
    gm_params = dict(global_model.named_parameters())

    for client_id, client in client_models.items():
        cm_params = dict(client.named_parameters())
        pieces = []
        for name, gparam in gm_params.items():
            if not gparam.requires_grad:
                continue

            for layer in layer_names:
                if name.startswith(layer):
                    delta = cm_params[name].detach() - gparam.detach()
                    pieces.append(delta.reshape(-1))
                    break

        updates[client_id] = torch.cat(pieces, dim=0)
    return updates


def scaling_attack(gm, bm, scale):
    global_params = dict(gm.named_parameters())
    bad_params = dict(bm.named_parameters())

    with torch.no_grad():
        for name in bad_params:
            delta = bad_params[name].data - global_params[name].data
            attacked = global_params[name].data + scale * delta
            attacked = torch.nan_to_num(attacked, nan=0.0, posinf=1e6, neginf=-1e6)
            bad_params[name].data.copy_(attacked)
    return bm


def clear_hooks(client_hooks):
    for h in client_hooks: h.remove()
    client_hooks = []
    return client_hooks


hook_lock = threading.Lock()
def register_activations(model, client_id, layer_names, client_hooks, a_struct=None, type=None):
    def forward_hook(name):
        def hook(module, input, output):
            if output is None:
                return

            act = output.detach().view(output.size(0), -1)
            batch_sum = act.sum(dim=0)
            batch_count = act.shape[0]

            with hook_lock:
                if name not in a_struct[client_id][type]:
                    a_struct[client_id][type][name] = {
                        "sum": batch_sum,
                        "count": batch_count
                    }
                else:
                    a_struct[client_id][type][name]["sum"] += batch_sum
                    a_struct[client_id][type][name]["count"] += batch_count
        return hook

    for name, module in model.named_modules():
        if name in layer_names and name not in client_hooks:
            fh = module.register_forward_hook(forward_hook(name))
            client_hooks.append(fh)
    return client_hooks


def average_activations(data, id, hook):
    for layer in data.act_tracker[id][hook]:
        entry = data.act_tracker[id][hook][layer]
        mean_act = entry["sum"] / entry["count"]
        data.act_tracker[id][hook][layer] = mean_act


def activation_similarity(acts, eps=1e-8):
    kl_results = {}
    cos_results = {}
    act_gm = acts["gm"]["premerge"]
    for client_i in acts.keys():
        if client_i == "gm":
            continue
        print(f"Comparing {client_i}")
        act_i = acts[client_i]["premerge"]

        for layer_name in act_gm.keys():
            cos_sim = torch.nn.functional.cosine_similarity(act_gm[layer_name], act_i[layer_name], dim=0, eps=eps)
            print(f"  GM -> {client_i} Layer {layer_name} COS Similarity: {cos_sim:.4f}")
            cos_results[(client_i, layer_name)] = cos_sim

            log_gm = torch.log_softmax(act_gm[layer_name], dim=0)
            log_i = torch.log_softmax(act_i[layer_name], dim=0)

            dist_gm = torch.exp(log_gm)
            dist_i = torch.exp(log_i)

            kl = torch.sum(dist_gm * (log_gm - log_i))
            kl_results[(client_i, layer_name)] = kl
            print(f"  GM -> {client_i} Layer {layer_name} KL divergence: {kl:.4f}")
    return cos_results, kl_results


def retrieve_gradients(model, gradients_sum, layer_names):
    layer_grads = {layer: [] for layer in layer_names}
    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue

        for layer in layer_names:
            if name.startswith(layer):
                layer_grads[layer].append(param.grad.view(-1))
                break

    for layer in layer_grads:
        layer_grads[layer] = torch.cat(layer_grads[layer], dim=0)
        if gradients_sum[layer] is None:
            gradients_sum[layer] = torch.zeros_like(layer_grads[layer])
        gradients_sum[layer] += layer_grads[layer]
    return gradients_sum


def average_gradients(gradients_sum, layer_names, batch_count):
    for layer in layer_names:
        if gradients_sum[layer] is not None:
            gradients_sum[layer] /= max(batch_count, 1)
    return gradients_sum


def gradient_similarity(gt, eps=1e-8):
    cos_results = {}
    mag_results = {}
    grads_gm = gt["gm"]["premerge"]
    for client_i in gt.keys():
        if client_i == "gm":
            continue
        grads_i = gt[client_i]["premerge"]
        for layer_name in grads_i.keys():
            g_gm = grads_gm[layer_name]
            g_i = grads_i[layer_name]

            cos_sim = torch.nn.functional.cosine_similarity(g_gm, g_i, dim=0, eps=eps)

            norm_gm = torch.norm(g_gm, p=2)
            norm_i = torch.norm(g_i, p=2)

            mag_sim = (2 * norm_gm * norm_i) / (norm_gm**2 + norm_i**2 + eps)
            mag_sim = torch.nan_to_num(mag_sim, nan=0.0)

            cos_results[(client_i, layer_name)] = cos_sim
            mag_results[(client_i, layer_name)] = mag_sim

            print(f"   GM -> {client_i} Layer {layer_name} COS Similarity: {cos_sim:.4f}")
            print(f"   GM -> {client_i} Layer {layer_name} MAG Similarity: {mag_sim:.4f}")
    return cos_results, mag_results


def compute_total_parameter_distance(model_a, model_b):
    params_a = dict(model_a.named_parameters())
    params_b = dict(model_b.named_parameters())

    numerator = 0.0
    denominator = 0.0

    with torch.no_grad():
        for name in params_a:
            if not params_a[name].requires_grad:
                continue

            diff = params_a[name] - params_b[name]

            numerator += torch.sum(diff ** 2)
            denominator += torch.sum(params_b[name] ** 2)

    distance = torch.sqrt(numerator) / (torch.sqrt(denominator) + 1e-8)
    return distance.item()


def client_class_distribution(client_loaders):
    client_stats = {}
    for client_id, loader in enumerate(client_loaders):
        class_counts = Counter()
        total_samples = 0
        for batch in loader:
            targets = batch["target"]
            targets = targets.view(-1).cpu().numpy()
            class_counts.update(targets)
            total_samples += len(targets)
        percentages = {}

        for cls, count in class_counts.items():
            percentages[int(cls)] = 100.0 * count / total_samples
        client_stats[f"Client {client_id}"] = percentages

    df = pd.DataFrame(client_stats).T
    df = df.fillna(0.0)
    df = df.reindex(sorted(df.columns), axis=1)
    return df


def client_image_stats(client_loaders):
    rows = []

    for client_id, loader in enumerate(client_loaders):
        image_means = []
        image_stds = []

        for batch in loader:
            images = batch["image"].to("cpu", dtype=torch.float32)
            flat = images.view(images.size(0), -1)
            image_means.extend(flat.mean(dim=1).cpu().numpy())
            image_stds.extend(flat.std(dim=1).cpu().numpy())

        rows.append({
            "client": client_id,
            "mean_intensity": np.mean(image_means),
            "std_intensity": np.std(image_means),
            "mean_image_contrast": np.mean(image_stds),
            "std_image_contrast": np.std(image_stds),
        })

    return pd.DataFrame(rows)

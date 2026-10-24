import torch
import torch.nn as nn


class FeReMMPolicy(nn.Module):
    def __init__(self, num_layers, num_clients, hidden=64):
        super().__init__()

        self.num_clients = num_clients
        self.num_layers = num_layers

        self.grad_proj = nn.Linear(2, hidden)
        self.grad_layer_embedding = nn.Embedding(num_layers, hidden)

        # Gradient Branch
        self.grad_branch = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(inplace=True),
        )

        self.act_proj = nn.Linear(2, hidden)
        self.act_layer_embedding = nn.Embedding(num_layers, hidden)

        # Activation Branch
        self.act_branch = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(inplace=True),
        )

        self.met_proj = nn.Linear(2, hidden)
        self.met_layer_embedding = nn.Embedding(num_clients, hidden)

        # Metric Branch
        self.metric_branch = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(inplace=True),
        )

        self.loss_proj = nn.Linear(2, hidden)
        self.loss_layer_embedding = nn.Embedding(num_clients, hidden)

        # Loss Branch
        self.loss_branch = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(inplace=True),
        )

        # Prediction Head
        self.head = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, 2),
        )


    def forward(self, state, device):
        g = self.grad_proj(state["grad"])
        g_emb = self.grad_layer_embedding(torch.arange(self.num_layers, device=device))
        g = g + g_emb.unsqueeze(0)
        g = self.grad_branch(g)
        g = g.mean(dim=1)

        a = self.act_proj(state["act"])
        a_emb = self.act_layer_embedding(torch.arange(self.num_layers, device=device))
        a = a + a_emb.unsqueeze(0)
        a = self.act_branch(a)
        a = a.mean(dim=1)

        m = self.met_proj(state["metric"])
        m_emb = self.met_layer_embedding(torch.arange(self.num_clients, device=device))
        m = m + m_emb
        m = self.metric_branch(m)

        l = self.loss_proj(state["loss"])
        l_emb = self.loss_layer_embedding(torch.arange(self.num_clients, device=device))
        l = l + l_emb
        l = self.loss_branch(l)

        z = torch.cat([g, a, m, l], dim=-1)
        logits = self.head(z).view(-1, 2)

        bad_rows = torch.isnan(logits).any(dim=1) | torch.isinf(logits).any(dim=1)
        logits[bad_rows, 0] = 0.0
        logits[bad_rows, 1] = -1e9

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return logits, action


def create_state(data, act_cos, act_kl, grad_cos, grad_mag, metric_results,
                 loss_results, client_ids, layers, device):

    C = len(client_ids)
    L = len(layers)

    g = torch.zeros((C, L, 2), dtype=torch.float32, device=device)
    a = torch.zeros((C, L, 2), dtype=torch.float32, device=device)
    m = torch.zeros((C, 2), dtype=torch.float32, device=device)
    l = torch.zeros((C, 2), dtype=torch.float32, device=device)

    for cidx, client_i in enumerate(client_ids):
        for lidx, layer in enumerate(layers):
            gcf = grad_cos.get((client_i, layer))
            gmf = grad_mag.get((client_i, layer))
            g[cidx, lidx, :] = torch.tensor([gcf, gmf], dtype=torch.float32, device=device)

            acf = act_cos.get((client_i, layer))
            aklf = act_kl.get((client_i, layer))
            a[cidx, lidx, :] = torch.tensor([acf, aklf], dtype=torch.float32, device=device)

        mgm = metric_results.get("gm")
        mi = metric_results.get(client_i)
        m[cidx, :] = torch.tensor([mi, mgm - mi], dtype=torch.float32, device=device)

        lgm = loss_results.get("gm")
        li = loss_results.get(client_i)
        l[cidx, :] = torch.tensor([li, lgm - li], dtype=torch.float32, device=device)

    client_ids = torch.tensor(list(client_ids), device=device, dtype=torch.long)
    data.policy_inputs["grad"] = g
    data.policy_inputs["act"] = a
    data.policy_inputs["metric"] = m
    data.policy_inputs["loss"] = l
    data.policy_inputs["indexes"] = client_ids


def compute_eval(data):
    delta_acc = data.acc_after - data.acc_before["gm"]
    delta_auc = data.auc_after - data.auc_before["gm"]
    delta_loss = (data.loss_before["gm"] - data.loss_after) / data.loss_before["gm"]
    delta_loss = max(-1.0, min(1.0, delta_loss))
    return 0.5 * (delta_acc + delta_auc) + delta_loss


def compute_reward(data, action, device, wm=0.5):
    if isinstance(action, torch.Tensor) and action.numel() == 0:
        print("NO ACTION NO REWARD")
        return torch.tensor(0.0, device=device)

    else:
        gmg = data.grad_tracker["gm"]["postmerge"]
        order = data.policy_inputs["indexes"]
        updates = data.client_updates

        eval = compute_eval(data)

        gmg_flat = torch.cat([x.reshape(-1) for x in gmg.values()]).to(device)
        reward = torch.zeros(len(list(updates.keys())), dtype=torch.float32, device=device)
        # Each selected client's share of the selected clients' total update magnitude
        total_norm = sum(torch.norm(updates[cid], p=2).item() for cid in action)
        for idx, client_id in enumerate(order):
            cid = client_id.item()
            if cid in action:
                sign = torch.sign(torch.dot(gmg_flat, updates[cid]))
                norm = torch.norm(updates[cid], p=2).item() / total_norm if total_norm > 0 else 0.0
                reward[idx] = 10.0 * sign * norm * eval if eval < 0.0 else 10.0 * norm * eval
            else:
                reward[idx] = 0.0
        print(f"REWARD: {reward}")
        return reward


#REINFORCE-PPO
def update_policy(policy_model, optimizer, state,
                  policy_logits, merge_action,
                  reward):
    policy_model.train()
    device = next(policy_model.parameters()).device

    old_log_probs = torch.log_softmax(policy_logits, dim=1).detach().to(device)
    old_probs = torch.exp(old_log_probs).to(device)
    batch = torch.arange(old_log_probs.size(0), device=device)
    old_action = old_log_probs[batch, merge_action]

    last_loss = None
    last_entropy = None
    for _ in range(3):
        optimizer.zero_grad()
        logits = policy_model(state, device)[0]
        log_probs = torch.log_softmax(logits, dim=1).to(device)
        probs = torch.exp(log_probs).to(device)
        batch = torch.arange(log_probs.size(0), device=device)
        action = log_probs[batch, merge_action]

        # PPO ratio
        ratio = torch.exp(action - old_action)

        # Clipped surrogate
        surr1 = ratio * reward
        surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * reward
        policy_loss = -torch.min(surr1, surr2).mean()

        entropy = -(probs * log_probs).sum(dim=1).mean()

        loss = policy_loss - 0.2 * entropy
        loss.backward()

        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
        optimizer.step()

        last_loss = loss.detach()
        last_entropy = entropy.item()

    return last_loss, last_entropy


# VANILLA REINFORCE
def update_policy2(policy_model, optimizer, state,
                   policy_logits, merge_action,
                   reward):

    policy_model.train()
    optimizer.zero_grad()
    device = next(policy_model.parameters()).device

    log_probs = torch.log_softmax(policy_logits, dim=1).to(device)
    probs = torch.exp(log_probs).to(device)
    batch = torch.arange(log_probs.size(0), device=device)
    log_action = log_probs[batch, merge_action]
    action = probs[batch, merge_action]

    policy_loss = -(log_action * reward).mean()
    entropy = -(probs * log_probs).sum(dim=1).mean()

    loss = policy_loss - 0.2 * entropy
    loss.backward()
    optimizer.step()

    last_loss = loss.detach()
    last_entropy = entropy.item()
    return last_loss, last_entropy



import os
import torch
import feremm_utils as fu


def _save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)


def save_centralized_data(plotter, model_name, dataset):
    base = f"evaluation/saved_data/{model_name}/{dataset}/centralized"
    _save(plotter.test_pr, f"{base}_pr.pt")
    _save(plotter.test_re, f"{base}_re.pt")


def _save_federated_common(fw, plotter, framework, sbc):
    # Shared by FedAvg, FedProx and FeReMM
    base = f"evaluation/saved_data/{fw.model_name}/{fw.dataset}/{fw.num_clients}{sbc}{framework}"
    _save(plotter.client_distance, f"{base}_total_param_distances.pt")
    _save(plotter.client_validation_trajectory, f"{base}_validation_trajectory.pt")
    _save(plotter.client_training_trajectory, f"{base}_training_trajectory.pt")
    _save(plotter.test_pr, f"{base}_pr.pt")
    _save(plotter.test_re, f"{base}_re.pt")
    return base


def save_federated_data(fl, plotter, sbc):
    _save_federated_common(fl, plotter, fl.framework, sbc)


def save_feremm_data(fr, plotter, cdls, sbc):
    base = _save_federated_common(fr, plotter, "feremm", sbc)
    _save(plotter.policy_entropy, f"{base}_normal_entropy.pt")
    _save(plotter.policy_loss, f"{base}_normal_loss.pt")

    plotter.client_class_distribution = fu.client_class_distribution(cdls)
    plotter.client_image_features = fu.client_image_stats(cdls)
    _save(plotter.client_class_distribution, f"evaluation/saved_data/{fr.dataset}_{fr.num_clients}class_distribution.pt")
    _save(plotter.client_image_features, f"evaluation/saved_data/{fr.dataset}_{fr.num_clients}image_features.pt")

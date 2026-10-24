import load_resnet as lr
import feremm_policy as fp
import torch
import copy
import model_layers as ml


def load_model(model_name, dataset, device):
    model = lr.load_resnet(model_name, dataset)
    model = model.to(device)
    op = torch.optim.AdamW(params=model.parameters(), lr=1e-5, weight_decay=1e-4)
    return model, op


def load_federated(model_name, dataset, num_clients, framework, device):
    model = lr.load_resnet(model_name, dataset)
    if model_name == "resnet18":
        layers = ml.resnet18_layernames
    else:
        layers = ml.resnet50_layernames

    global_model = copy.deepcopy(model)
    global_op = torch.optim.AdamW(global_model.parameters(),
                                  lr=1e-5, weight_decay=1e-4)
    global_model.to(device)

    client_models = {}
    for i in range(num_clients):
        m = copy.deepcopy(model)
        m.to(device)
        client_models[i] = m
    del model

    client_ops = {}
    for i, m in client_models.items():
        client_ops[i] = torch.optim.AdamW(m.parameters(), lr=1e-5, weight_decay=1e-4)

    if framework == "feremm":
        global_policy = fp.FeReMMPolicy(len(layers), num_clients)
        global_policy.to(device)
        policy_op = torch.optim.AdamW(global_policy.parameters(), lr=1e-4, weight_decay=1e-4)
        return client_models, client_ops, global_policy, policy_op, global_model, global_op, layers
    return global_model, client_models, client_ops, layers



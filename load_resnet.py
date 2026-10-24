import torch
import torchvision.models as models
import torch.nn as nn


def load_resnet(model_name, dataset):
    if dataset == "breast_mnist":
        num_classes = 1
    elif dataset == "tissue_mnist":
        num_classes = 8
    elif dataset == "derma_mnist":
        num_classes = 7
    elif dataset == "oct_mnist":
        num_classes = 4
    else:
        num_classes = 5

    model = models.resnet50() if model_name == "resnet50" else models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

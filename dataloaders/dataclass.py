import numpy as np
import random
import torch
from . import medmnist_dataloader as medmnist


class DataClass():
    def __init__(self, dataset, framework,
                 num_clients=0, bad_client=False):

        self.dataset = dataset
        self.framework = framework
        self.num_clients = num_clients
        self.bad_client = bad_client

    def execute(self):
        if self.framework == "centralized":
            train_dl, val_dl, test_dl = medmnist.centralized_dataloaders(self.dataset)
        else:
            train_dl, val_dl, test_dl = medmnist.federated_dataloaders(self.dataset, self.num_clients, self.bad_client)
        return train_dl, val_dl, test_dl

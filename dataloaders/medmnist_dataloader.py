import numpy as np
import torch
import medmnist as med
from torchvision.transforms import v2
import os
from torch.utils.data import Dataset, DataLoader, Subset, \
ConcatDataset, random_split
from PIL import Image


# MedMNIST files are downloaded here on first use (not redistributed with this repo)
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".medmnist")
os.makedirs(PATH, exist_ok=True)

RGB_TRANSFORMS = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225])
])

GRAY_TRANSFORMS = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5],
                 std=[0.5, 0.5, 0.5])
])


class CustomDataset(Dataset):
    def __init__(self, base_dataset, transforms):
        self.ds = base_dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image, target = self.ds[idx]
        image = self.transforms(image)

        return {
            "image": image,
            "target": target,
        }


def load_retina():
    train = med.RetinaMNIST(split="train", size=224, root=PATH, download=True)
    val = med.RetinaMNIST(split="val", size=224, root=PATH, download=True)
    test = med.RetinaMNIST(split="test", size=224, root=PATH, download=True)
    return train, val, test


def load_tissue():
    train = med.TissueMNIST(split="train", size=28, root=PATH, download=True)
    val = med.TissueMNIST(split="val", size=28, root=PATH, download=True)
    test = med.TissueMNIST(split="test", size=28, root=PATH, download=True)
    return train, val, test


def load_derma():
    train = med.DermaMNIST(split="train", size=224, root=PATH, download=True)
    val = med.DermaMNIST(split="val", size=224, root=PATH, download=True)
    test = med.DermaMNIST(split="test", size=224, root=PATH, download=True)
    return train, val, test


def load_breast():
    train = med.BreastMNIST(split="train", size=224, root=PATH, as_rgb=True, download=True)
    val = med.BreastMNIST(split="val", size=224, root=PATH, as_rgb=True, download=True)
    test = med.BreastMNIST(split="test", size=224, root=PATH, as_rgb=True, download=True)
    return train, val, test


def load_oct():
    train = med.OCTMNIST(split="train", size=64, root=PATH, as_rgb=True, download=True)
    val = med.OCTMNIST(split="val", size=64, root=PATH, as_rgb=True, download=True)
    test = med.OCTMNIST(split="test", size=64, root=PATH, as_rgb=True, download=True)
    return train, val, test


def centralized_dataloaders(name, batch_size=64):
    if name == "retina_mnist":
        train_ds, val_ds, test_ds = load_retina()
        train_batch = len(train_ds) // 4
    elif name == "breast_mnist":
        train_ds, val_ds, test_ds = load_breast()
        train_batch = len(train_ds) // 4
    elif name == "derma_mnist":
        train_ds, val_ds, test_ds = load_derma()
        train_batch = len(train_ds) // 15
    elif name == "oct_mnist":
        train_ds, val_ds, test_ds = load_oct()
        train_batch = len(train_ds) // 15
    else:
        train_ds, val_ds, test_ds = load_tissue()
        train_batch = len(train_ds) // 20

    print(f"Train Dataset Samples: {len(train_ds)}")
    print(f"Validation Dataset Samples: {len(val_ds)}")
    print(f"Test Dataset Samples: {len(test_ds)}")

    t = GRAY_TRANSFORMS if name in ["breast_mnist", "oct_mnist"] else RGB_TRANSFORMS
    train_ds = CustomDataset(train_ds, t)
    val_ds = CustomDataset(val_ds, t)
    test_ds = CustomDataset(test_ds, t)

    train_dl = DataLoader(train_ds, batch_size=train_batch, shuffle=True,
                          pin_memory=True, drop_last=False, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        pin_memory=True, drop_last=False, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                         pin_memory=True, drop_last=False, num_workers=4)
    return train_dl, val_dl, test_dl


def federated_dataloaders(name, num_clients, bad_client, batch_size=64):
    if name == "retina_mnist":
        train_ds, val_ds, test_ds = load_retina()
    elif name == "breast_mnist":
        train_ds, val_ds, test_ds = load_breast()
    elif name == "derma_mnist":
        train_ds, val_ds, test_ds = load_derma()
    elif name == "oct_mnist":
        train_ds, val_ds, test_ds = load_oct()
    else:
        train_ds, val_ds, test_ds = load_tissue()

    t = GRAY_TRANSFORMS if name in ["breast_mnist", "oct_mnist"] else RGB_TRANSFORMS
    val_ds = CustomDataset(val_ds, t)
    test_ds = CustomDataset(test_ds, t)
    val_batch = len(val_ds) // 4

    val_loader = DataLoader(val_ds, batch_size=val_batch, shuffle=False,
                            pin_memory=True, drop_last=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                            pin_memory=True, drop_last=False, num_workers=4)

    indices = np.random.permutation(len(train_ds))
    partitions = np.array_split(indices, num_clients)

    client_loaders = []
    for i, files in enumerate(partitions):
        cds = Subset(train_ds, files)
        cds = CustomDataset(cds, t)
        client_batch = 5000 if name == "tissue_mnist" else len(cds) // 4
        cl = DataLoader(cds, batch_size=client_batch, shuffle=True,
                        pin_memory=True, drop_last=False, num_workers=4)

        client_loaders.append(cl)
        print(f"Client data len: {len(cl.dataset)}")
    return client_loaders, val_loader, test_loader


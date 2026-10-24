import os
import torch, random, numpy as np
from torchvision.utils import save_image
from torchvision.transforms import v2
import monai
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd,
    Orientationd, Spacingd,
    ScaleIntensityRanged, ToTensord,
    Resized, ScaleIntensityRangePercentilesd,
    ResizeWithPadOrCropd, EnsureTyped, NormalizeIntensityd
)
from monai.data import Dataset, DataLoader


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed set] {seed}")


def set_device():
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            device = torch.device("cuda:0")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def denormalize(images, d):
    if d == "gray":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    mean = torch.tensor(mean).view(1, -1, 1, 1)
    std = torch.tensor(std).view(1, -1, 1, 1)

    return images * std + mean



def save_dataloader_images(
    dataloader,
    save_dir="saved_images",
    max_images=50,
    d=""
):

    os.makedirs(save_dir, exist_ok=True)

    saved = 0

    for batch_idx, batch in enumerate(dataloader):
        # Handle datasets that return (image, label)
        if isinstance(batch, dict):
            images = batch["image"]
        elif isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch

        images = images.detach().cpu()

        images = denormalize(images, d)

        for i in range(images.size(0)):
            img = images[i]

            # If single channel, keep it valid for save_image
            if img.dim() == 2:
                img = img.unsqueeze(0)

            save_path = os.path.join(save_dir, f"img_{saved}.png")
            save_image(img, save_path)

            saved += 1
            if saved >= max_images:
                return



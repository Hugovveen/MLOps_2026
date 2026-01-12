from pathlib import Path
from typing import Dict, Tuple

from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import numpy as np

from .pcam import PCAMDataset

def get_dataloaders(config):
    data_cfg = config["data"]
    base = Path(data_cfg["data_path"])

    transform = transforms.ToTensor()

    train_ds = PCAMDataset(
        base / "camelyonpatch_level_2_split_train_x.h5",
        base / "camelyonpatch_level_2_split_train_y.h5",
        transform=transform,
        filter_data=True,
    )

    val_ds = PCAMDataset(
        base / "camelyonpatch_level_2_split_val_x.h5",
        base / "camelyonpatch_level_2_split_val_y.h5",
        transform=transform,
        filter_data=False,
    )

    labels = [train_dataset[i][1].item() for i in range(len(train_dataset))]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[l] for l in labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg["batch_size"],
        sampler=sampler,
        num_workers=data_cfg.get("num_workers", 0),
    )

    return train_loader, val_loader

from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from .pcam import PCAMDataset


<<<<<<< HEAD
def get_dataloaders(config: Dict) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Factory function to create Train and Validation DataLoaders
    using pre-split H5 files.

    Validation data is optional: if validation files are missing,
    the function will return (train_loader, None).
    """
    data_cfg = config["data"]
    base_path = Path(data_cfg["data_path"])

    batch_size = data_cfg["batch_size"]
    num_workers = data_cfg.get("num_workers", 0)


    transform = transforms.ToTensor()


    x_train = base_path / "camelyonpatch_level_2_split_train_x.h5"
    y_train = base_path / "camelyonpatch_level_2_split_train_y.h5"

    train_dataset = PCAMDataset(
        x_path=str(x_train),
        y_path=str(y_train),
        transform=transform,
        filter_data=True,
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
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
    )

    x_val = base_path / "camelyonpatch_level_2_split_val_x.h5"
    y_val = base_path / "camelyonpatch_level_2_split_val_y.h5"

    val_loader = None
    if x_val.exists() and y_val.exists():
        val_dataset = PCAMDataset(
            x_path=str(x_val),
            y_path=str(y_val),
            transform=transform,
            filter_data=False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
=======
def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    base_path = Path(data_cfg["data_path"])

    def create_loader(split: str, use_sampler: bool = False):
        x_p = str(base_path / f"camelyonpatch_level_2_split_{split}_x.h5")
        y_p = str(base_path / f"camelyonpatch_level_2_split_{split}_y.h5")

        # Using ToTensor handles the (C, H, W) conversion and scaling to [0, 1]
        ds = PCAMDataset(x_p, y_p, transform=transforms.ToTensor())

        sampler = None
        if use_sampler:
            # Flatten labels for weight calculation
            labels = ds.y_data[:].flatten()
            class_counts = np.bincount(labels)
            weights = 1.0 / class_counts[labels]
            sampler = WeightedRandomSampler(weights, len(weights))

        return DataLoader(
            ds,
            batch_size=data_cfg["batch_size"],
            sampler=sampler,
            num_workers=data_cfg.get("num_workers", 0),
            shuffle=(sampler is None),  # Shuffle only if not using sampler
        )

    train_loader = create_loader("train", use_sampler=True)
    val_loader = create_loader("valid", use_sampler=False)
>>>>>>> 64df295 (data class and loader minimal versions)

    return train_loader, val_loader

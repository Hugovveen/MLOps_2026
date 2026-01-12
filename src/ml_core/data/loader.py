from pathlib import Path
from typing import Dict, Tuple

from torch.utils.data import DataLoader
from torchvision import transforms

from .pcam import PCAMDataset


def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to create Train and Validation DataLoaders
    using pre-split H5 files.
    """
    data_cfg = config["data"]
    base_path = Path(data_cfg["data_path"])

    # -----------------------
    # Transforms
    # -----------------------
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # -----------------------
    # Paths
    # -----------------------
    x_train = base_path / "x_train.h5"
    y_train = base_path / "y_train.h5"

    x_val = base_path / "x_val.h5"
    y_val = base_path / "y_val.h5"

    # -----------------------
    # Datasets
    # -----------------------
    train_dataset = PCAMDataset(
        x_path=str(x_train),
        y_path=str(y_train),
        transform=train_transform,
    )

    val_dataset = PCAMDataset(
        x_path=str(x_val),
        y_path=str(y_val),
        transform=val_transform,
    )

    # -----------------------
    # DataLoaders
    # -----------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
    )

    return train_loader, val_loader

from pathlib import Path
from typing import Callable, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class PCAMDataset(Dataset):
    """
    PatchCamelyon (PCAM) Dataset reader for H5 format.
    """

    def __init__(
        self,
        x_path: str,
        y_path: str,
        transform: Optional[Callable] = None,
    ):
        self.x_path = Path(x_path)
        self.y_path = Path(y_path)
        self.transform = transform

        # 1. Check files exist
        if not self.x_path.exists():
            raise FileNotFoundError(f"X file not found: {self.x_path}")
        if not self.y_path.exists():
            raise FileNotFoundError(f"Y file not found: {self.y_path}")

        # 2. Read dataset length (without loading data)
        with h5py.File(self.x_path, "r") as f:
            self.length = f["x"].shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        with h5py.File(self.x_path, "r") as fx, h5py.File(self.y_path, "r") as fy:
            image = fx["x"][idx]
            label = fy["y"][idx]

        image = image.astype(np.uint8)

        # Apply transform if provided
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Default: convert to float tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)

        return image, label

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
        filter_data: bool = False,
    ):
        self.x_path = Path(x_path)
        self.y_path = Path(y_path)
        self.transform = transform
        self.filter_data = filter_data

        if not self.x_path.exists() or not self.y_path.exists():
            raise FileNotFoundError(f"Missing dataset files: {self.x_path}, {self.y_path}")

        with h5py.File(self.x_path, "r") as fx, h5py.File(self.y_path, "r") as fy:
            labels = fy["y"][:].astype(int)
            indices = np.arange(len(labels))

            if self.filter_data:
                images = fx["x"][:]
                means = images.mean(axis=(1, 2, 3))
                mask = (means > 10) & (means < 245)
                indices = indices[mask]

        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        real_idx = int(self.indices[idx])

        with h5py.File(self.x_path, "r") as fx, h5py.File(self.y_path, "r") as fy:
            image = fx["x"][real_idx]
            label = int(fy["y"][real_idx])  # <-- critical fix

        image = np.clip(image, 0, 255).astype(np.uint8)

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        label = torch.tensor(label, dtype=torch.long)

        return image, label

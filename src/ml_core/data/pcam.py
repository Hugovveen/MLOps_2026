from typing import Callable, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class PCAMDataset(Dataset):
<<<<<<< HEAD
    """
    PatchCamelyon (PCAM) Dataset reader for H5 format.
    """

=======
>>>>>>> 64df295 (data class and loader minimal versions)
    def __init__(
        self,
        x_path: str,
        y_path: str,
        transform: Optional[Callable] = None,
        filter_data: bool = False,
    ):
<<<<<<< HEAD
        self.x_path = Path(x_path)
        self.y_path = Path(y_path)
=======
        self.x_data = h5py.File(x_path, "r")["x"]
        self.y_data = h5py.File(y_path, "r")["y"]
>>>>>>> 64df295 (data class and loader minimal versions)
        self.transform = transform
        self.filter_data = filter_data

<<<<<<< HEAD
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
=======
        # Initialize indices for filtering
        self.indices = np.arange(len(self.x_data))

        if filter_data:
            valid_indices = []
            for i in range(len(self.x_data)):
                # Heuristic: Drop blackouts (0) and washouts (255)
                mean_val = np.mean(self.x_data[i])
                if 0 < mean_val < 255:
                    valid_indices.append(i)
            self.indices = np.array(valid_indices)
>>>>>>> 64df295 (data class and loader minimal versions)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
<<<<<<< HEAD
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
=======
        real_idx = self.indices[idx]
        img = self.x_data[real_idx]
        label = self.y_data[real_idx].item()

        # Handle NaNs explicitly before clipping/casting
        # This replaces NaNs with 0.0 (black)
        img = np.nan_to_num(img, nan=0.0)

        # Numerical Stability: Clip before uint8 cast
        img = np.clip(img, 0, 255).astype(np.uint8)

        if self.transform:
            img = self.transform(img)
        else:
            # Basic conversion if no transform provided
            img = torch.from_numpy(img).permute(2, 0, 1).float()

        return img, torch.tensor(label, dtype=torch.long)
>>>>>>> 64df295 (data class and loader minimal versions)

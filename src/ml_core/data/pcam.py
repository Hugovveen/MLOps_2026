class PCAMDataset(Dataset):
    def __init__(
        self,
        x_path: str,
        y_path: str,
        transform=None,
        filter_data: bool = False,
    ):
        self.x_path = Path(x_path)
        self.y_path = Path(y_path)
        self.transform = transform
        self.filter_data = filter_data

        if not self.x_path.exists() or not self.y_path.exists():
            raise FileNotFoundError

        with h5py.File(self.x_path, "r") as fx, h5py.File(self.y_path, "r") as fy:
            self.images = fx["x"]
            self.labels = fy["y"][:].astype(int)

            self.indices = np.arange(len(self.labels))

            if self.filter_data:
                means = self.images[:].mean(axis=(1, 2, 3))
                mask = (means > 10) & (means < 245)
                self.indices = self.indices[mask]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        with h5py.File(self.x_path, "r") as fx, h5py.File(self.y_path, "r") as fy:
            image = fx["x"][real_idx]
            label = fy["y"][real_idx]

        image = np.clip(image, 0, 255).astype(np.uint8)

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, torch.tensor(label, dtype=torch.long)

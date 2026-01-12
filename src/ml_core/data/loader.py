from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

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

    labels = [train_ds[i][1].item() for i in range(len(train_ds))]
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts
    sample_weights = [weights[l] for l in labels]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg["batch_size"],
        sampler=sampler,
        num_workers=data_cfg.get("num_workers", 0),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 0),
    )

    return train_loader, val_loader

import argparse
import csv
import os

import torch
import torch.optim as optim
from ml_core.data import get_dataloaders
from ml_core.models import MLP
from ml_core.solver import Trainer
from ml_core.utils import load_yaml_config
from ml_core.utils.logging import seed_everything
from torch.optim.lr_scheduler import StepLR

# logger = setup_logger("Experiment_Runner")


def main(args):
    # 1. Load Config & Set Seed
    config = load_yaml_config(args.config)
    seed = int(config.get("seed", 42)) if args.seed is None else int(args.seed)
    seed_everything(seed)
    print(f"Using seed: {seed}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    train_loader, val_loader = get_dataloaders(config)
    print("loaded data")
    model = MLP(**config["model"])

    optimizer = optim.SGD(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )

    scheduler = StepLR(
        optimizer,
        step_size=1,
        gamma=0.9,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=config,
        device=device,
        scheduler=scheduler,
    )
    # Make a folder for each seperate seed
    trainer.fit(train_loader, val_loader)
    out_dir = f"outputs/q4_seed_{seed}"
    os.makedirs(out_dir, exist_ok=True)

    # Save gradient norms (per step)
    with open(os.path.join(out_dir, "grad_norms.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "grad_norm"])
        for i, g in enumerate(trainer.grad_norm_history):
            w.writerow([i, g])

    # Save LR (per epoch)
    with open(os.path.join(out_dir, "lr_per_epoch.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "lr"])
        for i, lr in enumerate(trainer.lr_history):
            w.writerow([i + 1, lr])

    print(f"saved Q4 outputs to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Simple MLP on PCAM")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument(
        "--seed", type=int, default=None, help="Override seed from config"
    )

    args = parser.parse_args()
    main(args)

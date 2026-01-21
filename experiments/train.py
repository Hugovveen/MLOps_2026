import argparse
import csv
import os
import sys

def assert_in_venv():
    # Works for venv, virtualenv, conda
    if sys.prefix == sys.base_prefix:
        raise RuntimeError(
            "Not running inside a virtual environment. "
            "Activate the correct venv before running this script."
        )

assert_in_venv()

def assert_correct_venv(expected_name="my_venv"):
    venv_path = os.environ.get("VIRTUAL_ENV")
    if venv_path is None or expected_name not in venv_path:
        raise RuntimeError(
            f"Expected virtual environment '{expected_name}', "
            f"but VIRTUAL_ENV={venv_path}"
        )

assert_correct_venv("my_venv")

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from ml_core.data import get_dataloaders
from ml_core.models import MLP
from ml_core.solver import Trainer
from ml_core.utils import load_yaml_config
from ml_core.utils.logging import seed_everything
from ml_core.utils.tracker import WandBTracker


def main(args):
    # 1. Load config & set seed
    config = load_yaml_config(args.config)
    seed = int(config.get("seed", 42)) if args.seed is None else int(args.seed)
    seed_everything(seed)
    print(f"Using seed: {seed}")

    # 2. Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 3. Init W&B tracker (AFTER device exists)
    tracker = WandBTracker(
        project="mlops-assignment-2",
        config={
            **config,
            "seed": seed,
            "device": device,
        },
        run_name=f"seed-{seed}",
    )

    # 4. Data
    train_loader, val_loader = get_dataloaders(config)
    print("Loaded data")

    # 5. Model & optimizer
    model = MLP(**config["model"])
    optimizer = optim.SGD(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

    # 6. Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=config,
        device=device,
        scheduler=scheduler,
    )

    # 7. Train
    trainer.fit(train_loader, val_loader)

    # 8. Log best validation metrics + checkpoint
    if trainer.best_val_metrics is not None:
        tracker.log_metrics(trainer.best_val_metrics)

    tracker.log_checkpoint("experiments/checkpoints/best.pt")
    tracker.finish()

    # 9. (Optional) Q4 outputs
    out_dir = f"outputs/q4_seed_{seed}"
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "grad_norms.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "grad_norm"])
        for i, g in enumerate(trainer.grad_norm_history):
            w.writerow([i, g])

    with open(os.path.join(out_dir, "lr_per_epoch.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "lr"])
        for i, lr in enumerate(trainer.lr_history):
            w.writerow([i + 1, lr])

    print(f"Saved Q4 outputs to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Simple MLP on PCAM")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    args = parser.parse_args()
    main(args)

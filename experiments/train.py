from torch.optim.lr_scheduler import StepLR
from ml_core.utils.logging import seed_everything
from ml_core.utils import load_yaml_config
import argparse
import torch
import torch.optim as optim
from ml_core.data import get_dataloaders
from ml_core.models import MLP
from ml_core.solver import Trainer

# logger = setup_logger("Experiment_Runner")

def main(args):
    # 1. Load Config & Set Seed
    config = load_yaml_config(args.config)
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    print(f"Using seed: {seed}")
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print (f"using device: {device}")

    train_loader, val_loader = get_dataloaders(config)

    model = MLP(**config["model"])

    optimizer = optim.SGD(model.parameters(),
        lr = config['training']['learning_rate'],)

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

    trainer.fit(train_loader, val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Simple MLP on PCAM")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()

    main(args)

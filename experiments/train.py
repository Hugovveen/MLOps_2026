from ml_core.utils import load_yaml_config
import argparse
import torch
import torch.optim as optim
# from ml_core.data import get_dataloaders
# from ml_core.models import MLP
# from ml_core.solver import Trainer
# from ml_core.utils import load_config, seed_everything, setup_logger

# logger = setup_logger("Experiment_Runner")

def main(args):
    # 1. Load Config & Set Seed
    config = load_yaml_config(args.config)

    print("Loaded config:", args.config)
    print("seed:", config.get("seed"))
    print("data.batch_size:", config["data"]["batch_size"])
    print("training.epochs:", config["training"]["epochs"])
    print("training.learning_rate:", config["training"]["learning_rate"])
    
    # 2. Setup Device
    
    # 3. Data
    # train_loader, val_loader = get_dataloaders(config)
    
    # 4. Model
    # model = MLP(...)
    
    # 5. Optimizer
    # optimizer = optim.SGD(...)
    
    # 6. Trainer & Fit
    # trainer = Trainer(...)
    # trainer.fit(...)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Simple MLP on PCAM")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()

    main(args)

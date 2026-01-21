import argparse
import time

import torch

from ml_core.data import get_dataloaders
from ml_core.models import MLP
from ml_core.utils.config import load_yaml_config
from ml_core.utils.logging import seed_everything


def measure_throughput(dataloader, model, device, num_batches=100, warmup_batches=10):
    model.eval()
    it = iter(dataloader)

    for _ in range(warmup_batches):
        x, _ = next(it)
        x = x.to(device, non_blocking=True)
        with torch.no_grad():
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    images = 0

    for _ in range(num_batches):
        x, _ = next(it)
        bs = x.shape[0]
        x = x.to(device, non_blocking=True)
        with torch.no_grad():
            _ = model(x)
        images += bs

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()
    seconds = end - start
    return images / seconds


def main(args):
    config = load_yaml_config(args.config)

    if args.batch_size is not None:
        config["data"]["batch_size"] = int(args.batch_size)

    seed = int(config.get("seed", 42)) if args.seed is None else int(args.seed)
    seed_everything(seed)

    device_str = args.device
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print(f"seed={seed}")
    print(f"device={device.type}")
    print(f"batch_size={config['data']['batch_size']}")

    if device.type == "cuda":
        print(f"gpu_name={torch.cuda.get_device_name(0)}")

    train_loader, _ = get_dataloaders(config)
    model = MLP(**config["model"]).to(device)

    img_per_s = measure_throughput(
        dataloader=train_loader,
        model=model,
        device=device,
        num_batches=args.num_batches,
        warmup_batches=args.warmup_batches,
    )
    print(f"throughput_img_per_s={img_per_s:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark throughput (images/sec)")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--seed", type=int, default=None, help="Override seed from config")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Force device, otherwise auto-detect",
    )
    parser.add_argument("--num_batches", type=int, default=100)
    parser.add_argument("--warmup_batches", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size from config")
    args = parser.parse_args()
    main(args)

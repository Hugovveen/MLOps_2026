from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# from ml_core.utils.metrics import compute_fbeta, compute_pr_auc
from ml_core.metrics import compute_fbeta, compute_pr_auc

from pathlib import Path

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
        device: str,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.scheduler = scheduler

        # Loss function for classification with logits (MLP outputs 2 logits)
        self.criterion = nn.CrossEntropyLoss()

        # Storage for Q4
        self.grad_norm_history: List[float] = []  # per step global grad norm
        self.lr_history: List[float] = []  # per epoch LR

    def _current_lr(self) -> float:
        # Most optimizers have one param_group; if multiple, take the first.
        return float(self.optimizer.param_groups[0]["lr"])

    def train_epoch(
        self, dataloader: DataLoader, epoch_idx: int
    ) -> Tuple[float, float, float]:
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        log_every = int(self.config["training"].get("log_every_steps", 100))

        for step_idx, (images, labels) in enumerate(
            tqdm(dataloader, desc=f"Train epoch {epoch_idx+1}")
        ):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()

            # Compute L2 norm over all parameter gradients that exist.
            total_norm_sq = 0.0
            for p in self.model.parameters():
                if p.grad is None:
                    continue
                param_norm = p.grad.data.norm(2)
                total_norm_sq += float(param_norm) ** 2
            grad_norm = total_norm_sq**0.5
            self.grad_norm_history.append(grad_norm)

            self.optimizer.step()

            running_loss += float(loss.item())

            preds = outputs.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())

            if log_every > 0 and step_idx % log_every == 0:
                acc = correct / max(total, 1)
                print(
                    f"[Train] epoch={epoch_idx+1} step={step_idx} loss={loss.item():.4f} acc={acc:.4f} grad_norm={grad_norm:.4f}"
                )

        avg_loss = running_loss / max(len(dataloader), 1)
        acc = correct / max(total, 1)

        f1_placeholder = 0.0
        return avg_loss, acc, f1_placeholder

    def validate(
        self, dataloader: DataLoader, epoch_idx: int
    ) -> Tuple[float, float, float]:
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_probs = []
        all_preds = []

        if dataloader is None:
            # No validation available
            return 0.0, 0.0, 0.0

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc=f"Val epoch {epoch_idx+1}"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += float(loss.item())

                probs = F.softmax(outputs, dim=1)[:, 1]  # probability of positive class
                preds = outputs.argmax(dim=1)

                all_labels.append(labels.cpu())
                all_probs.append(probs.cpu())
                all_preds.append(preds.cpu())

                correct += int((preds == labels).sum().item())
                total += int(labels.numel())

        avg_loss = running_loss / max(len(dataloader), 1)
        acc = correct / max(total, 1)

        y_true = torch.cat(all_labels).numpy()
        y_prob = torch.cat(all_probs).numpy()
        y_pred = torch.cat(all_preds).numpy()

        fbeta = compute_fbeta(y_true, y_pred, beta=1.5)
        pr_auc = compute_pr_auc(y_true, y_prob)
        return avg_loss, acc, fbeta, pr_auc

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        val_metrics: dict,
    ):
        checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_metrics": val_metrics,
        }
        torch.save(checkpoint, path)
    def load_checkpoint(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint


    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader]) -> None:
        epochs = int(self.config["training"]["epochs"])
        print(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            # log LR once per epoch (Q4b requirement)
            current_lr = self._current_lr()
            self.lr_history.append(current_lr)
            print(f"[LR] epoch={epoch+1} lr={current_lr}")

            train_loss, train_acc, _ = self.train_epoch(train_loader, epoch)
            val_loss, val_acc, val_fbeta, val_pr_auc = (
                self.validate(val_loader, epoch)
                if val_loader is not None
                else (0.0, 0.0, 0.0, 0.0)
            )
            print(
                f"--- Epoch {epoch+1} Summary: "
                f"Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f} | "
                f"Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}, "
                f"Val Fβ {val_fbeta:.4f}, Val PR-AUC {val_pr_auc:.4f} ---"
            )

            # Scheduler stepping: default to per epoch schedulers
            if self.scheduler is not None:
                self.scheduler.step()

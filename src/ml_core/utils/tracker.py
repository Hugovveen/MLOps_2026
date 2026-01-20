import csv
from pathlib import Path
from typing import Any, Dict

# TODO: Add TensorBoard Support


class ExperimentTracker:
    def __init__(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        base_dir: str = "experiments/results",
    ):
        self.run_dir = Path(base_dir) / experiment_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # TODO: Save config to yaml in run_dir

        # Initialize CSV
        self.csv_path = self.run_dir / "metrics.csv"
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        # Header (TODO: add the rest of things we want to track, loss, gradients, accuracy etc.)
        self.csv_writer.writerow(["epoch"])

    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """
        Writes metrics to CSV (and TensorBoard).
        """
        # TODO: Write other useful metrics to CSV
        self.csv_writer.writerow([epoch])  # Currently only logging epoch
        self.csv_file.flush()

        # TODO: Log to TensorBoard

    def get_checkpoint_path(self, filename: str) -> str:
        return str(self.run_dir / filename)

    def close(self):
        self.csv_file.close()

import wandb
from typing import Optional


class WandBTracker:
    def __init__(
        self,
        project: str,
        config: Dict[str, Any],
        run_name: Optional[str] = None,
    ):
        self.run = wandb.init(
            project=project,
            config=config,
            name=run_name,
        )

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        wandb.log(metrics, step=step)

    def log_checkpoint(self, path: str | Path):
        artifact = wandb.Artifact(
            name="model-checkpoint",
            type="model",
        )
        artifact.add_file(str(path))
        wandb.log_artifact(artifact)

    def finish(self):
        wandb.finish()

# MLOps UvA Bachelor AI Course: Medical Image Classification

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Build Status](https://github.com/Hugovveen/MLOps_2026/actions/workflows/ci.yml)
![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)

A repo exemplifying **MLOps best practices**: modularity, reproducibility, automation, and experiment tracking.

This project implements a standardized workflow for training neural networks on medical data (PCAM/TCGA). 

---

## 1. Installation
This project requires Python 3.10 or higher.

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install the project and dependencies
pip install -e .
``` 
---

## 2. Data Setup

This repository does not include the dataset itself.
To run the project, the dataset must be placed at the path specified in the configuration file, or the configuration must be updated accordingly.


The dataset configuration is defined in:

```text
experiments/configs/train_config.yaml
``` 

#### Default dataset settings

The following values are the default configuration used in this project.
They can be modified by editing the YAML configuration file.

```yaml
dataset_type: pcam
data_path: ./data/camelyonpatch_level_2
input_shape: [3, 96, 96]
batch_size: 32
num_workers: 2
``` 

These defaults define how the dataset is expected to be loaded and processed when running the training script with the provided configuration.

#### Required dataset files

The exact required dataset files and file format depend on the dataloader implementation.

TODO:
- Confirm the exact dataset file format used by the dataloader (e.g. HDF5).
- Confirm the required train, validation, and test split files.
- Document the dataset source and download procedure used in this project (Q1–Q2). 

---

## 3. Training

Training is performed using the script:

```text
experiments/train.py
``` 

The training pipeline is configured via a YAML configuration file.

```text
experiments/configs/train_config.yaml
``` 

Run command:

```bash
python experiments/train.py --config experiments/configs/train_config.yaml
``` 

TODO
- Confirm the exact command used for the final training run
- Confirm whether additional command-line arguments are used

#### Default training settings

```yaml
seed: 42
training:
  epochs: 5
  learning_rate: 0.001
  save_dir: ./experiments/results
``` 

TODO
- Confirm whether these defaults are used for the final run
- Document any configuration overrides

#### Artifacts

Default output directory:

```text
./experiments/results
``` 

TODO
- Document which artifacts are stored in this directory
- Document the exact checkpoint file path
- Document logging and metrics outputs

---

## 4. Inference
This section describes how to run a single prediction using a trained model checkpoint.

Inference entrypoint (to be finalized)
TODO add a dedicated inference.py or a notebook that loads the selected best checkpoint and runs a prediction on a single sample image.

Model checkpoint (to be finalized)
TODO add the selected best model checkpoint file (.pt or .pth) to the repository.
TODO document the final checkpoint path used for inference.

Example ousage (to be finalized)
TODO provide the exact command to run inference once inference.py or the notebook is available.
TODO include the path to a sample image used for the example.

---

## 📂 Project Structure
---

```text
.
├── src/ml_core/          # The Source Code (Library)
│   ├── data/             # Data loaders and transformations
│   ├── models/           # PyTorch model architectures
│   ├── solver/           # Trainer class and loops
│   └── utils/            # Loggers and experiment trackers
├── experiments/          # The Laboratory
│   ├── configs/          # YAML files for hyperparameters
│   ├── results/          # Checkpoints and logs (Auto-generated)
│   └── train.py          # Entry point for training
├── scripts/              # Helper scripts (plotting, etc)
├── tests/                # Unit tests for QA
├── pyproject.toml        # Config for Tools (Ruff, Pytest)
└── setup.py              # Package installation script
```

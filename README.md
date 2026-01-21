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
- The used dataset file format was HDF5
- Required train, validation and test split files:
    camelyonpatch_level_2_split_train_x.h5
    camelyonpatch_level_2_split_train_y.h5
    camelyonpatch_level_2_split_valid_x.h5
    camelyonpatch_level_2_split_valid_y.h5
- Dataset source: https://surfdrive.surf.nl/s/wjRYtSborgbPF2P
- Download to snellius with scp

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

#### Default training settings

```yaml
seed: 42
training:
  epochs: 5
  learning_rate: 0.001
  save_dir: ./experiments/results
```

#### Artifacts

Default output directory:

```text
./experiments/results
```
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

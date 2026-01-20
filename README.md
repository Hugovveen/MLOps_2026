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

This project uses the PCAM dataset provided as HDF5 files.

The dataset is not included in this repository.

The data directory is defined in the YAML config under: data.data_path
Default in this repo: ./data/camelyonpatch_level_2

All PCAM H5 files must be placed inside this directory.

Required files for training:

```bash
camelyonpatch_level_2_split_train_x.h5

camelyonpatch_level_2_split_train_y.h5
``` 

Optional validation files:

```bash
camelyonpatch_level_2_split_val_x.h5

camelyonpatch_level_2_split_val_y.h5
``` 

Validation is optional. If the validation .h5 files are not present, the loader returns no validation set and training runs without a validation DataLoader.

Example directory layout:

```bash
<DATA_PATH>/
camelyonpatch_level_2_split_train_x.h5
camelyonpatch_level_2_split_train_y.h5
camelyonpatch_level_2_split_val_x.h5
camelyonpatch_level_2_split_val_y.h5
``` 

Ensure that the directory structure matches exactly, as the dataset loader expects this layout

TODO:

Add test H5 files to this section if test evaluation is implemented later.

Update this section if the data_path configuration key or default value changes.

Verify and document the exact dataset location used for the final training run.

---

## 3. Training

Training is performed using the script:

```bash
experiments/train.py
``` 

The training pipeline is configured via a YAML configuration file.

Configuration template:

```bash
experiments/configs/train_config.yaml
``` 

Training command (to be finalized)
TODO exact command used to run experiments/train.py for the best model
TODO exact configuration file path
TODO random seed used

Expected performance (to be finalized)

TODO validation metric and value
TODO test metric and value

Artifacts (to be finalized)

TODO path to saved model checkpoint (.pt or .pth)
TODO path to logs and experiment outputs

---

## 4. Inference
This section describes how to run a single prediction using a trained model checkpoint.

Inference entrypoint (to be finalized)
TODO add a dedicated inference.py or a notebook that loads the selected best checkpoint and runs a prediction on a single sample image.

Model checkpoint (to be finalized)
TODO add the selected best model checkpoint file (.pt or .pth) to the repository.
TODO document the final checkpoint path used for inference.

Example usage (to be finalized)
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

# MLOps UvA Bachelor AI Course: Medical Image Classification Skeleton Code

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Build Status](https://github.com/yourusername/mlops_course/actions/workflows/ci.yml/badge.svg)
![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)

A repo exemplifying **MLOps best practices**: modularity, reproducibility, automation, and experiment tracking.

This project implements a standardized workflow for training neural networks on medical data (PCAM/TCGA). 

The idea is that you fill in the repository with the necessary functions so you can execute the ```train.py``` function. Please also fill in this ```README.md``` clearly to setup, install and run your code. 

Don't forget to setup CI and linting!

---

## 1. Installation
Clone the repository and set up your isolated environment.

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install the package in "Editable" mode
pip install -e .

# 3. Install pre-commit hooks
pre-commit install
```

### 2. Verify Setup
```bash
pytest tests/
```

### 3. Run an Experiment
```bash
python experiments/train.py --config experiments/configs/train_config.yaml
```
---

## 2. Data Setup
```
This project uses the PCAM dataset provided as HDF5 files.

The dataset is not included in this repository.

The data directory is defined via the configuration file under the key:
bash ```data.data_path``` 

All PCAM H5 files must be placed inside this directory.

The following files are required:

```bash
camelyonpatch_level_2_split_train_x.h5
camelyonpatch_level_2_split_train_y.h5
camelyonpatch_level_2_split_val_x.h5
camelyonpatch_level_2_split_val_y.h5
``` 

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

Validation is optional. If the validation files are not present, the training pipeline will run without a validation DataLoader.

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
```bash
# TODO exact command
# TODO expected val/test performance
# TODO config file
```
---

## 4. Inference
```bash
# TODO inference script
# TODO checkpoint path
# TODO example command
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

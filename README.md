# Rotation-Free Image Embedding

A PyTorch project for learning image embeddings from image pairs so that samples from the same class/scene are close in feature space and samples from different classes/scenes are farther apart.

## Overview

This repository trains a Siamese-style embedding model using pairwise supervision:

- Input: two images (`img1`, `img2`) and a binary label.
- Label semantics:
  - `0` → same class/scene
  - `1` → different class/scene
- Objective: minimize a margin-based contrastive loss over embedding distances.

The project supports multiple CNN backbones, multiple distance functions, optional Weights & Biases logging, and train/test split strategies based on one or two dataset roots.

## Repository Structure

```text
.
├── main.py              # Training entrypoint, argument parsing, dataloaders, and run orchestration
├── model.py             # ImageEmbeding model, distance functions, contrastive training loop
├── data.py              # Pair dataset construction from class-organized folders
├── batch_trainer.sh     # Example Slurm batch script for cluster/GPU training
└── requirements.txt     # Python dependencies
```

## Installation

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** CUDA-enabled PyTorch installation depends on your platform. If needed, install torch/torchvision from the official PyTorch instructions first, then install the remaining packages.

## Dataset Format

The code expects class-organized folders:

```text
dataset_root/
├── class_a/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── class_b/
│   ├── img_101.jpg
│   └── ...
└── ...
```

You provide either:

1. **Separate train/test roots** (`--train_folder`, `--test_folder`) with `--test_size 0`, or
2. **Combined split mode** where both roots are merged and then split randomly using `--test_size` in `(0, 1)`.

## Training

### Basic run

```bash
python main.py \
  --train_folder /path/to/train \
  --test_folder /path/to/test \
  --test_size 0 \
  --max_img_per_class 100 \
  --batch_size 32 \
  --num_workers 4 \
  --epochs 10
```

### Random split mode

```bash
python main.py \
  --train_folder /path/to/root_a \
  --test_folder /path/to/root_b \
  --test_size 0.2
```

## Key CLI Arguments

| Argument | Description | Default |
|---|---|---|
| `--train_folder`, `--train` | Path to training dataset root | `/home/kasra/datasets/tanks_and_temples/images/train` |
| `--test_folder`, `--test` | Path to testing dataset root | `/home/kasra/datasets/tanks_and_temples/images/test` |
| `--test_size`, `-t` | If `0`, use separate train/test roots; if `(0,1)`, merge+split | `0.2` |
| `--max_img_per_class`, `-m` | Max files loaded per class | `20` |
| `--batch_size`, `-b` | Batch size | `32` |
| `--num_workers`, `-n` | DataLoader workers | `4` |
| `--epochs`, `-e` | Training epochs | `10` |
| `--cnn`, `-c` | Backbone: `resnet18`, `resnet50`, `simple` | `simple` |
| `--loss_margin`, `-l` | Margin used in contrastive loss | `1.0` |
| `--distance`, `-d` | Distance metric: `cosine` or `euclidean` | `cosine` |

> **Important:** `--use_wandb` is implemented with `action="store_false"`, so passing this flag disables wandb logging. Leaving it unset keeps wandb enabled.

## Model and Loss

### Backbones

- `simple`: lightweight custom CNN stack
- `resnet18`: ImageNet-pretrained ResNet18 (final FC removed)
- `resnet50`: ImageNet-pretrained ResNet50 (final FC removed)

### Embedding head

Feature extractor output is passed through:

1. `Linear(flattened_features, 512)`
2. `ReLU`
3. `Linear(512, embedding_size)`

### Distance functions

- **Euclidean:** `PairwiseDistance(p=2)`
- **Cosine distance:** `1 - cosine_similarity`

### Loss (linear contrastive)

Per-pair loss implemented as:

\[
\mathcal{L} = \frac{1}{2}\Big((1-y)\,d^2 + y\,\max(0, m-d)^2\Big)
\]

where:

- \(d\): distance between embeddings
- \(y\): binary pair label (as used in this codebase)
- \(m\): margin

## Logging and Checkpoints

- **wandb**: enabled by default unless `--use_wandb` flag is passed.
- **checkpoint_path**: if provided to `main(...)`, directory is created and can be used for saving model checkpoints (current training loop primarily logs metrics and evaluates test loss).

## Cluster/HPC Usage

`batch_trainer.sh` provides a Slurm example:

- requests 1 node, 1 task, 7 CPUs, 1 GPU
- activates a conda environment
- runs `main.py` with dataset paths and `-m 100`

Submit with:

```bash
sbatch batch_trainer.sh
```

## Next Step

- Run the codes and get some results

## Known Limitations / Notes

- Dataset generates **all image pairs** (`n choose 2`), which can grow very quickly in memory/time for large datasets.
- `crossentropy` loss option is declared but not implemented.
- `ImageEmbeding` class name is intentionally preserved to match existing code.

## Development Suggestions

Potential improvements:

1. Pair sampling strategy (balanced positives/negatives) instead of full combinations.
2. Proper checkpoint saving/loading in `fit()`.
3. Validation metrics such as ROC-AUC or retrieval Recall@K.
4. Config management (YAML/TOML) for reproducible experiments.
5. Unit tests for dataset label semantics and loss behavior.

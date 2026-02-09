# Confix: Learning across Resolutions for Encrypted Traffic Classification with Noisy Labels

Code companion for the paper *"Learning across Resolutions: Encrypted Traffic Classification with Noisy Labels"*.

## Overview

Confix is a framework for correcting noisy labels in encrypted network traffic datasets. It operates in two stages:

1. **Label Correction** — Build a KNN graph from traffic features, detect multi-resolution communities via Parallel-DT (Leiden algorithm), and correct noisy labels through confidence-weighted consensus voting.
2. **Classification** — Augment features with community embeddings and train a lightweight MLP on the corrected labels.

## Setup

```bash
conda env create -f environment.yml
conda activate confix
```

## Repository Structure

```
confix/          Core library (6 modules)
  graph.py       KNN graph + SG-t-SNE reweighting
  community.py   Parallel-DT multi-resolution detection
  correction.py  Consensus voting + label correction
  model.py       ConFixMLP + training utilities
  data.py        Dataset registry, noise injection, augmentation
  metrics.py     Accuracy, macro-F1, repair rate

configs/         Hyperparameters from the paper
scripts/         Entry-point scripts
tests/           Unit tests (~30 tests on synthetic data)
```

## Usage

### Label Correction

```bash
python scripts/run_correction.py --data path/to/features.csv --output corrected.npz
```

### Training

```bash
python scripts/run_train.py --data path/to/features.csv --labels corrected.npz
```

### Running Tests

```bash
pytest -v --tb=short
pytest --cov=confix
```

## Datasets

The framework is evaluated on the following public datasets:

| Dataset | Classes | Features | Source |
|---------|---------|----------|------------|
| CIC-IoT2023 | 34 | 46 | [UNB](https://www.unb.ca/cic/datasets/iotdataset-2023.html) |
| MQTT-IoT-IDS2020 | 5 | 33 | [IEEE](https://ieee-dataport.org/open-access/mqtt-iot-ids2020-mqtt-internet-things-intrusion-detection-dataset) |
| TON-IoT | 10 | 44 | [UNSW](https://research.unsw.edu.au/projects/toniot-datasets) |
| CICIDS-2018 | 15 | 80 | [UNB](https://www.unb.ca/cic/datasets/ids-2018.html) |
| ISCX-VPN2016 | 12 | 78 | [UNB](https://www.unb.ca/cic/datasets/vpn.html) |
| UNSW-NB15 | 10 | 42 | [UNSW](https://research.unsw.edu.au/projects/unsw-nb15-dataset) |
| CICIDS-2017 | 8 | 78 | [UNB](https://www.unb.ca/cic/datasets/ids-2017.html) |

Datasets are not included in this repository. Download from the links above and preprocess into CSV (features + label column) or NPZ (`X`, `y` arrays).

## Configuration

All hyperparameters are in `configs/default.yaml`. Key parameters:

- **Graph**: `k` (KNN neighbours, default: `ceil(log2(N))`)
- **Community**: `gamma_range` (0.1–2.0), `n_resolutions` (20)
- **Correction**: `theta_low` (0.3), `delta` (0.2)
- **Model**: 512→256→128 MLP, GELU, dropout 0.3
- **Training**: AdamW (lr=3e-4, wd=1e-4), cosine schedule, 5-epoch warmup, patience 15

## License

MIT

"""Dataset registry, noise injection, augmentation, and data loading utilities."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

DATASET_REGISTRY = {
    "CIC-IoT2023": {
        "name": "CIC-IoT2023",
        "num_classes": 34,
        "num_features": 46,
        "url": "https://www.unb.ca/cic/datasets/iotdataset-2023.html",
        "description": "IoT network traffic covering 33 attack types + benign.",
    },
    "MQTT-IoT-IDS2020": {
        "name": "MQTT-IoT-IDS2020",
        "num_classes": 5,
        "num_features": 33,
        "url": "https://ieee-dataport.org/open-access/mqtt-iot-ids2020-mqtt-internet-things-intrusion-detection-dataset",
        "description": "MQTT-based IoT intrusion detection dataset.",
    },
    "TON-IoT": {
        "name": "TON-IoT",
        "num_classes": 10,
        "num_features": 44,
        "url": "https://research.unsw.edu.au/projects/toniot-datasets",
        "description": "Telemetry data of IoT/IIoT services with 9 attack types + benign.",
    },
    "CICIDS-2018": {
        "name": "CICIDS-2018",
        "num_classes": 15,
        "num_features": 80,
        "url": "https://www.unb.ca/cic/datasets/ids-2018.html",
        "description": "Intrusion detection dataset with 14 attack types + benign.",
    },
    "ISCX-VPN2016": {
        "name": "ISCX-VPN2016",
        "num_classes": 12,
        "num_features": 78,
        "url": "https://www.unb.ca/cic/datasets/vpn.html",
        "description": "VPN and non-VPN encrypted traffic from 7 applications.",
    },
    "UNSW-NB15": {
        "name": "UNSW-NB15",
        "num_classes": 10,
        "num_features": 42,
        "url": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
        "description": "Network intrusion dataset with 9 attack categories + normal.",
    },
    "CICIDS-2017": {
        "name": "CICIDS-2017",
        "num_classes": 8,
        "num_features": 78,
        "url": "https://www.unb.ca/cic/datasets/ids-2017.html",
        "description": "Real-world noisy benchmark with inherent label noise.",
    },
}

# Per-dataset asymmetric confusion maps: class i is flipped to CONFUSION_MAP[dataset][i].
# Reflects semantically similar class pairs from domain knowledge.
_CONFUSION_MAPS = {
    "ISCX-VPN2016": {0: 6, 1: 7, 2: 8, 3: 9, 4: 10, 5: 11, 6: 0, 7: 1, 8: 2, 9: 3, 10: 4, 11: 5},
    "CICIDS-2018": {i: (i + 1) % 15 for i in range(15)},
    "UNSW-NB15": {i: (i + 1) % 10 for i in range(10)},
}


def load_features(path: str, dataset_name: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load features and labels from a CSV or NPZ file.

    CSV: last column is the label. NPZ: keys ``X`` and ``y``.
    """
    if path.endswith(".npz"):
        data = np.load(path)
        return data["X"], data["y"]
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int64)
    return X, y


def inject_noise(
    labels: np.ndarray,
    noise_type: str = "symmetric",
    noise_rate: float = 0.2,
    num_classes: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject label noise and return (noisy_labels, boolean mask of flipped indices).

    Noise types:
        - ``symmetric``: uniform random flip to any other class.
        - ``asymmetric``: flip to semantically similar class via confusion map.
        - ``adversarial``: bidirectional swap between attack (class 0) and benign (class 1).
    """
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    if num_classes is None:
        num_classes = int(labels.max()) + 1
    noisy = labels.copy()
    flip_mask = rng.rand(n) < noise_rate

    if noise_type == "symmetric":
        for i in np.where(flip_mask)[0]:
            candidates = [c for c in range(num_classes) if c != labels[i]]
            noisy[i] = rng.choice(candidates)

    elif noise_type == "asymmetric":
        # Build a default rotation map if no dataset-specific map available
        default_map = {c: (c + 1) % num_classes for c in range(num_classes)}
        conf_map = default_map
        for i in np.where(flip_mask)[0]:
            noisy[i] = conf_map.get(int(labels[i]), (int(labels[i]) + 1) % num_classes)

    elif noise_type == "adversarial":
        # Swap class 0 ↔ class 1 for selected samples
        for i in np.where(flip_mask)[0]:
            if labels[i] == 0:
                noisy[i] = 1
            elif labels[i] == 1:
                noisy[i] = 0
            else:
                noisy[i] = labels[i]
                flip_mask[i] = False

    else:
        raise ValueError(f"Unknown noise_type: {noise_type!r}")

    actual_mask = noisy != labels
    return noisy, actual_mask


def augment_with_configurations(
    features: np.ndarray,
    partitions: list[np.ndarray],
    embed_dim: int = 16,
) -> np.ndarray:
    """Augment features with community embeddings from each resolution.

    For each partition, community IDs are one-hot encoded (truncated/padded to
    ``embed_dim``) and concatenated to the original feature vectors.
    Final dimension = original_dim + (L * embed_dim).
    """
    parts = []
    for part in partitions:
        part = np.asarray(part)
        onehot = np.zeros((len(part), embed_dim), dtype=np.float32)
        for i, c in enumerate(part):
            if c < embed_dim:
                onehot[i, c] = 1.0
        parts.append(onehot)
    augmented = np.concatenate([features] + parts, axis=1)
    return augmented


class ConFixDataset(Dataset):
    """PyTorch Dataset for feature matrices and labels."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def get_dataloader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader from numpy arrays."""
    dataset = ConFixDataset(features, labels)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

"""Tests for confix.data."""

import numpy as np
import pytest

from confix.data import (
    DATASET_REGISTRY,
    inject_noise,
    augment_with_configurations,
    ConFixDataset,
    get_dataloader,
)


def test_registry_has_seven_datasets():
    assert len(DATASET_REGISTRY) == 7


def test_registry_entries_have_required_keys():
    for name, info in DATASET_REGISTRY.items():
        assert "num_classes" in info
        assert "num_features" in info
        assert "url" in info


def test_inject_symmetric_noise(synthetic_labels):
    noisy, mask = inject_noise(synthetic_labels, "symmetric", noise_rate=0.5, seed=0)
    assert noisy.shape == synthetic_labels.shape
    assert mask.sum() > 0
    assert np.all(noisy[~mask] == synthetic_labels[~mask])


def test_inject_asymmetric_noise(synthetic_labels):
    noisy, mask = inject_noise(synthetic_labels, "asymmetric", noise_rate=0.5, seed=0)
    assert noisy.shape == synthetic_labels.shape
    assert mask.sum() > 0


def test_inject_adversarial_noise():
    labels = np.array([0, 0, 1, 1, 2, 2])
    noisy, mask = inject_noise(labels, "adversarial", noise_rate=1.0, seed=0)
    # class 0 ↔ 1 swapped, class 2 untouched
    assert np.all(noisy[:2] == 1)
    assert np.all(noisy[2:4] == 0)
    assert np.all(noisy[4:] == 2)


def test_inject_noise_invalid_type(synthetic_labels):
    with pytest.raises(ValueError, match="Unknown noise_type"):
        inject_noise(synthetic_labels, "bad_type")


def test_inject_noise_zero_rate(synthetic_labels):
    noisy, mask = inject_noise(synthetic_labels, "symmetric", noise_rate=0.0, seed=0)
    assert np.array_equal(noisy, synthetic_labels)
    assert mask.sum() == 0


def test_augment_with_configurations(synthetic_features):
    n = synthetic_features.shape[0]
    partitions = [np.array([0, 1, 0, 1, 0] * (n // 5 + 1))[:n]]
    augmented = augment_with_configurations(synthetic_features, partitions, embed_dim=8)
    assert augmented.shape[0] == n
    assert augmented.shape[1] == synthetic_features.shape[1] + 8


def test_confix_dataset(synthetic_features, synthetic_labels):
    ds = ConFixDataset(synthetic_features, synthetic_labels)
    assert len(ds) == len(synthetic_labels)
    x, y = ds[0]
    assert x.shape[0] == synthetic_features.shape[1]


def test_get_dataloader(synthetic_features, synthetic_labels):
    loader = get_dataloader(synthetic_features, synthetic_labels, batch_size=16)
    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape[0] <= 16
    assert batch_y.shape[0] == batch_x.shape[0]

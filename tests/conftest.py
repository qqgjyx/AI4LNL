"""Shared pytest fixtures: synthetic data for fast, reproducible tests."""

import numpy as np
import pytest


@pytest.fixture
def synthetic_features():
    """50 samples, 10 features, 3 clusters."""
    rng = np.random.RandomState(42)
    centers = rng.randn(3, 10) * 3
    X = np.vstack([centers[i % 3] + rng.randn(10) * 0.5 for i in range(50)])
    return X.astype(np.float32)


@pytest.fixture
def synthetic_labels():
    """50 labels cycling through 3 classes."""
    return np.array([i % 3 for i in range(50)], dtype=np.int64)


@pytest.fixture
def sample_partitions():
    """Two partitions for 50 samples."""
    p1 = np.array([i % 3 for i in range(50)], dtype=np.int64)
    p2 = np.array([i % 5 for i in range(50)], dtype=np.int64)
    return [p1, p2]

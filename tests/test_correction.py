"""Tests for confix.correction."""

import numpy as np

from confix.correction import (
    compute_label_distribution,
    compute_label_consistency,
    compute_merge_split_consistency,
    consensus_vote,
    correct_labels,
)


def _make_partitions(n=50):
    """Two simple partitions for testing."""
    p1 = np.array([i % 3 for i in range(n)])
    p2 = np.array([i % 5 for i in range(n)])
    return [p1, p2]


def test_compute_label_distribution_shape(synthetic_labels):
    partitions = _make_partitions(len(synthetic_labels))
    dist = compute_label_distribution(partitions, synthetic_labels)
    num_classes = int(synthetic_labels.max()) + 1
    assert dist.shape == (len(synthetic_labels), 2, num_classes)


def test_compute_label_distribution_sums_to_one(synthetic_labels):
    partitions = _make_partitions(len(synthetic_labels))
    dist = compute_label_distribution(partitions, synthetic_labels)
    row_sums = dist.sum(axis=2)
    assert np.allclose(row_sums, 1.0, atol=1e-8)


def test_compute_label_consistency_range(synthetic_labels):
    partitions = _make_partitions(len(synthetic_labels))
    dist = compute_label_distribution(partitions, synthetic_labels)
    lc = compute_label_consistency(dist)
    assert lc.shape == (len(synthetic_labels),)
    assert np.all(lc >= 0) and np.all(lc <= 1.0 + 1e-8)


def test_compute_msc_shape(synthetic_labels):
    partitions = _make_partitions(len(synthetic_labels))
    msc = compute_merge_split_consistency(partitions)
    assert msc.shape == (len(synthetic_labels),)
    assert np.all(msc >= 0) and np.all(msc <= 1.0 + 1e-8)


def test_compute_msc_single_resolution():
    part = [np.array([0, 0, 1, 1, 2])]
    msc = compute_merge_split_consistency(part)
    assert np.allclose(msc, 1.0)


def test_consensus_vote_shape(synthetic_labels):
    partitions = _make_partitions(len(synthetic_labels))
    dist = compute_label_distribution(partitions, synthetic_labels)
    lc = compute_label_consistency(dist)
    msc = compute_merge_split_consistency(partitions)
    consensus = consensus_vote(dist, lc, msc)
    num_classes = int(synthetic_labels.max()) + 1
    assert consensus.shape == (len(synthetic_labels), num_classes)


def test_consensus_vote_normalized(synthetic_labels):
    partitions = _make_partitions(len(synthetic_labels))
    dist = compute_label_distribution(partitions, synthetic_labels)
    lc = compute_label_consistency(dist)
    msc = compute_merge_split_consistency(partitions)
    consensus = consensus_vote(dist, lc, msc)
    row_sums = consensus.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)


def test_correct_labels_no_change_on_clean():
    # If consensus strongly supports existing labels, nothing changes
    labels = np.array([0, 1, 2, 0, 1])
    consensus = np.array([
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
    ])
    corrected, mask = correct_labels(labels, consensus)
    assert np.array_equal(corrected, labels)
    assert mask.sum() == 0


def test_correct_labels_flips_noisy():
    # Sample 0 has label 0 but consensus strongly says 1
    labels = np.array([0, 1, 2])
    consensus = np.array([
        [0.1, 0.8, 0.1],   # label=0, but consensus says 1
        [0.1, 0.8, 0.1],   # label=1, matches
        [0.1, 0.1, 0.8],   # label=2, matches
    ])
    corrected, mask = correct_labels(labels, consensus, theta_low=0.3, delta=0.2)
    assert corrected[0] == 1
    assert mask[0] is np.True_
    assert corrected[1] == 1  # unchanged
    assert corrected[2] == 2  # unchanged

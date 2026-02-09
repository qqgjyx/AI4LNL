"""Consensus voting and label correction — core Confix contribution."""

import numpy as np


def compute_label_distribution(
    partitions: list[np.ndarray], labels: np.ndarray
) -> np.ndarray:
    """Compute per-sample, per-resolution label distributions.

    For each resolution r and sample i, counts how often each label appears
    among the neighbours of i within the same community at resolution r.

    Returns
    -------
    distributions : (N, R, C) array
        Normalized label distributions. R = number of resolutions,
        C = number of classes.
    """
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    num_classes = int(labels.max()) + 1
    R = len(partitions)
    dist = np.zeros((n, R, num_classes), dtype=np.float64)

    for r, part in enumerate(partitions):
        part = np.asarray(part)
        # Group samples by community
        communities: dict[int, list[int]] = {}
        for i, c in enumerate(part):
            communities.setdefault(int(c), []).append(i)

        for members in communities.values():
            # Count label frequencies within this community
            label_counts = np.zeros(num_classes, dtype=np.float64)
            for idx in members:
                label_counts[labels[idx]] += 1
            # Assign the community label distribution to each member
            total = label_counts.sum()
            if total > 0:
                label_freq = label_counts / total
            else:
                label_freq = label_counts
            for idx in members:
                dist[idx, r, :] = label_freq

    return dist


def compute_label_consistency(distributions: np.ndarray) -> np.ndarray:
    """Compute normalized entropy H̃ per sample across resolutions.

    H̃_i = 1 - H(mean_distribution_i) / log(C)

    Returns values in [0, 1]; higher means more consistent.
    """
    # distributions: (N, R, C)
    mean_dist = distributions.mean(axis=1)  # (N, C)
    C = mean_dist.shape[1]
    log_C = np.log(C + 1e-12)

    # Entropy of mean distribution
    eps = 1e-12
    entropy = -np.sum(mean_dist * np.log(mean_dist + eps), axis=1)
    normalized_entropy = entropy / log_C
    consistency = 1.0 - normalized_entropy
    return consistency


def compute_merge_split_consistency(partitions: list[np.ndarray]) -> np.ndarray:
    """Compute merge-split consistency (MSC) between adjacent resolutions.

    MSC_i measures how stable sample i's community assignment is between
    consecutive resolution levels. Returns per-sample scores in [0, 1].
    """
    n = len(partitions[0])
    R = len(partitions)

    if R < 2:
        return np.ones(n, dtype=np.float64)

    scores = np.zeros(n, dtype=np.float64)

    for r in range(R - 1):
        part_a = np.asarray(partitions[r])
        part_b = np.asarray(partitions[r + 1])

        for i in range(n):
            # Find co-members in resolution r
            comm_a = part_a[i]
            members_a = set(np.where(part_a == comm_a)[0].tolist())

            # Find co-members in resolution r+1
            comm_b = part_b[i]
            members_b = set(np.where(part_b == comm_b)[0].tolist())

            # Jaccard similarity of co-membership sets
            intersection = len(members_a & members_b)
            union = len(members_a | members_b)
            if union > 0:
                scores[i] += intersection / union

    scores /= (R - 1)
    return scores


def consensus_vote(
    distributions: np.ndarray,
    label_consistency: np.ndarray,
    msc: np.ndarray,
) -> np.ndarray:
    """Confidence-weighted consensus vote P_i(y).

    Combines multi-resolution label distributions weighted by label consistency
    and merge-split consistency.

    Returns
    -------
    consensus : (N, C) array of consensus label probabilities.
    """
    # distributions: (N, R, C), label_consistency: (N,), msc: (N,)
    N, R, C = distributions.shape
    weights = (label_consistency * msc)[:, None, None]  # (N, 1, 1)
    weighted_dist = distributions * weights  # (N, R, C)
    consensus = weighted_dist.mean(axis=1)  # (N, C)

    # Normalize rows
    row_sums = consensus.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    consensus = consensus / row_sums
    return consensus


def correct_labels(
    labels: np.ndarray,
    consensus: np.ndarray,
    theta_low: float = 0.3,
    delta: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Correct noisy labels based on consensus probabilities.

    A label is corrected when:
    1. The consensus confidence for the current label < theta_low, AND
    2. The top consensus candidate has probability > theta_low + delta.

    Parameters
    ----------
    labels : (N,) integer labels
    consensus : (N, C) consensus probabilities
    theta_low : confidence threshold below which a label is considered suspect
    delta : margin above theta_low required for correction

    Returns
    -------
    corrected : (N,) corrected labels
    corrected_mask : (N,) boolean mask where True means the label was changed
    """
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    corrected = labels.copy()
    corrected_mask = np.zeros(n, dtype=bool)

    for i in range(n):
        current_conf = consensus[i, labels[i]]
        best_class = int(np.argmax(consensus[i]))
        best_conf = consensus[i, best_class]

        if current_conf < theta_low and best_conf > theta_low + delta:
            corrected[i] = best_class
            corrected_mask[i] = True

    return corrected, corrected_mask

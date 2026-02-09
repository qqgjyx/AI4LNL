"""Consensus voting and label correction — core Confix contribution."""

import numpy as np


def compute_label_distribution(
    partitions: list[np.ndarray], labels: np.ndarray
) -> np.ndarray:
    """Compute per-sample, per-resolution label distributions.

    p_{l,i}(y) = |{j : omega_l(j) = c_i^(l) AND y_tilde_j = y}|
                 / |{j : omega_l(j) = c_i^(l)}|

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
    """Compute per-sample, per-resolution normalized entropy.

    H̃_{l,i} = -(1 / log C) * SUM_y p_{l,i}(y) * log(p_{l,i}(y))

    Returns
    -------
    H_tilde : (N, R) array with values in [0, 1]. Lower means more consistent.
    """
    # distributions: (N, R, C)
    C = distributions.shape[2]
    log_C = np.log(C + 1e-12)
    eps = 1e-12
    # Entropy per (sample, resolution)
    entropy = -np.sum(distributions * np.log(distributions + eps), axis=2)  # (N, R)
    H_tilde = entropy / log_C
    return H_tilde


def compute_merge_split_consistency(partitions: list[np.ndarray]) -> np.ndarray:
    """Compute merge-split consistency (MSC) between adjacent resolutions.

    MSC_{l,i} measures how stable sample i's community co-membership is between
    resolution l and l+1 (Jaccard similarity of co-member sets).

    Returns
    -------
    msc : (N, R) array. For the last resolution, MSC is set to 1.0.
    """
    n = len(partitions[0])
    R = len(partitions)
    msc = np.ones((n, R), dtype=np.float64)

    for r in range(R - 1):
        part_a = np.asarray(partitions[r])
        part_b = np.asarray(partitions[r + 1])

        for i in range(n):
            members_a = set(np.where(part_a == part_a[i])[0].tolist())
            members_b = set(np.where(part_b == part_b[i])[0].tolist())

            intersection = len(members_a & members_b)
            union = len(members_a | members_b)
            if union > 0:
                msc[i, r] = intersection / union
            else:
                msc[i, r] = 1.0

    return msc


def consensus_vote(
    distributions: np.ndarray,
    H_tilde: np.ndarray,
    msc: np.ndarray,
) -> np.ndarray:
    """Confidence-weighted consensus vote P_i(y).

    Conf_{l,i} = (1 - H̃_{l,i}) * MSC_{l,i}
    alpha_l^(i) = Conf_{l,i} / SUM_{l'} Conf_{l',i}
    P_i(y) = SUM_l alpha_l^(i) * p_{l,i}(y)

    Returns
    -------
    consensus : (N, C) array of consensus label probabilities.
    """
    # distributions: (N, R, C), H_tilde: (N, R), msc: (N, R)
    conf = (1.0 - H_tilde) * msc  # (N, R)

    # Normalize per sample to get alpha weights
    conf_sum = conf.sum(axis=1, keepdims=True)  # (N, 1)
    conf_sum = np.where(conf_sum > 0, conf_sum, 1.0)
    alpha = conf / conf_sum  # (N, R)

    # Weighted sum across resolutions
    consensus = np.einsum("nr,nrc->nc", alpha, distributions)  # (N, C)

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
    1. P_i(ỹ_i) < theta_low  (current label has low consensus support), AND
    2. max_{y != ỹ_i} P_i(y) - P_i(ỹ_i) > delta  (clear alternative exists).

    Parameters
    ----------
    labels : (N,) integer labels
    consensus : (N, C) consensus probabilities
    theta_low : confidence threshold below which a label is suspect
    delta : margin between best alternative and current label required for correction

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
        # Best class excluding the current label
        alt_probs = consensus[i].copy()
        alt_probs[labels[i]] = -1.0
        best_alt_class = int(np.argmax(alt_probs))
        best_alt_conf = consensus[i, best_alt_class]

        if current_conf < theta_low and (best_alt_conf - current_conf) > delta:
            corrected[i] = best_alt_class
            corrected_mask[i] = True

    return corrected, corrected_mask

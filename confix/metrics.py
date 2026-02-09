"""Evaluation metrics: accuracy, macro-F1, and repair rate."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy."""
    return float(accuracy_score(y_true, y_pred))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro-averaged F1 score."""
    return float(f1_score(y_true, y_pred, average="macro"))


def repair_rate(
    y_clean: np.ndarray, y_noisy: np.ndarray, y_corrected: np.ndarray
) -> float:
    """Fraction of noisy labels successfully restored to their clean value.

    repair_rate = |{i : y_noisy[i] != y_clean[i] AND y_corrected[i] == y_clean[i]}|
                  / |{i : y_noisy[i] != y_clean[i]}|

    Returns 1.0 if there are no noisy labels (nothing to repair).
    """
    y_clean = np.asarray(y_clean)
    y_noisy = np.asarray(y_noisy)
    y_corrected = np.asarray(y_corrected)

    noisy_mask = y_noisy != y_clean
    n_noisy = noisy_mask.sum()
    if n_noisy == 0:
        return 1.0
    repaired = np.sum((y_corrected[noisy_mask] == y_clean[noisy_mask]))
    return float(repaired / n_noisy)

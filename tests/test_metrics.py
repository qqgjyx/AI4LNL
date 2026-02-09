"""Tests for confix.metrics."""

import numpy as np
from confix.metrics import accuracy, macro_f1, repair_rate


def test_accuracy_perfect():
    y = np.array([0, 1, 2, 0, 1])
    assert accuracy(y, y) == 1.0


def test_accuracy_partial():
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([0, 1, 1, 0])
    assert accuracy(y_true, y_pred) == 0.75


def test_macro_f1_perfect():
    y = np.array([0, 1, 2, 0, 1, 2])
    assert macro_f1(y, y) == 1.0


def test_macro_f1_partial():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 0, 2, 2])
    score = macro_f1(y_true, y_pred)
    assert 0.0 < score < 1.0


def test_repair_rate_full():
    y_clean = np.array([0, 1, 2])
    y_noisy = np.array([1, 1, 0])  # indices 0, 2 are noisy
    y_corrected = np.array([0, 1, 2])  # both repaired
    assert repair_rate(y_clean, y_noisy, y_corrected) == 1.0


def test_repair_rate_none():
    y_clean = np.array([0, 1, 2])
    y_noisy = np.array([1, 1, 0])
    y_corrected = np.array([1, 1, 0])  # nothing repaired
    assert repair_rate(y_clean, y_noisy, y_corrected) == 0.0


def test_repair_rate_partial():
    y_clean = np.array([0, 1, 2])
    y_noisy = np.array([1, 1, 0])  # indices 0, 2 noisy
    y_corrected = np.array([0, 1, 0])  # only index 0 repaired
    assert repair_rate(y_clean, y_noisy, y_corrected) == 0.5


def test_repair_rate_no_noise():
    y = np.array([0, 1, 2])
    assert repair_rate(y, y, y) == 1.0

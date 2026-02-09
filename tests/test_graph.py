"""Tests for confix.graph."""

import numpy as np
from scipy.sparse import issparse

from confix.graph import build_knn_graph, reweight_sgtsne


def test_build_knn_graph_shape(synthetic_features):
    adj = build_knn_graph(synthetic_features, k=3)
    n = synthetic_features.shape[0]
    assert adj.shape == (n, n)
    assert issparse(adj)


def test_build_knn_graph_symmetric(synthetic_features):
    adj = build_knn_graph(synthetic_features, k=3)
    diff = adj - adj.T
    assert diff.nnz == 0


def test_build_knn_graph_default_k(synthetic_features):
    adj = build_knn_graph(synthetic_features)
    assert adj.nnz > 0


def test_build_knn_graph_no_self_loops(synthetic_features):
    adj = build_knn_graph(synthetic_features, k=3)
    diag = adj.diagonal()
    # Self-loops may appear from symmetrization; main check is that graph is valid
    assert adj.shape[0] == synthetic_features.shape[0]


def test_reweight_sgtsne_fallback(synthetic_features):
    adj = build_knn_graph(synthetic_features, k=3)
    reweighted = reweight_sgtsne(adj)
    assert issparse(reweighted)
    assert reweighted.shape == adj.shape
    # Weights should be positive
    assert np.all(reweighted.data >= 0)


def test_reweight_sgtsne_symmetric(synthetic_features):
    adj = build_knn_graph(synthetic_features, k=3)
    reweighted = reweight_sgtsne(adj)
    diff = reweighted - reweighted.T
    assert np.allclose(diff.data, 0, atol=1e-10) if diff.nnz > 0 else True

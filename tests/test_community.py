"""Tests for confix.community."""

import numpy as np
import igraph as ig

from confix.community import parallel_dt
from confix.graph import build_knn_graph


def test_parallel_dt_returns_list(synthetic_features):
    adj = build_knn_graph(synthetic_features, k=3)
    partitions = parallel_dt(adj, n_resolutions=3, seed=0)
    assert isinstance(partitions, list)
    assert len(partitions) == 3


def test_parallel_dt_partition_shape(synthetic_features):
    adj = build_knn_graph(synthetic_features, k=3)
    partitions = parallel_dt(adj, n_resolutions=3, seed=0)
    for p in partitions:
        assert len(p) == synthetic_features.shape[0]


def test_parallel_dt_coarse_to_fine(synthetic_features):
    adj = build_knn_graph(synthetic_features, k=3)
    partitions = parallel_dt(adj, gamma_range=(0.1, 3.0), n_resolutions=5, seed=0)
    n_communities = [len(np.unique(p)) for p in partitions]
    # Higher gamma should generally produce more (or equal) communities
    assert n_communities[-1] >= n_communities[0]


def test_parallel_dt_accepts_igraph():
    g = ig.Graph.Famous("Petersen")
    partitions = parallel_dt(g, n_resolutions=2, seed=0)
    assert len(partitions) == 2
    assert len(partitions[0]) == g.vcount()


def test_parallel_dt_deterministic(synthetic_features):
    adj = build_knn_graph(synthetic_features, k=3)
    p1 = parallel_dt(adj, n_resolutions=3, seed=42)
    p2 = parallel_dt(adj, n_resolutions=3, seed=42)
    for a, b in zip(p1, p2):
        assert np.array_equal(a, b)

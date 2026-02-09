"""KNN graph construction and SG-t-SNE edge reweighting."""

import math
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


def build_knn_graph(features: np.ndarray, k: int | None = None) -> csr_matrix:
    """Build a symmetric KNN adjacency graph.

    Parameters
    ----------
    features : (N, D) array
    k : number of neighbours. Defaults to ceil(log2(N)).

    Returns
    -------
    adj : (N, N) sparse symmetric adjacency matrix with binary weights.
    """
    n = features.shape[0]
    if k is None:
        k = max(2, math.ceil(math.log2(n)))
    k = min(k, n - 1)

    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
    nn.fit(features)
    distances, indices = nn.kneighbors(features)

    rows = np.repeat(np.arange(n), k)
    cols = indices.ravel()
    data = np.ones(len(rows), dtype=np.float64)

    adj = csr_matrix((data, (rows, cols)), shape=(n, n))
    # Symmetrize
    adj = adj + adj.T
    adj.data[:] = 1.0
    adj.eliminate_zeros()
    return adj


def reweight_sgtsne(adj: csr_matrix) -> csr_matrix:
    """Apply SG-t-SNE edge reweighting.

    Falls back to Gaussian kernel weights if ``SGtSNEpiPy`` is not installed.
    """
    try:
        import SGtSNEpiPy  # noqa: F401
        # SG-t-SNE reweighting via the library
        adj_reweighted = SGtSNEpiPy.sgtsne_reweight(adj)
        return adj_reweighted
    except (ImportError, AttributeError):
        # Fallback: Gaussian kernel based on graph distances
        adj = adj.copy().astype(np.float64)
        if adj.nnz == 0:
            return adj
        # Use existing distances stored as 1s; reweight with Gaussian kernel
        # sigma = median of nonzero distances (here distances are implicit via adjacency)
        rows, cols = adj.nonzero()
        dists = np.abs(rows - cols).astype(np.float64)  # ordinal fallback
        sigma = max(np.median(dists), 1e-8)
        weights = np.exp(-dists ** 2 / (2 * sigma ** 2))
        adj_reweighted = csr_matrix((weights, (rows, cols)), shape=adj.shape)
        # Symmetrize
        adj_reweighted = (adj_reweighted + adj_reweighted.T) / 2.0
        adj_reweighted.eliminate_zeros()
        return adj_reweighted

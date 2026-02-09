"""Parallel-DT: multi-resolution community detection via the Leiden algorithm."""

import numpy as np
import igraph as ig
import leidenalg
from scipy.sparse import issparse


def parallel_dt(
    graph,
    gamma_range: tuple[float, float] = (0.1, 2.0),
    n_resolutions: int = 10,
    seed: int = 42,
) -> list[np.ndarray]:
    """Run Leiden community detection across a range of resolution parameters.

    Parameters
    ----------
    graph : scipy.sparse matrix or igraph.Graph
        If sparse matrix, converted to igraph internally.
    gamma_range : (low, high)
        Resolution parameter range for Leiden (RBConfigurationVertexPartition).
    n_resolutions : int
        Number of evenly-spaced gamma values to sweep.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    partitions : list of np.ndarray
        Ordered from coarse (low gamma) to fine (high gamma). Each array has
        length N with integer community labels.
    """
    # Convert sparse matrix to igraph.Graph if needed
    if issparse(graph):
        graph = graph.tocoo()
        edges = list(zip(graph.row.tolist(), graph.col.tolist()))
        weights = graph.data.tolist()
        n_vertices = graph.shape[0]
        ig_graph = ig.Graph(n=n_vertices, edges=edges, directed=False)
        ig_graph.es["weight"] = weights
        ig_graph.simplify(combine_edges="max")
    elif isinstance(graph, ig.Graph):
        ig_graph = graph
    else:
        raise TypeError(f"Unsupported graph type: {type(graph)}")

    gammas = np.linspace(gamma_range[0], gamma_range[1], n_resolutions)
    partitions = []

    for gamma in gammas:
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=gamma,
            weights="weight" if "weight" in ig_graph.es.attributes() else None,
            seed=seed,
        )
        partitions.append(np.array(partition.membership, dtype=np.int64))

    return partitions

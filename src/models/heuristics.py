"""Unsupervised neighborhood scores for link prediction."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _one_hop_scores(csr: sp.csr_matrix, pairs: np.ndarray, method: str) -> np.ndarray:
    degrees = np.asarray(csr.sum(axis=1)).flatten()
    scores = np.zeros(len(pairs), dtype=np.float32)
    for i, (u, v) in enumerate(pairs):
        u, v = int(u), int(v)
        nu = csr.indices[csr.indptr[u] : csr.indptr[u + 1]]
        nv = csr.indices[csr.indptr[v] : csr.indptr[v + 1]]
        common = np.intersect1d(nu, nv, assume_unique=True)
        if common.size == 0:
            continue
        if method == "common_neighbors":
            scores[i] = float(common.size)
        elif method == "jaccard":
            union = np.union1d(nu, nv)
            scores[i] = float(common.size) / max(1, len(union))
        elif method == "adamic_adar":
            scores[i] = float(np.sum(1.0 / np.log(np.maximum(2.0, degrees[common]))))
        elif method == "resource_allocation":
            scores[i] = float(np.sum(1.0 / np.maximum(1.0, degrees[common])))
        else:
            raise ValueError(f"Unknown heuristic method: {method}")
    return scores


def _bipartite_three_walk_scores(csr: sp.csr_matrix, pairs: np.ndarray) -> np.ndarray:
    """Score (u, v) by 3-walks u-p-u'-v. 1-hop CN is empty on bipartite graphs."""
    deg = np.asarray(csr.sum(axis=1), dtype=np.float32).flatten()
    inv_deg = np.zeros_like(deg)
    nz = deg > 0
    inv_deg[nz] = 1.0 / deg[nz]
    w = csr @ sp.diags(inv_deg) @ csr
    three = (w @ csr).tocsr()
    u = pairs[:, 0].astype(np.int64)
    v = pairs[:, 1].astype(np.int64)
    # CSR fancy indexing with two 1-d arrays of equal length returns the
    # selected entries (SciPy >= 1.4), not an outer product.
    return np.asarray(three[u, v], dtype=np.float32).ravel()


def compute_heuristic_scores(
    adj: sp.spmatrix,
    pairs: np.ndarray,
    method: str = "resource_allocation",
    bipartite: bool = False,
) -> np.ndarray:
    """Unsupervised link scores. Bipartite graphs use 3-walk RA (1-hop CN is empty).

    Args:
        adj: Undirected adjacency. Converted to CSR float32 inside.
        pairs: Endpoint indices, integer ``[P, 2]``.
        method: One of ``common_neighbors``, ``jaccard``, ``adamic_adar``,
            ``resource_allocation``. Used only when ``bipartite`` is false.
        bipartite: Score 3-walks instead of 1-hop overlap.

    Returns:
        Scores ``[P]`` float32, aligned with ``pairs``.

    Raises:
        ValueError: Unknown ``method`` on a non-bipartite graph.
    """
    csr = adj.tocsr().astype(np.float32)
    if bipartite:
        return _bipartite_three_walk_scores(csr, pairs)
    return _one_hop_scores(csr, pairs, method)

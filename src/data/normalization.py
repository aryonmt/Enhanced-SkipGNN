"""Skip-graph operators and symmetric Laplacian normalization."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch


def zscore_columns(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Column-wise z-score.

    Args:
        x: Feature matrix ``[N, F]``.
        eps: Standard deviations below this are replaced by 1.

    Returns:
        Float32 matrix ``[N, F]``.
    """
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (x - mean) / std


def scipy_to_torch_sparse(mat: sp.spmatrix, device: torch.device | None = None) -> torch.Tensor:
    """Convert a SciPy matrix to a coalesced float32 sparse COO tensor.

    Args:
        mat: Two-dimensional sparse matrix.
        device: Optional destination device.

    Returns:
        Coalesced sparse COO tensor with the same shape as ``mat``.
    """
    mat = mat.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((mat.row, mat.col)).astype(np.int64))
    values = torch.from_numpy(mat.data.astype(np.float32))
    tensor = torch.sparse_coo_tensor(indices, values, torch.Size(mat.shape)).coalesce()
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def identity_sparse(n: int, device: torch.device | None = None) -> torch.Tensor:
    """Sparse identity used as one-hot node features.

    Args:
        n: Number of nodes.
        device: Optional destination device.

    Returns:
        Coalesced sparse COO identity ``[n, n]``.
    """
    idx = torch.arange(n, dtype=torch.int64)
    indices = torch.stack([idx, idx], dim=0)
    values = torch.ones(n, dtype=torch.float32)
    tensor = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def laplacian_normalize(adj: sp.spmatrix, add_self_loops: bool) -> sp.coo_matrix:
    """Symmetric ``D^{-1/2} A D^{-1/2}``. Isolated nodes stay zero (no epsilon fill).

    Args:
        adj: Square adjacency.
        add_self_loops: Add ``I`` before computing degrees.

    Returns:
        COO matrix with the same shape as ``adj``.
    """
    adj = adj.tocsr().astype(np.float32)
    if add_self_loops:
        adj = adj + sp.eye(adj.shape[0], dtype=np.float32, format="csr")
    deg = np.asarray(adj.sum(axis=1)).flatten().astype(np.float32)
    inv_sqrt = np.zeros_like(deg)
    mask = deg > 0
    inv_sqrt[mask] = np.power(deg[mask], -0.5)
    d = sp.diags(inv_sqrt)
    return (d @ adj @ d).tocoo()


def build_binary_skip(adj: sp.spmatrix) -> sp.coo_matrix:
    """Unweighted 2-hop skip ``A A^T`` with a zero diagonal.

    Args:
        adj: Square adjacency.

    Returns:
        COO matrix with the same shape as ``adj``.
    """
    a = adj.tocsr().astype(np.float32)
    skip = a @ a.T
    skip = skip.sign().tolil()
    skip.setdiag(0)
    return skip.tocoo()


def build_weighted_skip(adj: sp.spmatrix) -> sp.coo_matrix:
    """Resource-allocation skip: ``W = A D^{-1} A^T`` with a zero diagonal.

    Args:
        adj: Square adjacency.

    Returns:
        COO matrix with the same shape as ``adj``.
    """
    a = adj.tocsr().astype(np.float32)
    deg = np.asarray(a.sum(axis=1)).flatten()
    inv_deg = 1.0 / np.maximum(deg, 1.0)
    d_inv = sp.diags(inv_deg.astype(np.float32))
    w = a @ d_inv @ a.T
    w = w.tolil()
    w.setdiag(0)
    return w.tocoo()


def build_three_hop_return(adj: sp.spmatrix) -> sp.coo_matrix:
    """3-walk operator ``W3 = W2 D^{-1} A``, where ``W2`` is the RA skip graph.

    Args:
        adj: Square adjacency.

    Returns:
        COO matrix with the same shape as ``adj``.
    """
    a = adj.tocsr().astype(np.float32)
    deg = np.asarray(a.sum(axis=1)).flatten().astype(np.float32)
    inv_deg = np.zeros_like(deg)
    nz = deg > 0
    inv_deg[nz] = 1.0 / deg[nz]
    d_inv = sp.diags(inv_deg)
    w2 = build_weighted_skip(a).tocsr()
    w3 = (w2 @ d_inv @ a).tocsr()
    w3.eliminate_zeros()
    return w3.tocoo()

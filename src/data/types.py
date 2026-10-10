"""Dataset bundle, split arrays, and the pair dataset."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset


@dataclass
class SplitArrays:
    """Edge list for one split.

    Attributes:
        pairs: Node-index pairs, int64 ``[E, 2]``.
        labels: Labels aligned with ``pairs``, shape ``[E]``.
    """

    pairs: np.ndarray
    labels: np.ndarray


@dataclass
class DatasetBundle:
    """Leak-free graph + splits used by every trainer and metric helper.

    ``adj_train`` and the Laplacian tensors (``f_orig``, ``f_skip_*``, ``f_3hop``)
    are built from **train positives only**. Val/test edges never enter the graph.
    For bipartite datasets, sources occupy ``[0, n_source)`` and targets occupy
    ``[n_source, n_nodes)``.

    Attributes:
        features: Dense ``[N, F]`` or, for one-hot input, sparse identity ``[N, N]``.
        adj_train: Train-positive adjacency, SciPy CSR ``[N, N]``.
        f_orig: Normalized adjacency with self-loops, sparse COO ``[N, N]``.
        f_skip_bin: Normalized binary 2-hop skip, sparse COO ``[N, N]``.
        f_skip_weighted: Normalized resource-allocation skip, sparse COO ``[N, N]``.
        f_3hop: Normalized 3-walk operator, sparse COO ``[N, N]``.
        train: Pairs ``[E_train, 2]`` and labels ``[E_train]``.
        val: Pairs ``[E_val, 2]`` and labels ``[E_val]``.
        test: Pairs ``[E_test, 2]`` and labels ``[E_test]``.
    """

    name: str
    n_nodes: int
    n_source: int
    n_target: int
    bipartite: bool
    features: torch.Tensor
    adj_train: sp.csr_matrix
    f_orig: torch.Tensor
    f_skip_bin: torch.Tensor
    f_skip_weighted: torch.Tensor
    f_3hop: torch.Tensor
    train: SplitArrays
    val: SplitArrays
    test: SplitArrays
    known_positives: set[tuple[int, int]]
    input_type: str
    idx_map: dict[str, int] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        """Feature width, ``features.shape[1]`` (``N`` for a sparse identity)."""
        return int(self.features.shape[1])


class PairDataset(Dataset):
    """Map-style dataset of link pairs and float labels."""

    def __init__(self, pairs: np.ndarray, labels: np.ndarray) -> None:
        """Store link pairs and labels as tensors.

        Args:
            pairs: Integer endpoint indices ``[B, 2]``.
            labels: Labels aligned with ``pairs``, shape ``[B]``.
        """
        self.pairs = torch.as_tensor(pairs, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        """Number of pairs."""
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one pair ``[2]`` and its float label."""
        return self.pairs[index], self.labels[index]

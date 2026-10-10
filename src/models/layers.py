"""Sparse graph convolution used by the GCN and SkipGNN encoders."""

from __future__ import annotations

import torch
import torch.nn as nn


class SparseGraphConvolution(nn.Module):
    """Graph convolution ``A (X W)`` with an optional bias.

    ``X`` may be dense or sparse. ``A`` must be a sparse COO matrix.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        """Build the weight (and optional bias).

        Args:
            in_features: Input width ``F``.
            out_features: Output width ``F'``.
            bias: When true, add a learned bias of shape ``[F']``.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier-uniform weights and a zero bias."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Apply one sparse graph convolution.

        Args:
            x: Node features, dense or sparse ``[N, F]``.
            adj: Sparse adjacency ``[N, N]``.

        Returns:
            Updated node states ``[N, F']``.
        """
        if x.is_sparse:
            support = torch.sparse.mm(x, self.weight)
        else:
            support = x @ self.weight
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output

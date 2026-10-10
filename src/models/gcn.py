"""Two-layer GCN link predictor on the original graph."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers import SparseGraphConvolution


class StandardGCN(nn.Module):
    """Two-layer GCN on the original graph only (no skip graph).

    The decoder concatenates the two endpoint embeddings. ``adj_skip`` is
    accepted so the trainer can call every encoder the same way, and is ignored.

    Attributes:
        skip_kind: Always ``"none"``.
    """

    skip_kind = "none"

    def __init__(self, nfeat: int, nhid1: int, nhid2: int, nhid_dec: int, dropout: float = 0.5) -> None:
        """Build the two convolutions and the concat decoder.

        Args:
            nfeat: Input feature width ``F``.
            nhid1: Width of the first graph convolution.
            nhid2: Embedding width.
            nhid_dec: Hidden width of the pair decoder.
            dropout: Dropout probability on the first hidden layer.
        """
        super().__init__()
        self.gc1 = SparseGraphConvolution(nfeat, nhid1)
        self.gc2 = SparseGraphConvolution(nhid1, nhid2)
        self.dropout = dropout
        self.decoder1 = nn.Linear(nhid2 * 2, nhid_dec)
        self.decoder2 = nn.Linear(nhid_dec, 1)

    def encode(self, x: torch.Tensor, adj: torch.Tensor, adj_skip: torch.Tensor | None = None) -> torch.Tensor:
        """Encode nodes with two convolutions on ``adj``.

        Args:
            x: Node features, dense or sparse ``[N, F]``.
            adj: Normalized adjacency, sparse ``[N, N]``.
            adj_skip: Unused. Present for a shared trainer signature.

        Returns:
            Node embeddings ``[N, nhid2]``.
        """
        del adj_skip
        h1 = F.relu(self.gc1(x, adj))
        h1 = F.dropout(h1, self.dropout, training=self.training)
        return self.gc2(h1, adj)

    def decode(self, emb: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        """Score pairs from concatenated endpoint embeddings.

        Args:
            emb: Node embeddings ``[N, nhid2]``.
            pairs: Endpoint indices, long ``[B, 2]``.

        Returns:
            Link logits ``[B]``.
        """
        u, v = pairs[:, 0], pairs[:, 1]
        feat = torch.cat((emb[u], emb[v]), dim=-1)
        logits = self.decoder2(F.relu(self.decoder1(feat)))
        return logits.squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        pairs: torch.Tensor,
        adj_skip: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the graph and score ``pairs``.

        Args:
            x: Node features, dense or sparse ``[N, F]``.
            adj: Normalized adjacency, sparse ``[N, N]``.
            pairs: Endpoint indices, long ``[B, 2]``.
            adj_skip: Unused.

        Returns:
            Logits ``[B]`` and node embeddings ``[N, nhid2]``.
        """
        emb = self.encode(x, adj, adj_skip)
        return self.decode(emb, pairs), emb

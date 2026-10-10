"""Dual-view SkipGNN with a batch InfoNCE term."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers import SparseGraphConvolution


class ContrastiveSkipGNN(nn.Module):
    """Dual-view encoder: original vs skip graph, aligned with batch InfoNCE.

    Training ``forward`` returns logits, embeddings, and the contrastive term.
    Eval ``forward`` returns logits and embeddings only.

    Attributes:
        skip_kind: ``"weighted"``.
    """

    skip_kind = "weighted"

    def __init__(
        self,
        nfeat: int,
        nhid1: int,
        nhid2: int,
        nhid_dec: int,
        dropout: float = 0.5,
        proj_dim: int = 32,
        tau: float = 0.2,
        lambda_cl: float = 0.1,
    ) -> None:
        """Build the two view encoders, the gate, the projectors, and the decoder.

        Args:
            nfeat: Input feature width ``F``.
            nhid1: Width of the first graph convolution in each view.
            nhid2: Embedding width of each view.
            nhid_dec: Hidden width of the pair decoder.
            dropout: Dropout probability inside each view and the decoder.
            proj_dim: Width of the InfoNCE projection head.
            tau: InfoNCE temperature.
            lambda_cl: Contrastive weight read by the trainer (not applied inside ``forward``).
        """
        super().__init__()
        self.dropout = dropout
        self.tau = tau
        self.lambda_cl = lambda_cl
        self.o_gc1 = SparseGraphConvolution(nfeat, nhid1)
        self.o_gc2 = SparseGraphConvolution(nhid1, nhid2)
        self.s_gc1 = SparseGraphConvolution(nfeat, nhid1)
        self.s_gc2 = SparseGraphConvolution(nhid1, nhid2)
        self.fuse_gate = nn.Linear(nhid2 * 2, 1)
        self.proj_o = nn.Sequential(nn.Linear(nhid2, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))
        self.proj_s = nn.Sequential(nn.Linear(nhid2, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))
        self.decoder_fc1 = nn.Linear(nhid2 * 4, nhid_dec)
        self.decoder_bn = nn.BatchNorm1d(nhid_dec)
        self.decoder_fc2 = nn.Linear(nhid_dec, 1)

    def encode(
        self,
        x: torch.Tensor,
        f_orig: torch.Tensor,
        f_skip: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode both views and gate them into one embedding.

        Args:
            x: Node features, dense or sparse ``[N, F]``.
            f_orig: Normalized original adjacency, sparse ``[N, N]``.
            f_skip: Normalized skip operator, sparse ``[N, N]``.

        Returns:
            Fused embeddings ``[N, nhid2]``, original-view states ``[N, nhid2]``,
            and skip-view states ``[N, nhid2]``.
        """
        h_o1 = F.dropout(F.relu(self.o_gc1(x, f_orig)), self.dropout, training=self.training)
        h_o2 = self.o_gc2(h_o1, f_orig)
        h_s1 = F.dropout(F.relu(self.s_gc1(x, f_skip)), self.dropout, training=self.training)
        h_s2 = self.s_gc2(h_s1, f_skip)
        g = torch.sigmoid(self.fuse_gate(torch.cat((h_o2, h_s2), dim=-1)))
        emb = g * h_o2 + (1.0 - g) * h_s2
        return emb, h_o2, h_s2

    def contrastive_loss(self, h_o2: torch.Tensor, h_s2: torch.Tensor, node_ids: torch.Tensor) -> torch.Tensor:
        """Symmetric InfoNCE between the original and skip views on unique nodes.

        Args:
            h_o2: Original-view states ``[N, nhid2]``.
            h_s2: Skip-view states ``[N, nhid2]``.
            node_ids: Flat node indices (the trainer passes ``pairs.reshape(-1)``).

        Returns:
            Scalar loss. Zero when fewer than two unique nodes are present.
        """
        unique_nodes = torch.unique(node_ids)
        if unique_nodes.numel() <= 1:
            return h_o2.new_zeros(())
        z_o = F.normalize(self.proj_o(h_o2[unique_nodes]), dim=-1)
        z_s = F.normalize(self.proj_s(h_s2[unique_nodes]), dim=-1)
        logits = z_o @ z_s.t() / self.tau
        labels = torch.arange(unique_nodes.size(0), device=h_o2.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    def decode(self, emb: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        """Score pairs with the 4-way decoder.

        Args:
            emb: Node embeddings ``[N, nhid2]``.
            pairs: Endpoint indices, long ``[B, 2]``.

        Returns:
            Link logits ``[B]``.
        """
        e_u, e_v = emb[pairs[:, 0]], emb[pairs[:, 1]]
        z = torch.cat((e_u, e_v, e_u * e_v, torch.abs(e_u - e_v)), dim=-1)
        h = F.relu(self.decoder_bn(self.decoder_fc1(z)))
        h = F.dropout(h, self.dropout, training=self.training)
        return self.decoder_fc2(h).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        pairs: torch.Tensor,
        adj_skip: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode, score ``pairs``, and add InfoNCE while training.

        Args:
            x: Node features, dense or sparse ``[N, F]``.
            adj: Normalized original adjacency, sparse ``[N, N]``.
            pairs: Endpoint indices, long ``[B, 2]``.
            adj_skip: Normalized skip operator, sparse ``[N, N]``.

        Returns:
            Eval: logits ``[B]`` and embeddings ``[N, nhid2]``.
            Train: those two plus a scalar contrastive loss.

        Raises:
            ValueError: ``adj_skip`` is missing.
        """
        if adj_skip is None:
            raise ValueError("ContrastiveSkipGNN requires adj_skip")
        emb, h_o2, h_s2 = self.encode(x, adj, adj_skip)
        logits = self.decode(emb, pairs)
        if self.training:
            return logits, emb, self.contrastive_loss(h_o2, h_s2, pairs.reshape(-1))
        return logits, emb

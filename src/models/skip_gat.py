"""SkipGATv2 with chunked sparse attention so GDI-scale skip graphs fit on a T4."""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Chunked path: peak activation is O(chunk), not O(E). Used only when a full
# (E, heads, d_k) score tensor would exceed GAT_VECTORIZED_MAX_BYTES.
EDGE_CHUNK = int(os.environ.get("GAT_EDGE_CHUNK", "1048576"))
VECTORIZED_MAX_BYTES = int(os.environ.get("GAT_VECTORIZED_MAX_BYTES", str(384 * 1024 * 1024)))


def _softmax_gatv2(
    h_src: torch.Tensor,
    h_dst: torch.Tensor,
    attn_vec: torch.Tensor,
    u_idx: torch.Tensor,
    v_idx: torch.Tensor,
    negative_slope: float,
    dropout: float,
    training: bool,
) -> torch.Tensor:
    """Vectorized sparse GATv2. Fast, but stores (E, heads, d_k) for autograd."""
    n_nodes, heads, d_k = h_src.shape
    edge_feat = F.leaky_relu(h_src[u_idx] + h_dst[v_idx], negative_slope=negative_slope)
    score = (edge_feat * attn_vec.unsqueeze(0)).sum(dim=-1)
    max_score = torch.full((n_nodes, heads), torch.finfo(score.dtype).min, device=score.device, dtype=score.dtype)
    max_score.scatter_reduce_(0, u_idx.unsqueeze(-1).expand_as(score), score, reduce="amax", include_self=True)
    exp = torch.exp(score - max_score[u_idx])
    denom = h_src.new_zeros(n_nodes, heads)
    denom.index_add_(0, u_idx, exp)
    alpha = exp / denom[u_idx].clamp_min(1e-12)
    if training and dropout > 0:
        keep = torch.rand_like(alpha) >= dropout
        alpha = alpha * keep.to(alpha.dtype) / (1.0 - dropout)
    out = h_src.new_zeros(n_nodes, heads, d_k)
    out.index_add_(0, u_idx, h_dst[v_idx] * alpha.unsqueeze(-1))
    return out


class _ChunkedSparseGATv2(torch.autograd.Function):
    """Sparse GATv2 whose peak activation is O(chunk) instead of O(E)."""

    @staticmethod
    def forward(
        ctx,
        h_src: torch.Tensor,
        h_dst: torch.Tensor,
        attn_vec: torch.Tensor,
        u_idx: torch.Tensor,
        v_idx: torch.Tensor,
        negative_slope: float,
        dropout: float,
        training: bool,
        chunk_size: int,
    ) -> torch.Tensor:
        n_nodes, heads, d_k = h_src.shape
        n_edges = int(u_idx.numel())
        device = h_src.device
        dtype = h_src.dtype
        chunk = max(1, int(chunk_size))

        def _score(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            edge_feat = F.leaky_relu(h_src[u] + h_dst[v], negative_slope=negative_slope)
            return (edge_feat * attn_vec.unsqueeze(0)).sum(dim=-1)

        with torch.no_grad():
            max_score = torch.full((n_nodes, heads), torch.finfo(dtype).min, device=device, dtype=dtype)
            for start in range(0, n_edges, chunk):
                sl = slice(start, min(start + chunk, n_edges))
                u, v = u_idx[sl], v_idx[sl]
                score = _score(u, v)
                max_score.scatter_reduce_(
                    0, u.unsqueeze(-1).expand_as(score), score, reduce="amax", include_self=True
                )

            denom = torch.zeros(n_nodes, heads, device=device, dtype=dtype)
            for start in range(0, n_edges, chunk):
                sl = slice(start, min(start + chunk, n_edges))
                u, v = u_idx[sl], v_idx[sl]
                score = _score(u, v)
                denom.index_add_(0, u, torch.exp(score - max_score[u]))

            alpha_sm = h_src.new_empty(n_edges, heads)
            for start in range(0, n_edges, chunk):
                sl = slice(start, min(start + chunk, n_edges))
                u, v = u_idx[sl], v_idx[sl]
                score = _score(u, v)
                alpha_sm[sl] = torch.exp(score - max_score[u]) / denom[u].clamp_min(1e-12)

            keep = torch.ones_like(alpha_sm, dtype=torch.bool)
            alpha = alpha_sm
            if training and dropout > 0:
                keep = torch.rand(alpha_sm.shape, device=device, dtype=dtype) >= dropout
                alpha = alpha_sm * keep.to(dtype) / (1.0 - dropout)

            out = h_src.new_zeros(n_nodes, heads, d_k)
            for start in range(0, n_edges, chunk):
                sl = slice(start, min(start + chunk, n_edges))
                u, v = u_idx[sl], v_idx[sl]
                out.index_add_(0, u, h_dst[v] * alpha[sl].unsqueeze(-1))

        ctx.save_for_backward(h_src, h_dst, attn_vec, u_idx, v_idx, alpha_sm, keep)
        ctx.negative_slope = float(negative_slope)
        ctx.dropout = float(dropout) if training else 0.0
        ctx.chunk_size = chunk
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        h_src, h_dst, attn_vec, u_idx, v_idx, alpha_sm, keep = ctx.saved_tensors
        n_nodes, heads, d_k = h_src.shape
        n_edges = int(u_idx.numel())
        chunk = ctx.chunk_size
        negative_slope = ctx.negative_slope
        dropout = ctx.dropout
        grad_out = grad_out.reshape(n_nodes, heads, d_k)

        scale = 1.0 / (1.0 - dropout) if dropout > 0 else 1.0
        alpha = alpha_sm * keep.to(alpha_sm.dtype) * scale if dropout > 0 else alpha_sm

        grad_h_src = torch.zeros_like(h_src)
        grad_h_dst = torch.zeros_like(h_dst)
        grad_attn = torch.zeros_like(attn_vec)
        grad_alpha_used = alpha.new_zeros(alpha.shape)

        for start in range(0, n_edges, chunk):
            sl = slice(start, min(start + chunk, n_edges))
            u, v = u_idx[sl], v_idx[sl]
            a = alpha[sl]
            g_u = grad_out[u]
            h_v = h_dst[v]
            grad_h_dst.index_add_(0, v, g_u * a.unsqueeze(-1))
            grad_alpha_used[sl] = (g_u * h_v).sum(dim=-1)

        grad_alpha_sm = (
            grad_alpha_used * keep.to(alpha.dtype) * scale if dropout > 0 else grad_alpha_used
        )

        weighted = alpha_sm * grad_alpha_sm
        src_dot = torch.zeros(n_nodes, heads, device=alpha_sm.device, dtype=alpha_sm.dtype)
        src_dot.index_add_(0, u_idx, weighted)
        grad_score = alpha_sm * (grad_alpha_sm - src_dot[u_idx])

        for start in range(0, n_edges, chunk):
            sl = slice(start, min(start + chunk, n_edges))
            u, v = u_idx[sl], v_idx[sl]
            pre = h_src[u] + h_dst[v]
            edge_feat = F.leaky_relu(pre, negative_slope=negative_slope)
            gs = grad_score[sl].unsqueeze(-1)
            grad_attn += (edge_feat * gs).sum(dim=0)
            grad_pre = gs * attn_vec.unsqueeze(0)
            grad_pre = torch.where(pre > 0, grad_pre, grad_pre * negative_slope)
            grad_h_src.index_add_(0, u, grad_pre)
            grad_h_dst.index_add_(0, v, grad_pre)

        return grad_h_src, grad_h_dst, grad_attn, None, None, None, None, None, None


class SparseGATv2Layer(nn.Module):
    """Edge-sparse GATv2: attention on COO edges, O(E) memory.

    Uses the vectorized softmax when the score tensor would fit in
    ``GAT_VECTORIZED_MAX_BYTES``; otherwise the chunked autograd path.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int = 4,
        dropout: float = 0.5,
        negative_slope: float = 0.2,
    ) -> None:
        """Build source/destination projections and the attention vector.

        Args:
            in_features: Input width ``F``.
            out_features: Output width. Must be divisible by ``heads``.
            heads: Number of attention heads.
            dropout: Attention dropout probability, applied only while training.
            negative_slope: LeakyReLU slope on the pre-attention sum.

        Raises:
            ValueError: ``out_features`` is not divisible by ``heads``.
        """
        super().__init__()
        if out_features % heads != 0:
            raise ValueError("out_features must be divisible by heads")
        self.heads = heads
        self.d_k = out_features // heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.w_src = nn.Linear(in_features, out_features, bias=False)
        self.w_dst = nn.Linear(in_features, out_features, bias=False)
        self.attn_vec = nn.Parameter(torch.empty(heads, self.d_k))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier-uniform projections and attention vector; zero bias."""
        nn.init.xavier_uniform_(self.w_src.weight)
        nn.init.xavier_uniform_(self.w_dst.weight)
        nn.init.xavier_uniform_(self.attn_vec)
        nn.init.zeros_(self.bias)

    def _linear(self, x: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        if x.is_sparse:
            return torch.sparse.mm(x, layer.weight.t())
        return layer(x)

    def forward(self, x: torch.Tensor, adj_sparse: torch.Tensor) -> torch.Tensor:
        """Attend over COO edges and return concatenated heads.

        Args:
            x: Node features, dense or sparse ``[N, in_features]``.
            adj_sparse: Sparse adjacency ``[N, N]``. Coalesced to COO if needed.

        Returns:
            Node states ``[N, out_features]``.

        Raises:
            TypeError: ``adj_sparse`` is dense.
        """
        if not adj_sparse.is_sparse:
            raise TypeError("SparseGATv2Layer expects a sparse COO adjacency")
        adj = adj_sparse if adj_sparse.is_coalesced() else adj_sparse.coalesce()
        n_nodes = x.size(0)
        h_src = self._linear(x, self.w_src).view(n_nodes, self.heads, self.d_k)
        h_dst = self._linear(x, self.w_dst).view(n_nodes, self.heads, self.d_k)

        indices = adj.indices()
        if indices.numel() == 0:
            return self.bias.expand(n_nodes, -1).clone()

        u_idx, v_idx = indices[0], indices[1]
        nbytes = int(u_idx.numel()) * self.heads * self.d_k * h_src.element_size()
        if nbytes <= VECTORIZED_MAX_BYTES:
            out = _softmax_gatv2(
                h_src,
                h_dst,
                self.attn_vec,
                u_idx,
                v_idx,
                float(self.negative_slope),
                float(self.dropout),
                bool(self.training),
            )
        else:
            out = _ChunkedSparseGATv2.apply(
                h_src,
                h_dst,
                self.attn_vec,
                u_idx,
                v_idx,
                float(self.negative_slope),
                float(self.dropout),
                bool(self.training),
                int(EDGE_CHUNK),
            )
        return out.reshape(n_nodes, self.heads * self.d_k) + self.bias


class SkipGATv2(nn.Module):
    """Gated original/skip encoder with sparse GATv2 instead of static Laplacian GCN.

    The decoder concatenates ``[e_u, e_v, e_u * e_v, |e_u - e_v|]``.

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
        heads: int = 4,
    ) -> None:
        """Build the gated GATv2 stack and the 4-way decoder.

        Args:
            nfeat: Input feature width ``F``.
            nhid1: Width after the first gated layer. Must be divisible by ``heads``.
            nhid2: Embedding width. Must be divisible by ``heads``.
            nhid_dec: Hidden width of the pair decoder.
            dropout: Dropout on attention and on the decoder hidden state.
            heads: Attention heads in every ``SparseGATv2Layer``.
        """
        super().__init__()
        self.dropout = dropout
        self.o_gat1 = SparseGATv2Layer(nfeat, nhid1, heads=heads, dropout=dropout)
        self.so_gat1 = SparseGATv2Layer(nfeat, nhid1, heads=heads, dropout=dropout)
        self.s_gat1 = SparseGATv2Layer(nfeat, nhid1, heads=heads, dropout=dropout)
        self.os_gat1 = SparseGATv2Layer(nhid1, nhid1, heads=heads, dropout=dropout)
        self.gate1 = nn.Linear(nhid1 * 2, 1)
        self.o_gat2 = SparseGATv2Layer(nhid1, nhid2, heads=heads, dropout=dropout)
        self.so_gat2 = SparseGATv2Layer(nhid1, nhid2, heads=heads, dropout=dropout)
        self.gate2 = nn.Linear(nhid2 * 2, 1)
        self.decoder_fc1 = nn.Linear(nhid2 * 4, nhid_dec)
        self.decoder_bn = nn.BatchNorm1d(nhid_dec)
        self.decoder_fc2 = nn.Linear(nhid_dec, 1)

    def encode(self, x: torch.Tensor, adj: torch.Tensor, adj_skip: torch.Tensor) -> torch.Tensor:
        """Fuse original-graph and skip-graph GATv2 layers into node embeddings.

        Args:
            x: Node features, dense or sparse ``[N, F]``.
            adj: Normalized original adjacency, sparse ``[N, N]``.
            adj_skip: Normalized skip operator, sparse ``[N, N]``.

        Returns:
            Node embeddings ``[N, nhid2]``.
        """
        h_o1 = self.o_gat1(x, adj)
        h_so1 = self.so_gat1(x, adj_skip)
        g1 = torch.sigmoid(self.gate1(torch.cat((h_o1, h_so1), dim=-1)))
        o1 = F.elu(g1 * h_o1 + (1.0 - g1) * h_so1)
        s1 = F.elu(self.s_gat1(x, adj_skip) + self.os_gat1(o1, adj))
        o1 = F.dropout(o1, self.dropout, training=self.training)
        s1 = F.dropout(s1, self.dropout, training=self.training)
        h_o2 = self.o_gat2(o1, adj)
        h_so2 = self.so_gat2(s1, adj_skip)
        g2 = torch.sigmoid(self.gate2(torch.cat((h_o2, h_so2), dim=-1)))
        return g2 * h_o2 + (1.0 - g2) * h_so2

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the graph and score ``pairs``.

        Args:
            x: Node features, dense or sparse ``[N, F]``.
            adj: Normalized original adjacency, sparse ``[N, N]``.
            pairs: Endpoint indices, long ``[B, 2]``.
            adj_skip: Normalized skip operator, sparse ``[N, N]``.

        Returns:
            Logits ``[B]`` and node embeddings ``[N, nhid2]``.

        Raises:
            ValueError: ``adj_skip`` is missing.
        """
        if adj_skip is None:
            raise ValueError("SkipGATv2 requires adj_skip")
        emb = self.encode(x, adj, adj_skip)
        return self.decode(emb, pairs), emb

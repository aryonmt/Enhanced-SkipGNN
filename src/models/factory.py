"""Map a short model name to a concrete encoder/decoder."""

from __future__ import annotations

import torch.nn as nn

from src.data.types import DatasetBundle
from src.models.ams_skipgnn import AMSSkipGNN, make_ablation
from src.models.contrastive_skipgnn import ContrastiveSkipGNN
from src.models.gcn import StandardGCN
from src.models.skip_gat import SkipGATv2
from src.models.skipgnn import SkipGNNBaseline
from src.models.three_hop_skipgnn import ThreeHopSkipGNN

# Canonical CLI keys used in benchmark.csv. Extra aliases keep older notebooks working.
MODEL_ALIASES = {
    "gcn": ("gcn", "standard_gcn", "baseline_gcn"),
    "skipgnn": ("skipgnn", "skipgnn_baseline", "baseline"),
    "ams": ("ams", "ams_skipgnn", "ams-skipgnn"),
    "gat": ("gat", "skip_gat", "skipgatv2", "skipgat"),
    "3hop": ("3hop", "three_hop", "three_hop_skipgnn", "threehop"),
    "contrastive": ("contrastive", "contrastive_skipgnn", "cl_skip"),
}


def build_model(
    name: str,
    bundle: DatasetBundle,
    hidden1: int = 64,
    hidden2: int = 64,
    decoder_hidden: int = 64,
    dropout: float = 0.5,
) -> nn.Module:
    """Build a named model whose width matches ``bundle.n_features``.

    Known names: ``gcn``, ``skipgnn``, ``ams``, ``gat``, ``3hop``, ``contrastive``.
    Ablation aliases (``weighted``, ``gated``, ``full``, …) are handled by
    :func:`make_ablation`.

    Args:
        name: Model key or alias (case-insensitive).
        bundle: Dataset whose ``n_features`` sets the encoder input width.
        hidden1: First hidden width.
        hidden2: Embedding width.
        decoder_hidden: Pair-decoder hidden width.
        dropout: Dropout probability forwarded to the model.

    Returns:
        A module with ``encode``, ``decode``, and ``skip_kind``.
    """
    kwargs = dict(
        nfeat=int(bundle.n_features),
        nhid1=hidden1,
        nhid2=hidden2,
        nhid_dec=decoder_hidden,
        dropout=dropout,
    )
    key = name.lower().strip()
    if key in MODEL_ALIASES["gcn"]:
        return StandardGCN(**kwargs)
    if key in MODEL_ALIASES["skipgnn"]:
        return SkipGNNBaseline(**kwargs)
    if key in MODEL_ALIASES["ams"]:
        return AMSSkipGNN(**kwargs)
    if key in MODEL_ALIASES["gat"]:
        return SkipGATv2(**kwargs)
    if key in MODEL_ALIASES["3hop"]:
        return ThreeHopSkipGNN(**kwargs)
    if key in MODEL_ALIASES["contrastive"]:
        return ContrastiveSkipGNN(**kwargs)
    return make_ablation(key, **kwargs)

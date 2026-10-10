"""BCE trainer with validation-AUPRC early stopping and dual-bank test metrics."""

from __future__ import annotations

import copy
import random
import time
from typing import Any, Callable

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from src.data.samplers import generate_bipartite_aware_hard_negatives
from src.data.types import DatasetBundle, PairDataset
from src.eval.metrics import evaluate_dual_bank


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch (CPU and all CUDA devices).

    Args:
        seed: Integer seed applied to ``random``, NumPy, and ``torch``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _skip_matrix(bundle: DatasetBundle, model: torch.nn.Module):
    """Pick the skip tensor(s) the model declared via `skip_kind`."""
    kind = getattr(model, "skip_kind", "none")
    if kind == "binary":
        return bundle.f_skip_bin
    if kind == "weighted":
        return bundle.f_skip_weighted
    if kind == "three_hop_bundle":
        return (bundle.f_skip_weighted, bundle.f_3hop)
    return None


def _encode_outputs(
    model: torch.nn.Module,
    bundle: DatasetBundle,
    skip,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """One full-graph encoder pass. Contrastive models also return the two views."""
    kind = getattr(model, "skip_kind", "none")
    x, adj = bundle.features, bundle.f_orig
    if kind == "three_hop_bundle":
        return model.encode(x, adj, skip[0], skip[1]), None, None
    if kind == "none":
        return model.encode(x, adj, None), None, None
    packed = model.encode(x, adj, skip)
    if isinstance(packed, tuple) and len(packed) == 3:
        return packed[0], packed[1], packed[2]
    return packed, None, None


def _unpack_logits(out) -> torch.Tensor:
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


@torch.no_grad()
def predict_probs(
    model: torch.nn.Module,
    bundle: DatasetBundle,
    pairs: np.ndarray,
    batch_size: int = 2048,
) -> np.ndarray:
    """Score pairs. Encodes the graph once, then decodes in batches.

    Args:
        model: Encoder with ``encode`` / ``decode`` and a ``skip_kind`` attribute.
        bundle: Graph and feature tensors. Scores are computed on ``bundle.features.device``.
        pairs: Endpoint indices, integer ``[P, 2]``.
        batch_size: Decoder batch size.

    Returns:
        Sigmoid probabilities ``[P]``. Empty input yields shape ``[0]``.
    """
    model.eval()
    device = bundle.features.device
    skip = _skip_matrix(bundle, model)
    emb, _, _ = _encode_outputs(model, bundle, skip)
    outs: list[np.ndarray] = []
    for start in range(0, len(pairs), batch_size):
        batch = torch.as_tensor(pairs[start : start + batch_size], device=device)
        logits = model.decode(emb, batch)
        outs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(outs, axis=0) if outs else np.zeros((0,), dtype=np.float32)


@torch.no_grad()
def extract_node_embeddings(model: torch.nn.Module, bundle: DatasetBundle) -> np.ndarray:
    """Return node embeddings from one encoder pass.

    Args:
        model: Trained encoder.
        bundle: Graph the encoder reads (``features``, ``f_orig``, and the skip tensor).

    Returns:
        Float array ``[N, D]`` on CPU, with ``D`` the encoder width.
    """
    model.eval()
    skip = _skip_matrix(bundle, model)
    emb, _, _ = _encode_outputs(model, bundle, skip)
    return emb.detach().cpu().numpy()


def train_one_model(
    model: torch.nn.Module,
    bundle: DatasetBundle,
    *,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    patience: int = 8,
    grad_clip: float = 5.0,
    seed: int = 42,
    log_fn: Callable[[str], None] | None = None,
    encode_once: bool = False,
) -> dict[str, Any]:
    """Train with BCE (+ optional contrastive term), early-stop on val AUPRC.

    After restoring the best checkpoint, scores the official uniform test split
    and a hard negative bank sampled from the train graph only.

    Default matches AMS/SkipGNN: encode the graph on every decoder batch so
    Adam sees one step per minibatch. ``encode_once=True`` is a speed hack
    (one encoder step per epoch) and underfits GDI-scale models.

    Args:
        model: Encoder/decoder moved onto ``bundle.features.device``.
        bundle: Leak-free graph and train/val/test pairs.
        epochs: Maximum training epochs.
        batch_size: Pair minibatch size. The last incomplete batch is dropped.
        lr: Adam learning rate.
        weight_decay: Adam weight decay.
        patience: Epochs without a val-AUPRC improvement before stopping.
        grad_clip: Global gradient-norm clip.
        seed: Seed passed to :func:`set_seed` and to hard-negative sampling.
        log_fn: Optional line logger. Defaults to ``print``.
        encode_once: One encoder backward per epoch when true; per batch when false.

    Returns:
        Dual-bank metric dict from :func:`evaluate_dual_bank`, plus
        ``best_val_auprc``, ``history`` (epoch, loss, val AUPRC),
        ``hard_pairs`` ``[P, 2]``, ``hard_labels`` ``[P]``, ``hard_probs`` ``[P]``,
        and ``test_probs`` ``[E_test]``.
    """
    set_seed(seed)
    device = bundle.features.device
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loader = DataLoader(
        PairDataset(bundle.train.pairs, bundle.train.labels),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    skip = _skip_matrix(bundle, model)
    best_state = copy.deepcopy(model.state_dict())
    best_auprc = -1.0
    stall = 0
    history: list[dict[str, float]] = []
    lambda_cl = float(getattr(model, "lambda_cl", 0.1))

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0
        t_enc = 0.0
        if encode_once:
            opt.zero_grad(set_to_none=True)
            t0 = time.perf_counter()
            emb, h_o2, h_s2 = _encode_outputs(model, bundle, skip)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_enc = time.perf_counter() - t0
            batch_losses: list[torch.Tensor] = []
            for pairs, labels in loader:
                pairs = pairs.to(device)
                labels = labels.to(device)
                logits = model.decode(emb, pairs)
                loss = loss_fn(logits, labels)
                if h_o2 is not None and h_s2 is not None:
                    loss = loss + lambda_cl * model.contrastive_loss(h_o2, h_s2, pairs.reshape(-1))
                bs = int(labels.size(0))
                batch_losses.append(loss * bs)
                n_seen += bs
            if batch_losses:
                mean_loss = torch.stack(batch_losses).sum() / max(1, n_seen)
                mean_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                total_loss = float(mean_loss.detach().item()) * n_seen
        else:
            for pairs, labels in loader:
                pairs = pairs.to(device)
                labels = labels.to(device)
                opt.zero_grad(set_to_none=True)
                out = model(bundle.features, bundle.f_orig, pairs, skip)
                if isinstance(out, tuple) and len(out) == 3:
                    logits, _, cl_loss = out
                    loss = loss_fn(logits, labels) + lambda_cl * cl_loss
                else:
                    logits = _unpack_logits(out)
                    loss = loss_fn(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                bs = int(labels.size(0))
                total_loss += float(loss.item()) * bs
                n_seen += bs
        epoch_loss = total_loss / max(1, n_seen)
        val_probs = predict_probs(model, bundle, bundle.val.pairs)
        val_auprc = float(average_precision_score(bundle.val.labels, val_probs))
        history.append({"epoch": epoch, "loss": epoch_loss, "val_auprc": val_auprc})
        msg = f"epoch={epoch:02d} loss={epoch_loss:.4f} val_auprc={val_auprc:.4f}"
        if encode_once:
            msg = f"{msg} encode_s={t_enc:.2f}"
        if log_fn:
            log_fn(msg)
        else:
            print(msg, flush=True)
        if val_auprc > best_auprc:
            best_auprc = val_auprc
            best_state = copy.deepcopy(model.state_dict())
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                break

    model.load_state_dict(best_state)
    pos_test = bundle.test.pairs[bundle.test.labels >= 0.5]
    hard_neg = generate_bipartite_aware_hard_negatives(
        pos_test,
        bundle.adj_train,
        bundle.known_positives,
        bundle.bipartite,
        bundle.n_source,
        bundle.n_target,
        seed=seed,
    )
    hard_pairs = np.concatenate([pos_test, hard_neg], axis=0)
    hard_labels = np.concatenate(
        [np.ones(len(pos_test), dtype=np.int32), np.zeros(len(hard_neg), dtype=np.int32)]
    )
    val_p = predict_probs(model, bundle, bundle.val.pairs)
    test_p = predict_probs(model, bundle, bundle.test.pairs)
    hard_p = predict_probs(model, bundle, hard_pairs)
    metrics = evaluate_dual_bank(
        val_p, bundle.val.labels, test_p, bundle.test.labels, hard_p, hard_labels
    )
    metrics["best_val_auprc"] = best_auprc
    metrics["history"] = history
    metrics["hard_pairs"] = hard_pairs
    metrics["hard_labels"] = hard_labels
    metrics["hard_probs"] = hard_p
    metrics["test_probs"] = test_p
    return metrics


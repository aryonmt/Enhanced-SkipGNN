"""Validation-only F1 thresholding plus AUROC / AUPRC / Brier helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def find_optimal_f1_threshold(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    n_steps: int = 91,
) -> float:
    """Select the F1 threshold on validation scores only.

    Args:
        val_probs: 1-D predicted probabilities.
        val_labels: 1-D labels, same length as ``val_probs``.
        n_steps: Number of candidate thresholds on ``[0.05, 0.95]``.

    Returns:
        Earliest grid point in ``linspace(0.05, 0.95, n_steps)`` with the highest F1.
    """
    thresholds = np.linspace(0.05, 0.95, n_steps)
    best_f1 = -1.0
    best_t = 0.5
    y = np.asarray(val_labels, dtype=np.int32)
    p = np.asarray(val_probs, dtype=np.float64)
    for t in thresholds:
        pred = (p >= t).astype(np.int32)
        score = float(f1_score(y, pred, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return best_t


def evaluate_at_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """AUROC, AUPRC, F1, and Brier score at a frozen threshold.

    Args:
        probs: 1-D predicted probabilities.
        labels: 1-D labels, same length as ``probs``.
        threshold: Cutoff for the F1 prediction. Ranking metrics ignore it.

    Returns:
        Dict with ``auroc``, ``auprc``, ``f1``, ``brier``, and ``threshold``.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    preds = (probs >= threshold).astype(np.int32)
    return {
        "auroc": float(roc_auc_score(labels, probs)),
        "auprc": float(average_precision_score(labels, probs)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "brier": float(np.mean((probs - labels) ** 2)),
        "threshold": float(threshold),
    }


def evaluate_dual_bank(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    hard_probs: np.ndarray,
    hard_labels: np.ndarray,
) -> dict[str, Any]:
    """Fit τ* on validation F1, then score uniform and hard test banks.

    Args:
        val_probs: 1-D validation probabilities.
        val_labels: 1-D validation labels.
        test_probs: 1-D uniform-test probabilities.
        test_labels: 1-D uniform-test labels.
        hard_probs: 1-D hard-bank probabilities.
        hard_labels: 1-D hard-bank labels.

    Returns:
        ``tau_star`` plus, for ``val``, ``test_uniform``, and ``test_hard``,
        the dicts ``<name>@0.5`` and ``<name>@tau`` from
        :func:`evaluate_at_threshold`.
    """
    tau = find_optimal_f1_threshold(val_probs, val_labels)
    out: dict[str, Any] = {"tau_star": tau}
    for name, probs, labels in (
        ("val", val_probs, val_labels),
        ("test_uniform", test_probs, test_labels),
        ("test_hard", hard_probs, hard_labels),
    ):
        out[f"{name}@0.5"] = evaluate_at_threshold(probs, labels, 0.5)
        out[f"{name}@tau"] = evaluate_at_threshold(probs, labels, tau)
    return out

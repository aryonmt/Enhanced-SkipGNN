"""Retrain after randomly dropping train edges."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import load_dataset_splits
from src.data.normalization import (
    build_binary_skip,
    build_weighted_skip,
    laplacian_normalize,
    scipy_to_torch_sparse,
)
from src.models.factory import build_model
from src.training.trainer import train_one_model


def _symmetric_adj(n: int, pairs: np.ndarray) -> sp.csr_matrix:
    if len(pairs) == 0:
        return sp.csr_matrix((n, n), dtype=np.float32)
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    data = np.ones(rows.shape[0], dtype=np.float32)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    adj.sum_duplicates()
    adj.data[:] = 1.0
    return adj.tocsr()


def drop_edges(bundle, frac: float, seed: int, device: torch.device):
    rng = np.random.default_rng(seed)
    b = copy.copy(bundle)
    pos = bundle.train.pairs[bundle.train.labels >= 0.5]
    keep = rng.random(len(pos)) > frac
    kept = pos[keep]
    adj = _symmetric_adj(bundle.n_nodes, kept)
    b.adj_train = adj
    b.f_orig = scipy_to_torch_sparse(laplacian_normalize(adj, add_self_loops=True), device)
    b.f_skip_bin = scipy_to_torch_sparse(
        laplacian_normalize(build_binary_skip(adj), add_self_loops=False), device
    )
    b.f_skip_weighted = scipy_to_torch_sparse(
        laplacian_normalize(build_weighted_skip(adj), add_self_loops=False), device
    )
    return b


def main() -> int:
    p = argparse.ArgumentParser(description="Retrain after randomly dropping train edges.")
    p.add_argument("--dataset", default="DTI")
    p.add_argument("--models", nargs="+", default=["gcn", "skipgnn", "ams"])
    p.add_argument("--fractions", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.7, 0.9])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    base = load_dataset_splits(args.dataset, device=device)
    rows = []
    for frac in args.fractions:
        for model_name in args.models:
            auprcs = []
            for seed in args.seeds:
                print(f"=== robustness {model_name} frac={frac} seed={seed} ===", flush=True)
                bundle = drop_edges(base, frac, seed, device)
                model = build_model(
                    model_name,
                    bundle,
                    hidden1=int(cfg["hidden_dim"]),
                    hidden2=int(cfg["hidden_dim"]),
                    decoder_hidden=int(cfg["decoder_hidden"]),
                    dropout=float(cfg["dropout"]),
                )
                metrics = train_one_model(
                    model,
                    bundle,
                    epochs=args.epochs,
                    batch_size=int(cfg["batch_size"]),
                    lr=float(cfg["lr"]),
                    weight_decay=float(cfg["weight_decay"]),
                    patience=max(4, int(cfg["patience"]) - 2),
                    seed=seed,
                )
                auprcs.append(metrics["test_uniform@0.5"]["auprc"])
            rows.append(
                {
                    "dataset": args.dataset,
                    "model": model_name,
                    "missing_frac": frac,
                    "auprc_mean": float(np.mean(auprcs)),
                    "auprc_std": float(np.std(auprcs)),
                }
            )
            print(rows[-1], flush=True)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "robustness.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

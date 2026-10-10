"""Four-step AMS ablation on one dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import load_dataset_splits
from src.models.factory import build_model
from src.training.trainer import train_one_model

STEPS = [
    ("0_skipgnn", "skipgnn"),
    ("1_weighted", "weighted"),
    ("2_gated", "gated"),
    ("3_full", "full"),
]


def main() -> int:
    p = argparse.ArgumentParser(description="Four-step AMS ablation on one dataset.")
    p.add_argument("--dataset", default="DTI")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    bundle = load_dataset_splits(args.dataset, device=device)
    rows = []
    for step_name, variant in STEPS:
        for seed in args.seeds:
            print(f"=== ablation {step_name} seed={seed} ===", flush=True)
            model = build_model(
                variant,
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
                patience=int(cfg["patience"]),
                seed=seed,
            )
            rec = {
                "dataset": args.dataset,
                "step": step_name,
                "seed": seed,
                "hard_auprc": metrics["test_hard@0.5"]["auprc"],
                "uniform_auprc": metrics["test_uniform@0.5"]["auprc"],
            }
            rows.append(rec)
            print(rec, flush=True)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "ablation.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

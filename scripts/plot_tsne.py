"""t-SNE figures from saved node-embedding archives."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.plotting import plot_saved_embedding_tsne


def main() -> int:
    p = argparse.ArgumentParser(description="t-SNE plots from saved node embeddings")
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="figures")
    args = p.parse_args()
    plot_saved_embedding_tsne(ROOT / args.results, ROOT / args.out)
    print(f"t-SNE figures written to {ROOT / args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Scripts

Run from the **repo root** after `pip install -e .`.

| Script | Purpose |
| --- | --- |
| `fetch_data.py` | Copy upstream fold-1 splits into `data/raw/` |
| `run_benchmark.py` | Train models; merge `results/<DS>/benchmark.csv` |
| `run_ablation.py` | Ablation of the `ams` encoder only → `results/ablation.csv` |
| `run_robustness.py` | Missing-edge sweep → `results/robustness.csv` |
| `make_figures.py` | Rebuild `figures/` from CSVs |
| `plot_tsne.py` | t-SNE from saved embeddings |
| `import_kaggle_bundle.py` | Fold a Kaggle zip into this tree |
| `pack_kaggle_bundle.py` | Optional zip helper. The Kaggle runner also writes the zip inline. |
| `dump_for_review.py` | Local review dump (`review_bundle.txt`, gitignored) |

`run_benchmark.py --help` lists models and flags. `--quick` is smoke only.

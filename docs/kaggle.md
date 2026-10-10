# Kaggle T4 runner

Full training is designed for a **Kaggle Notebook with a T4 GPU**.

Paper tables for all four datasets × seven models are already in git. Upload the runner only if you need to retrain.

## Upload

File: [`notebooks/kaggle_runner.ipynb`](../notebooks/kaggle_runner.ipynb)

Settings:

- Accelerator: **GPU T4**
- Internet: **on** (the first cell clones this GitHub repo)
- **Save Version → Save & Run All** (not “Quick Save”)

The notebook clones `https://github.com/aryonmt/Enhanced-SkipGNN.git` branch `main` (override with env `REPO_BRANCH`), `pip install -e .`, then runs `scripts/run_benchmark.py` as subprocesses. Hardware and install details live in the [README quickstart](../README.md) and in `environment.yml`.

Default **`STAGE=0`** is a DTI `--quick` smoke so a Save & Run All cannot accidentally retrain GDI for hours. Set `STAGE=5` only to redo GDI extras.

Protocol and validity: [training-protocol.md](training-protocol.md).

## Stages (historical)

| STAGE | What it trained |
| --- | --- |
| 0 (default) | DTI `--quick` smoke |
| 1 | DTI + DDI: GCN, SkipGNN, AMS, heuristic |
| 2 | PPI + GDI baselines + DTI ablation/robustness |
| 3 | DTI extra models (`gat`, `3hop`, `contrastive`) + t-SNE |
| 4 | Extra models on DDI + PPI |
| 5 | Extra models on **GDI** (per-batch encode, same protocol as AMS) |

Executed archives live under `notebooks/executed/` (`stage1` … `stage5_gdi_extras`). They are a log of what ran, not the upload target.

This notebook **implements** stages 0, 4, and 5 only. Stages 1–3 are historical; setting `STAGE` to those values will not replay them.

## Batch sizes used on T4

- DDI / PPI extra models: `--batch-size 1024`
- GDI extra models: `--batch-size 1024`, encode every decoder batch
- Skip-graph GAT: vectorized when `(E, heads, d_k)` fits in ~384MB, otherwise chunked (`GAT_EDGE_CHUNK` default 1,048,576)

Models run **one subprocess at a time** so a crash does not lose earlier CSVs.

## Resume

If a version dies mid-run:

1. Start a new notebook from the latest `kaggle_runner.ipynb` on GitHub.
2. **Add Input → Notebook Output Files** → the failed version.
3. Set `STAGE` to `4` or `5` only. Unknown values fall back to smoke. For Stage 5, set `SKIP_COMPLETE=1` only when resuming a **fair** (per-batch) mid-run.

## Bring artifacts home

The last cell writes `/kaggle/working/ams_skipgnn_kaggle_bundle.zip`. Download **only that zip**. On the repo machine:

```bash
python scripts/import_kaggle_bundle.py --src ams_skipgnn_kaggle_bundle.zip --make-figures
```

Do not commit zips. `kaggle kernels pull` downloads notebook **source**, not Output files. Use the Output tab or `kaggle kernels output`.

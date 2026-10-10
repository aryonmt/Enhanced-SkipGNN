"""Paper figures from cached `results/` tables.

Comparison plots use a fixed dataset order (DTI, DDI, PPI, GDI) and a fixed
seven-model order. Seed-level rows drive error bars (`errorbar="sd"`).
Missing model×dataset cells are omitted, never drawn as zero.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve

DATASET_ORDER = ("DTI", "DDI", "PPI", "GDI")
MODEL_ORDER = ("gcn", "skipgnn", "ams", "heuristic", "gat", "3hop", "contrastive")

MODEL_PALETTE = {
    "gcn": "#6E6E6E",
    "skipgnn": "#5B8DB8",
    "ams": "#1F4E79",
    "heuristic": "#8C6D31",
    "gat": "#2E8B57",
    "3hop": "#C44E52",
    "contrastive": "#6B5B95",
}

MODEL_LABELS = {
    "gcn": "GCN",
    "skipgnn": "SkipGNN",
    "ams": "AMS",
    "heuristic": "Heuristic",
    "gat": "SkipGATv2",
    "3hop": "3-hop",
    "contrastive": "Contrastive",
}

LABEL_PALETTE = {MODEL_LABELS[k]: v for k, v in MODEL_PALETTE.items()}
LABEL_ORDER = [MODEL_LABELS[k] for k in MODEL_ORDER]

_SKIP_RESULT_PARTS = {"temp", "_kaggle_bundle_staging", ".git"}
SEED_NOTE = "Error bars: SD across seeds 42 / 123 / 7."
ABLATION_LABELS = {
    "0_skipgnn": "0. SkipGNN",
    "1_weighted": "1. Weighted skip",
    "2_gated": "2. Gated fusion",
    "3_full": "3. Full AMS",
}

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.18,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _result_files(results_dir: Path, pattern: str) -> list[Path]:
    return [
        path
        for path in Path(results_dir).rglob(pattern)
        if not _SKIP_RESULT_PARTS.intersection(path.parts)
    ]


def _model_key(name: object) -> str:
    return str(name).lower().strip()


def _finish(ax: plt.Axes, ylabel: str, xlabel: str = "Dataset") -> None:
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.grid(axis="x", visible=False)
    legend = ax.get_legend()
    if legend is not None:
        legend.set_frame_on(False)
        legend.set_title("")


def _footnote(fig: plt.Figure, text: str) -> None:
    fig.text(0.5, -0.03, text, ha="center", fontsize=8, color="#4a4a4a")


def plot_ablation_ladder(df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of hard AUPRC by ablation step and dataset.

    Args:
        df: Rows with ``step``, ``dataset``, and ``hard_auprc``.
        out_path: PNG path. Parent directories are not created here.
    """
    plot_df = df.copy()
    plot_df["step_label"] = plot_df["step"].map(lambda s: ABLATION_LABELS.get(str(s), str(s)))
    order = [ABLATION_LABELS[k] for k in ABLATION_LABELS if ABLATION_LABELS[k] in set(plot_df["step_label"])]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    sns.barplot(
        data=plot_df,
        x="step_label",
        y="hard_auprc",
        hue="dataset",
        order=order,
        errorbar="sd",
        capsize=0.06,
        ax=ax,
    )
    ax.set_title("Ablation ladder (hard AUPRC, mean ± SD)")
    ax.set_ylabel("Hard AUPRC")
    ax.set_xlabel("Ablation step")
    ax.set_ylim(0.5, 0.9)
    ax.tick_params(axis="x", rotation=12)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    legend = ax.get_legend()
    if legend is not None:
        legend.set_frame_on(False)
        legend.set_title("")
    fig.savefig(out_path)
    plt.close(fig)


def plot_robustness(df: pd.DataFrame, out_path: Path) -> None:
    """Line chart of test AUPRC against missing-edge fraction.

    Args:
        df: Rows with ``model``, ``missing_frac``, ``auprc_mean``, and ``auprc_std``.
        out_path: PNG path.
    """
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for model, sub in df.groupby("model"):
        key = _model_key(model)
        ax.plot(
            sub["missing_frac"],
            sub["auprc_mean"],
            marker="o",
            label=MODEL_LABELS.get(key, str(model)),
            color=MODEL_PALETTE.get(key),
        )
        ax.fill_between(
            sub["missing_frac"],
            sub["auprc_mean"] - sub["auprc_std"],
            sub["auprc_mean"] + sub["auprc_std"],
            alpha=0.15,
            color=MODEL_PALETTE.get(key),
        )
    ax.set_title("Missing-edge robustness")
    ax.set_xlabel("Missing-edge fraction")
    ax.set_ylabel("Test AUPRC")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    fig.savefig(out_path)
    plt.close(fig)


def plot_pr_grid(
    curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    out_path: Path,
) -> None:
    """One panel per dataset; overlay every model that has a seed-42 PR array.

    Args:
        curves: ``dataset -> model -> (precision, recall)`` arrays.
        out_path: PNG path.
    """
    names = [d for d in DATASET_ORDER if d in curves] or list(curves.keys())
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 8.6))
    for ax, name in zip(axes.ravel(), names):
        model_curves = curves.get(name, {})
        for key in MODEL_ORDER:
            if key not in model_curves:
                continue
            prec, rec = model_curves[key]
            ax.plot(
                rec,
                prec,
                color=MODEL_PALETTE.get(key),
                linewidth=1.5,
                label=MODEL_LABELS.get(key, key),
            )
        ax.set_title(name)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(linestyle=":", linewidth=0.5, alpha=0.6)
        ax.legend(frameon=False, fontsize=7, loc="lower left")
    for ax in axes.ravel()[len(names) :]:
        ax.axis("off")
    fig.suptitle("Precision–recall on the uniform test split (seed 42)", y=1.01)
    fig.tight_layout()
    n_models = max((len(v) for v in curves.values()), default=0)
    if n_models < len(MODEL_ORDER):
        _footnote(
            fig,
            "Curves are drawn for every stored pr_<model>_seed42.npz. "
            "Older runs saved AMS only; AUPRC for all seven models is in fig1.",
        )
    fig.savefig(out_path)
    plt.close(fig)


def write_summary(payload: dict[str, Any], out_path: Path) -> None:
    """Write ``payload`` as indented JSON, creating parent directories.

    Args:
        payload: JSON-serializable summary.
        out_path: Destination path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _metric_mean(group: pd.DataFrame, primary: str, fallback: str | None = None) -> float | None:
    if primary in group.columns and group[primary].notna().any():
        return float(group[primary].mean())
    if fallback and fallback in group.columns and group[fallback].notna().any():
        return float(group[fallback].mean())
    return None


def _load_benchmarks(results_dir: Path) -> pd.DataFrame | None:
    bench_files = _result_files(results_dir, "benchmark.csv")
    if not bench_files:
        return None
    frames = []
    for path in bench_files:
        df = pd.read_csv(path)
        if "dataset" not in df.columns:
            df["dataset"] = path.parent.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _seed_metric_frame(df: pd.DataFrame, metric: str, fallback: str | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    col = metric if metric in df.columns else fallback
    if col is None or col not in df.columns:
        return pd.DataFrame()
    for _, rec in df.iterrows():
        value = rec.get(col)
        if pd.isna(value):
            continue
        key = _model_key(rec["model"])
        rows.append(
            {
                "dataset": rec["dataset"],
                "model": MODEL_LABELS.get(key, str(rec["model"])),
                "model_key": key,
                "auprc": float(value),
            }
        )
    return pd.DataFrame(rows)


def plot_all_models_uniform_hard(df: pd.DataFrame, out_path: Path) -> None:
    """Side-by-side uniform and hard AUPRC bars (mean ± SD over seeds).

    Args:
        df: Benchmark rows with ``dataset``, ``model``, and AUPRC columns.
        out_path: PNG path. No file is written when both panels are empty.
    """
    uni = _seed_metric_frame(df, "uniform_auprc", "auprc")
    hard = _seed_metric_frame(df, "hard_auprc")
    if uni.empty and hard.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharey=True)
    panels = (("Uniform AUPRC", uni), ("Hard AUPRC", hard))
    hues = [lab for lab in LABEL_ORDER if lab in set(pd.concat([uni, hard])["model"])]
    for ax, (title, panel) in zip(axes, panels):
        if panel.empty:
            ax.set_title(title)
            continue
        sns.barplot(
            data=panel,
            x="dataset",
            y="auprc",
            hue="model",
            order=list(DATASET_ORDER),
            hue_order=[h for h in hues if h in set(panel["model"])],
            palette=LABEL_PALETTE,
            errorbar="sd",
            capsize=0.05,
            ax=ax,
        )
        ax.set_title(title)
        _finish(ax, "AUPRC")
        if ax is not axes[0]:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
        else:
            ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    fig.suptitle("All models — mean ± SD over seeds", y=1.03)
    fig.tight_layout()
    _footnote(fig, SEED_NOTE)
    fig.savefig(out_path)
    plt.close(fig)


def plot_delta_hard_auprc(df: pd.DataFrame, out_path: Path) -> None:
    """Hard AUPRC of every model minus AMS on that dataset (AMS = 0).

    Args:
        df: Benchmark rows with ``dataset``, ``model``, and ``hard_auprc``.
        out_path: PNG path. No file is written when AMS or the other models are missing.
    """
    rows = []
    for dataset in DATASET_ORDER:
        group = df[df["dataset"] == dataset]
        ams_h = _metric_mean(group[group["model"].map(_model_key) == "ams"], "hard_auprc")
        if ams_h is None:
            continue
        for key in MODEL_ORDER:
            if key == "ams":
                continue
            val = _metric_mean(group[group["model"].map(_model_key) == key], "hard_auprc")
            if val is None:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "model": MODEL_LABELS[key],
                    "delta": val - ams_h,
                }
            )
    if not rows:
        return
    plot_df = pd.DataFrame(rows)
    hues = [lab for lab in LABEL_ORDER if lab != "AMS" and lab in set(plot_df["model"])]
    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    sns.barplot(
        data=plot_df,
        x="dataset",
        y="delta",
        hue="model",
        order=list(DATASET_ORDER),
        hue_order=hues,
        palette=LABEL_PALETTE,
        ax=ax,
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Hard AUPRC relative to AMS (positive = better than AMS)")
    ax.set_ylabel("Model − AMS")
    ax.set_xlabel("Dataset")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    fig.savefig(out_path)
    plt.close(fig)


def plot_hard_auprc_box(df: pd.DataFrame, out_path: Path) -> None:
    """Box plot of hard AUPRC by dataset and model.

    Args:
        df: Benchmark rows with ``dataset``, ``model``, and ``hard_auprc``.
        out_path: PNG path. No file is written when ``hard_auprc`` is entirely missing.
    """
    sub = df.dropna(subset=["hard_auprc"]).copy()
    if sub.empty:
        return
    sub["label"] = sub["model"].map(lambda k: MODEL_LABELS.get(_model_key(k), str(k)))
    hues = [lab for lab in LABEL_ORDER if lab in set(sub["label"])]
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    sns.boxplot(
        data=sub,
        x="dataset",
        y="hard_auprc",
        hue="label",
        order=list(DATASET_ORDER),
        hue_order=hues,
        palette=LABEL_PALETTE,
        ax=ax,
        fliersize=3,
        linewidth=0.8,
    )
    ax.set_title("Hard AUPRC by seed")
    ax.set_ylabel("Hard AUPRC")
    ax.set_xlabel("Dataset")
    ax.legend(frameon=False, title="", fontsize=8, ncol=2)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    fig.savefig(out_path)
    plt.close(fig)


def plot_uniform_vs_hard_auroc(df: pd.DataFrame, out_path: Path) -> None:
    """Side-by-side uniform and hard AUROC bars (mean ± SD over seeds).

    Args:
        df: Benchmark rows with ``dataset``, ``model``, and AUROC columns.
        out_path: PNG path. No file is written when both panels are empty.
    """
    uni = _seed_metric_frame(df, "uniform_auroc", "auroc")
    hard = _seed_metric_frame(df, "hard_auroc")
    if uni.empty and hard.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharey=True)
    panels = (("Uniform AUROC", uni), ("Hard AUROC", hard))
    hues = [lab for lab in LABEL_ORDER if lab in set(pd.concat([uni, hard])["model"])]
    for ax, (title, panel) in zip(axes, panels):
        if panel.empty:
            ax.set_title(title)
            continue
        sns.barplot(
            data=panel,
            x="dataset",
            y="auprc",
            hue="model",
            order=list(DATASET_ORDER),
            hue_order=[h for h in hues if h in set(panel["model"])],
            palette=LABEL_PALETTE,
            errorbar="sd",
            capsize=0.05,
            ax=ax,
        )
        ax.set_title(title)
        _finish(ax, "AUROC")
        if ax is not axes[0]:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
        else:
            ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    fig.suptitle("All models — AUROC, mean ± SD over seeds", y=1.03)
    fig.tight_layout()
    _footnote(fig, SEED_NOTE)
    fig.savefig(out_path)
    plt.close(fig)


def plot_hard_auprc_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    """Mean hard AUPRC, dataset × model. Blank cells were never trained.

    Args:
        df: Benchmark rows with ``dataset``, ``model``, and ``hard_auprc``.
        out_path: PNG path. No file is written when ``hard_auprc`` is entirely missing.
    """
    sub = df.dropna(subset=["hard_auprc"]).copy()
    if sub.empty:
        return
    sub["model_key"] = sub["model"].map(_model_key)
    pivot = (
        sub.groupby(["dataset", "model_key"], as_index=False)["hard_auprc"]
        .mean()
        .pivot(index="dataset", columns="model_key", values="hard_auprc")
    )
    pivot = pivot.reindex(index=list(DATASET_ORDER), columns=list(MODEL_ORDER))
    pivot.columns = [MODEL_LABELS.get(c, c) for c in pivot.columns]
    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0.55,
        vmax=0.95,
        ax=ax,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Hard AUPRC"},
    )
    ax.set_title("Mean hard AUPRC (3 seeds)")
    ax.set_xlabel("Model")
    ax.set_ylabel("Dataset")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_f1_tau(df: pd.DataFrame, out_path: Path) -> None:
    """Uniform-test F1 at the validation-chosen threshold, mean ± SD.

    Args:
        df: Benchmark rows with ``dataset``, ``model``, and ``f1_tau``.
        out_path: PNG path. No file is written when ``f1_tau`` is missing.
    """
    panel = _seed_metric_frame(df, "f1_tau")
    if panel.empty:
        return
    hues = [lab for lab in LABEL_ORDER if lab in set(panel["model"])]
    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    sns.barplot(
        data=panel,
        x="dataset",
        y="auprc",
        hue="model",
        order=list(DATASET_ORDER),
        hue_order=hues,
        palette=LABEL_PALETTE,
        errorbar="sd",
        capsize=0.05,
        ax=ax,
    )
    ax.set_title("Uniform-test F1 at validation-chosen τ*")
    _finish(ax, "F1")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _footnote(fig, "Heuristic has no learned threshold; it is omitted. " + SEED_NOTE)
    fig.savefig(out_path)
    plt.close(fig)


def plot_learning_curves(results_dir: Path, out_path: Path) -> None:
    """Validation AUPRC versus epoch from ``summary.json`` histories.

    Args:
        results_dir: Root searched recursively for ``summary.json``.
        out_path: PNG path. No file is written when no history rows are found.
    """
    series: dict[tuple[str, str], list[pd.DataFrame]] = {}
    for path in sorted(_result_files(results_dir, "summary.json")):
        blob = json.loads(path.read_text(encoding="utf-8"))
        dataset = blob.get("dataset", path.parent.name)
        for run in blob.get("runs", []):
            history = run.get("history")
            model = _model_key(run.get("model", ""))
            if not history or model in {"heuristic", ""}:
                continue
            hist = pd.DataFrame(history)
            if "epoch" not in hist.columns or "val_auprc" not in hist.columns:
                continue
            series.setdefault((dataset, model), []).append(hist[["epoch", "val_auprc"]])
    if not series:
        return
    datasets = [d for d in DATASET_ORDER if d in {k[0] for k in series}]
    n = len(datasets)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.4, 3.5 * nrows), squeeze=False)
    for ax, dataset in zip(axes.ravel(), datasets):
        for model in MODEL_ORDER:
            runs = series.get((dataset, model))
            if not runs:
                continue
            merged = pd.concat(runs)
            mean = merged.groupby("epoch")["val_auprc"].mean()
            std = merged.groupby("epoch")["val_auprc"].std().fillna(0.0)
            color = MODEL_PALETTE.get(model)
            ax.plot(mean.index, mean.values, label=MODEL_LABELS.get(model, model), color=color, linewidth=1.6)
            ax.fill_between(mean.index, mean.values - std.values, mean.values + std.values, alpha=0.12, color=color)
        ax.set_title(dataset)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val AUPRC")
        ax.legend(frameon=False, fontsize=7, ncol=2)
        ax.grid(linestyle=":", linewidth=0.5, alpha=0.6)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Validation AUPRC learning curves (mean ± SD)", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_comparison_table(df: pd.DataFrame, out_path: Path) -> None:
    """Write mean uniform/hard AUPRC and AUROC, one row per dataset and model.

    Args:
        df: Benchmark rows. ``seed`` is used only to count distinct seeds.
        out_path: CSV path. Parent directories are created.
    """
    rows = []
    for (dataset, model), group in df.groupby(["dataset", "model"]):
        rows.append(
            {
                "dataset": dataset,
                "model": str(model),
                "uniform_auprc": _metric_mean(group, "uniform_auprc", "auprc"),
                "hard_auprc": _metric_mean(group, "hard_auprc"),
                "uniform_auroc": _metric_mean(group, "uniform_auroc", "auroc"),
                "hard_auroc": _metric_mean(group, "hard_auroc"),
                "n_seeds": int(group["seed"].nunique()) if "seed" in group.columns else len(group),
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table["dataset"] = pd.Categorical(table["dataset"], list(DATASET_ORDER), ordered=True)
    table["model"] = pd.Categorical(table["model"], list(MODEL_ORDER), ordered=True)
    table.sort_values(["dataset", "model"]).to_csv(out_path, index=False)


def tidy_benchmark_csvs(results_dir: Path) -> None:
    """Stable column order and model/seed sort so git diffs stay readable.

    Args:
        results_dir: Root searched recursively for ``benchmark.csv``. Files are overwritten.
    """
    preferred = [
        "dataset",
        "model",
        "seed",
        "uniform_auprc",
        "hard_auprc",
        "uniform_auroc",
        "hard_auroc",
        "f1_tau",
        "tau_star",
        "best_val_auprc",
        "method",
        "auroc",
        "auprc",
    ]
    rank = {name: i for i, name in enumerate(MODEL_ORDER)}
    for path in _result_files(results_dir, "benchmark.csv"):
        df = pd.read_csv(path)
        cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
        df = df[cols]
        df["_ord"] = df["model"].map(lambda m: rank.get(_model_key(m), 99))
        sort_cols = [c for c in ("_ord", "seed") if c in df.columns]
        df = df.sort_values(sort_cols).drop(columns=["_ord"])
        df.to_csv(path, index=False)


def select_pr_curve_files(paths: list[Path]) -> dict[str, Path]:
    """Prefer ``pr_ams_seed42.npz`` per dataset (legacy helper used by tests).

    Args:
        paths: PR ``.npz`` paths. The parent directory name is the dataset key.

    Returns:
        One path per dataset directory.
    """
    by_dataset: dict[str, list[Path]] = {}
    for path in paths:
        by_dataset.setdefault(path.parent.name, []).append(path)
    chosen: dict[str, Path] = {}
    for dataset, group in by_dataset.items():
        preferred = [p for p in group if p.stem.endswith("seed42") and "ams" in p.stem]
        if not preferred:
            preferred = [p for p in group if p.stem.endswith("seed42")]
        chosen[dataset] = preferred[0] if preferred else sorted(group)[0]
    return chosen


def collect_pr_curves_seed42(paths: list[Path]) -> dict[str, dict[str, Path]]:
    """Map each dataset and model to one PR ``.npz``, preferring seed 42.

    Args:
        paths: Files named ``pr_<model>_seed<int>.npz``.

    Returns:
        ``dataset -> model -> path``. Stems that do not match are skipped.
    """
    grouped: dict[str, dict[str, list[tuple[int, Path]]]] = {}
    for path in paths:
        stem = path.stem
        if not stem.startswith("pr_") or "_seed" not in stem:
            continue
        body = stem[3:]
        model, _, seed_s = body.rpartition("_seed")
        if not model or not seed_s.isdigit():
            continue
        grouped.setdefault(path.parent.name, {}).setdefault(model, []).append((int(seed_s), path))
    chosen: dict[str, dict[str, Path]] = {}
    for dataset, models in grouped.items():
        chosen[dataset] = {}
        for model, items in models.items():
            seed42 = [p for seed, p in items if seed == 42]
            chosen[dataset][model] = seed42[0] if seed42 else sorted(items)[0][1]
    return chosen


def generate_all_figures(
    results_dir: str | Path,
    output_dir: str | Path,
    skip_tsne: bool = False,
) -> None:
    """Rebuild the paper figures from cached CSVs, summaries, and PR arrays.

    Args:
        results_dir: Root of ``benchmark.csv``, ``ablation.csv``, ``robustness.csv``,
            ``summary.json``, and ``pr_*.npz``.
        output_dir: PNG directory. Created if missing.
        skip_tsne: When true, do not read ``embeddings_*.npz``.
    """
    results_dir = Path(results_dir)
    output_dir = _ensure_dir(Path(output_dir))
    tidy_benchmark_csvs(results_dir)

    merged = _load_benchmarks(results_dir)
    if merged is not None:
        plot_all_models_uniform_hard(merged, output_dir / "fig1_uniform_vs_hard_auprc.png")
        plot_hard_auprc_heatmap(merged, output_dir / "fig5_hard_auprc_heatmap.png")
        plot_delta_hard_auprc(merged, output_dir / "fig6_delta_hard_auprc.png")
        plot_hard_auprc_box(merged, output_dir / "fig7_hard_auprc_seeds.png")
        plot_uniform_vs_hard_auroc(merged, output_dir / "fig8_uniform_vs_hard_auroc.png")
        plot_f1_tau(merged, output_dir / "fig10_f1_at_tau.png")
        write_comparison_table(merged, results_dir / "model_comparison.csv")
    plot_learning_curves(results_dir, output_dir / "fig9_learning_curves.png")

    abl_files = _result_files(results_dir, "ablation.csv")
    if abl_files:
        plot_ablation_ladder(
            pd.concat([pd.read_csv(f) for f in abl_files], ignore_index=True),
            output_dir / "fig2_ablation_ladder.png",
        )
    rob_files = _result_files(results_dir, "robustness.csv")
    if rob_files:
        plot_robustness(
            pd.concat([pd.read_csv(f) for f in rob_files], ignore_index=True),
            output_dir / "fig3_missing_edge_robustness.png",
        )

    pr_files = _result_files(results_dir, "pr_*.npz")
    if pr_files:
        curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        for dataset, models in collect_pr_curves_seed42(pr_files).items():
            curves[dataset] = {}
            for model, path in models.items():
                blob = np.load(path)
                prec = blob["prec"] if "prec" in blob.files else blob["precision"]
                rec = blob["rec"] if "rec" in blob.files else blob["recall"]
                curves[dataset][model] = (prec, rec)
        if curves:
            plot_pr_grid(curves, output_dir / "fig4_precision_recall_curves.png")

    if not skip_tsne:
        plot_saved_embedding_tsne(results_dir, output_dir)


def pr_arrays(labels: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Precision and recall arrays from ``precision_recall_curve``.

    Args:
        labels: 1-D binary labels.
        probs: 1-D scores, same length as ``labels``.

    Returns:
        ``(precision, recall)`` with the threshold column dropped.
    """
    prec, rec, _ = precision_recall_curve(labels, probs)
    return prec, rec


NODE_TYPE_NAMES = {
    "DTI": ("Drug", "Gene"),
    "GDI": ("Gene", "Disease"),
    "DDI": ("Drug", "Drug"),
    "PPI": ("Protein", "Protein"),
}


def plot_embedding_tsne(
    embeddings: np.ndarray,
    n_source: int,
    dataset: str,
    model: str,
    out_path: Path,
    seed: int = 42,
    max_points: int = 4000,
) -> None:
    """Scatter a 2-D t-SNE of node embeddings, colored by node type.

    Args:
        embeddings: Node embeddings ``[N, D]``.
        n_source: First ``n_source`` rows are the source type.
        dataset: Name used for axis labels (``DTI``, ``GDI``, ``DDI``, ``PPI``).
        model: Model key used in the title.
        out_path: PNG path.
        seed: NumPy and t-SNE seed.
        max_points: Subsample size when ``N`` is larger.
    """
    from sklearn.manifold import TSNE

    emb = np.asarray(embeddings, dtype=np.float32)
    n = emb.shape[0]
    src_name, tgt_name = NODE_TYPE_NAMES.get(dataset.upper(), ("Source", "Target"))
    names = np.array([src_name] * n_source + [tgt_name] * (n - n_source))
    rng = np.random.default_rng(seed)
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        emb = emb[idx]
        names = names[idx]
    coords = TSNE(
        n_components=2,
        perplexity=min(30, max(5, len(emb) // 10)),
        init="pca",
        random_state=seed,
    ).fit_transform(emb)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for name, color in ((src_name, "#1f4e79"), (tgt_name, "#c0392b")):
        mask = names == name
        ax.scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.55, c=color, label=name, linewidths=0)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"{MODEL_LABELS.get(_model_key(model), model)} — {dataset}")
    ax.legend(frameon=False, markerscale=2)
    fig.savefig(out_path)
    plt.close(fig)


def plot_saved_embedding_tsne(results_dir: Path, output_dir: Path) -> None:
    """Write one t-SNE figure per saved ``embeddings_*.npz``.

    Args:
        results_dir: Root searched recursively for embedding archives.
        output_dir: PNG directory. Created when at least one archive exists.
    """
    files = sorted(_result_files(results_dir, "embeddings_*.npz"))
    if not files:
        return
    output_dir = _ensure_dir(Path(output_dir))
    for path in files:
        blob = np.load(path, allow_pickle=True)
        dataset = str(blob["dataset"]) if "dataset" in blob.files else path.parent.name
        model = str(blob["model"]) if "model" in blob.files else path.stem
        seed = int(blob["seed"]) if "seed" in blob.files else 42
        n_source = int(blob["n_source"])
        out = output_dir / f"fig_tsne_{dataset}_{model}_seed{seed}.png"
        plot_embedding_tsne(blob["embeddings"], n_source, dataset, model, out, seed=seed)

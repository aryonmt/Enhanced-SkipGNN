"""Load Huang et al. fold-1 CSVs into a leak-free DatasetBundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from src.data.constants import DATA_RAW, DATASET_SPECS
from src.data.normalization import (
    build_binary_skip,
    build_three_hop_return,
    build_weighted_skip,
    identity_sparse,
    laplacian_normalize,
    scipy_to_torch_sparse,
    zscore_columns,
)
from src.data.types import DatasetBundle, SplitArrays


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def _entity_ids(df: pd.DataFrame, preferred: str) -> list[str]:
    col = preferred if preferred in df.columns else df.columns[0]
    return df[col].astype(str).tolist()


def _pair_columns(df: pd.DataFrame, spec_cols: tuple[str, str]) -> tuple[str, str]:
    src, tgt = spec_cols
    if src in df.columns and tgt in df.columns:
        return src, tgt
    labelish = {"label", "y", "target"}
    rest = [c for c in df.columns if c.lower() not in labelish]
    if len(rest) < 2:
        raise ValueError(f"Cannot infer pair columns from {list(df.columns)}")
    return rest[0], rest[1]


def _label_column(df: pd.DataFrame) -> str:
    for name in ("label", "Label", "y"):
        if name in df.columns:
            return name
    return df.columns[-1]


def _build_idx_map(
    dataset: str,
    entity_ids: list[str],
    splits: list[pd.DataFrame],
    src_col: str,
    tgt_col: str,
    bipartite: bool,
) -> tuple[dict[str, int], int, int]:
    if not bipartite:
        ordered = list(dict.fromkeys(entity_ids))
        extras: list[str] = []
        for df in splits:
            extras.extend(df[src_col].astype(str).tolist())
            extras.extend(df[tgt_col].astype(str).tolist())
        for e in extras:
            if e not in ordered:
                ordered.append(e)
        idx_map = {eid: i for i, eid in enumerate(ordered)}
        n = len(idx_map)
        return idx_map, n, n

    src_ids: set[str] = set()
    tgt_ids: set[str] = set()
    for df in splits:
        src_ids.update(df[src_col].astype(str))
        tgt_ids.update(df[tgt_col].astype(str))

    def _unique_extend(dst: list[str], seen: set[str], items) -> None:
        for e in items:
            if e not in seen:
                dst.append(e)
                seen.add(e)

    if dataset == "DTI":
        is_src = lambda e: str(e).startswith("DB")
    elif dataset == "GDI":
        is_src = lambda e: str(e).isdigit()
    else:
        is_src = lambda e: e in src_ids and e not in tgt_ids

    seen: set[str] = set()
    sources: list[str] = []
    targets: list[str] = []
    _unique_extend(sources, seen, [e for e in entity_ids if is_src(e)])
    _unique_extend(targets, seen, [e for e in entity_ids if e not in seen])
    _unique_extend(sources, seen, [e for e in src_ids if is_src(e)])
    _unique_extend(targets, seen, [e for e in tgt_ids if e not in seen])
    leftover = [e for e in (src_ids | tgt_ids | set(entity_ids)) if e not in seen]
    _unique_extend(targets, seen, leftover)

    idx_map = {eid: i for i, eid in enumerate(sources + targets)}
    expected = int(DATASET_SPECS[dataset]["n_source_expected"])
    if abs(len(sources) - expected) > 1:
        raise RuntimeError(
            f"{dataset}: n_source={len(sources)} vs expected {expected}. "
            "Source/target typing heuristic likely failed."
        )
    return idx_map, len(sources), len(targets)


def _pairs_from_df(
    df: pd.DataFrame,
    idx_map: dict[str, int],
    src_col: str,
    tgt_col: str,
    label_col: str,
) -> SplitArrays:
    u = df[src_col].astype(str).map(idx_map)
    v = df[tgt_col].astype(str).map(idx_map)
    mask = u.notna() & v.notna()
    pairs = np.stack(
        [u[mask].to_numpy(dtype=np.int64), v[mask].to_numpy(dtype=np.int64)],
        axis=1,
    )
    labels = df.loc[mask, label_col].to_numpy(dtype=np.float32)
    return SplitArrays(pairs=pairs, labels=labels)


def _orient_bipartite(split: SplitArrays, n_source: int) -> SplitArrays:
    if split.pairs.size == 0:
        return split
    u = split.pairs[:, 0].copy()
    v = split.pairs[:, 1].copy()
    swap = (u >= n_source) & (v < n_source)
    split.pairs[swap, 0] = v[swap]
    split.pairs[swap, 1] = u[swap]
    return split


def _symmetric_adj(n: int, positive_pairs: np.ndarray) -> sp.csr_matrix:
    if positive_pairs.size == 0:
        return sp.csr_matrix((n, n), dtype=np.float32)
    rows = np.concatenate([positive_pairs[:, 0], positive_pairs[:, 1]])
    cols = np.concatenate([positive_pairs[:, 1], positive_pairs[:, 0]])
    data = np.ones(rows.shape[0], dtype=np.float32)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    adj.sum_duplicates()
    adj.data[:] = 1.0
    return adj.tocsr()


def _load_node2vec(emb_path: Path, n_nodes: int) -> np.ndarray:
    if not emb_path.exists():
        raise FileNotFoundError(f"node2vec file missing: {emb_path}")
    emb = pd.read_csv(emb_path, skiprows=1, header=None, sep=" ")
    emb = emb.set_index(0)
    dim = emb.shape[1]
    feats = np.zeros((n_nodes, dim), dtype=np.float32)
    mean = emb.values.mean(axis=0).astype(np.float32)
    index = set(emb.index.tolist())
    for i in range(n_nodes):
        feats[i] = np.asarray(emb.loc[i].values, dtype=np.float32) if i in index else mean
    return zscore_columns(feats)


def load_dataset_splits(
    dataset_name: str,
    data_dir: str | Path | None = None,
    input_type: str = "one_hot",
    device: torch.device | None = None,
) -> DatasetBundle:
    """Load leak-free splits. Adjacency is built from train positives only.

    Reads ``train.csv``, ``val.csv``, ``test.csv``, and the entity table under
    ``data_dir`` (default ``data/raw/<DATASET>``). Laplacian tensors are
    normalized from the train-positive graph. Val and test edges are stored
    as pairs and never added to that graph.

    Args:
        dataset_name: One of ``DTI``, ``DDI``, ``PPI``, ``GDI`` (case-insensitive).
        data_dir: Directory with the fold CSVs. Defaults to the raw data root.
        input_type: ``one_hot`` uses a sparse identity ``[N, N]``. ``node2vec``
            (alias ``node_2vec``) loads the dataset embedding file and
            column-zscores it to dense ``[N, F]``.
        device: Device for feature and Laplacian tensors. Defaults to CPU.

    Returns:
        Bundle whose split pair arrays are int64 ``[E, 2]`` with aligned labels
        ``[E]``, and whose operators ``f_orig``, ``f_skip_bin``,
        ``f_skip_weighted``, and ``f_3hop`` are sparse COO ``[N, N]``.

    Raises:
        ValueError: Unknown dataset name, or pair columns cannot be inferred.
        FileNotFoundError: Missing data directory or node2vec file.
        RuntimeError: Node or source counts disagree with the dataset spec.
    """
    name = dataset_name.upper()
    if name not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset {dataset_name}")
    spec = DATASET_SPECS[name]
    root = Path(data_dir) if data_dir is not None else DATA_RAW / name
    if not root.exists():
        raise FileNotFoundError(f"Missing data directory: {root}. Run scripts/fetch_data.py")

    train_df = _read_csv(root / "train.csv")
    val_df = _read_csv(root / "val.csv")
    test_df = _read_csv(root / "test.csv")
    entity_df = _read_csv(root / spec["entity_file"])

    src_col, tgt_col = _pair_columns(train_df, spec["pair_columns"])
    label_col = _label_column(train_df)
    if spec.get("stringify_source"):
        for frame in (train_df, val_df, test_df):
            frame[src_col] = frame[src_col].astype(str)

    entity_ids = _entity_ids(entity_df, spec["entity_column"])
    idx_map, n_source, n_target = _build_idx_map(
        name,
        entity_ids,
        [train_df, val_df, test_df],
        src_col,
        tgt_col,
        bool(spec["bipartite"]),
    )
    n_nodes = len(idx_map)

    expected_n = int(spec["n_nodes_expected"])
    if abs(n_nodes - expected_n) > 1:
        raise RuntimeError(f"{name}: n_nodes={n_nodes} vs expected {expected_n}")

    train = _pairs_from_df(train_df, idx_map, src_col, tgt_col, label_col)
    val = _pairs_from_df(val_df, idx_map, src_col, tgt_col, label_col)
    test = _pairs_from_df(test_df, idx_map, src_col, tgt_col, label_col)
    if spec["bipartite"]:
        train = _orient_bipartite(train, n_source)
        val = _orient_bipartite(val, n_source)
        test = _orient_bipartite(test, n_source)

    pos_train = train.pairs[train.labels >= 0.5]
    adj_train = _symmetric_adj(n_nodes, pos_train)
    f_orig = laplacian_normalize(adj_train, add_self_loops=True)
    f_skip_bin = laplacian_normalize(build_binary_skip(adj_train), add_self_loops=False)
    f_skip_w = laplacian_normalize(build_weighted_skip(adj_train), add_self_loops=False)
    f_3hop = laplacian_normalize(build_three_hop_return(adj_train), add_self_loops=False)

    known: set[tuple[int, int]] = set()
    for split in (train, val, test):
        pos = split.pairs[split.labels >= 0.5]
        for u, v in pos:
            uu, vv = int(u), int(v)
            known.add((uu, vv))
            if not spec["bipartite"]:
                known.add((vv, uu))

    if device is None:
        device = torch.device("cpu")

    if input_type in {"node2vec", "node_2vec"}:
        features = torch.from_numpy(_load_node2vec(root / spec["emb_file"], n_nodes)).to(device)
        resolved_input = "node2vec"
    else:
        features = identity_sparse(n_nodes, device)
        resolved_input = "one_hot"

    return DatasetBundle(
        name=name,
        n_nodes=n_nodes,
        n_source=n_source,
        n_target=n_target if spec["bipartite"] else n_nodes,
        bipartite=bool(spec["bipartite"]),
        features=features,
        adj_train=adj_train,
        f_orig=scipy_to_torch_sparse(f_orig, device),
        f_skip_bin=scipy_to_torch_sparse(f_skip_bin, device),
        f_skip_weighted=scipy_to_torch_sparse(f_skip_w, device),
        f_3hop=scipy_to_torch_sparse(f_3hop, device),
        train=train,
        val=val,
        test=test,
        known_positives=known,
        input_type=resolved_input,
        idx_map=idx_map,
    )

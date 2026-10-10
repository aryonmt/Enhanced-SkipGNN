from __future__ import annotations

"""Dataset sizes, upstream paths, and default hyperparameters."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
CONFIGS_DIR = REPO_ROOT / "configs"

SKIPGNN_REPO = "https://github.com/kexinhuang12345/SkipGNN.git"
GITHUB_REPO = "https://github.com/aryonmt/Enhanced-SkipGNN.git"

DATASET_SPECS: dict[str, dict[str, Any]] = {
    "DDI": {
        "bipartite": False,
        "n_nodes_expected": 1514,
        "n_source_expected": 1514,
        "pair_columns": ("Drug1_ID", "Drug2_ID"),
        "entity_file": "ddi_unique_smiles.csv",
        "entity_column": "Drug1_ID",
        "emb_file": "ddi.emb",
        "upstream_dir": "data/DDI",
        "onehot_dim_legacy": 1514,
    },
    "PPI": {
        "bipartite": False,
        "n_nodes_expected": 5604,
        "n_source_expected": 5604,
        "pair_columns": ("Protein1_ID", "Protein2_ID"),
        "entity_file": "protein_list.csv",
        "entity_column": "Protein1_ID",
        "emb_file": "ppi.emb",
        "upstream_dir": "data/PPI",
        "onehot_dim_legacy": 5604,
    },
    "DTI": {
        "bipartite": True,
        "n_nodes_expected": 7343,
        "n_source_expected": 5018,
        "pair_columns": ("Drug_ID", "Protein_ID"),
        "entity_file": "entity_list.csv",
        "entity_column": "Entity_ID",
        "emb_file": "dti.emb",
        "upstream_dir": "data/DTI",
        "onehot_dim_legacy": 7343,
    },
    "GDI": {
        "bipartite": True,
        "n_nodes_expected": 19783,
        "n_source_expected": 9413,
        "pair_columns": ("Gene_ID", "Disease_ID"),
        "entity_file": "entity_list.csv",
        "entity_column": "Entity_ID",
        "emb_file": "gdi.emb",
        "upstream_dir": "data/GDI",
        "onehot_dim_legacy": 19783,
        "stringify_source": True,
    },
}


@dataclass(frozen=True)
class DefaultHparams:
    hidden_dim: int = 64
    dropout: float = 0.5
    lr: float = 1e-3
    weight_decay: float = 5e-4
    batch_size: int = 128
    epochs: int = 30
    patience: int = 8
    decoder_hidden: int = 64
    skip_hidden: int = 64
    grad_clip: float = 5.0
    input_type: str = "one_hot"
    seeds_stage1: tuple[int, ...] = (42, 123, 7)
    seeds_full: tuple[int, ...] = (42, 123, 7, 2024, 11)
    missing_fractions: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)

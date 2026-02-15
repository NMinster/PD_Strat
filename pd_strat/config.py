"""
§0 — Paths, configuration, locked analysis plan, and constants.

All tunable parameters and pre-registered settings live here so that
every downstream module imports a single source of truth.
"""

from __future__ import annotations

import os
import json
import random
import argparse
import warnings
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Directory layout ────────────────────────────────────────────────────────
ROOT = Path(".")
OUT  = ROOT / "results"
TAB  = OUT / "tables"
FIG  = OUT / "figures"
ROB  = OUT / "robustness"
for p in (OUT, TAB, FIG, ROB):
    p.mkdir(parents=True, exist_ok=True)

# ── Optional YAML config ───────────────────────────────────────────────────
CFG_PATH = "config.yaml"
CFG: Dict[str, Any] = {}
if os.path.exists(CFG_PATH):
    import yaml
    with open(CFG_PATH, "r", encoding="utf-8") as fh:
        CFG = yaml.safe_load(fh) or {}

# ── Locked analysis plan (frozen for manuscript) ────────────────────────────
LOCKED: Dict[str, Any] = {
    "train_prefix":       "PP-",
    "test_prefix":        "PD-",
    "id_key":             "participant_id",
    "cv_group_col":       "patno",
    "n_cv_folds":         5,
    "exclude_batches":    [],
    "prot_completeness":  0.50,
    "prot_qc_filter":     "PASS",
    "hc_mode":            "global",
    "primary_endpoint":   "updrs_total",
    "clamp_lo":           -9.7,
    "clamp_hi":           106.7,
    "k_range":            range(2, 7),
    "k_selection_method": "bic",
    "ami_threshold":      0.70,
    "cluster_order_method": "pc1",
    "subtype_prefer_visit": "M0",
    "primary_estimand": "participant_level",
    # RidgeSVD hyper-parameters
    "n_svd_components":   128,
    "ridge_alphas":       "logspace(-3, 3, 15)",
    "enet_l1_ratios":     [0.1, 0.5, 0.7, 0.9, 0.95],
    # Panel-aware SVD (v2.1)
    "panel_svd_components": 32,
    # Subtyping & robustness
    "boot_B":             50,
    "boot_frac":          0.80,
    "ensemble_seeds":     [42, 777, 2027],
    # MSI_U
    "msi_u_n_components": 64,
    # Coefficient stability (v2.1)
    "coef_boot_B":        100,
    "coef_boot_frac":     0.80,
    # RNA modality (optional)
    "rna_svd_components": 64,
    "rna_celltype_pcs":   5,
    # Confirmatory protein list
    "confirmatory_n_proteins": 40,
}

# ── Derived constants ──────────────────────────────────────────────────────
RIDGE_ALPHAS       = np.logspace(-3, 3, 15)
ENET_L1_RATIOS     = LOCKED["enet_l1_ratios"]
TRAIN_PREFIX       = LOCKED["train_prefix"].upper()
TEST_PREFIX        = LOCKED["test_prefix"].upper()
Y_LO, Y_HI        = LOCKED["clamp_lo"], LOCKED["clamp_hi"]
PROT_COMPLETENESS_THRESHOLD = LOCKED["prot_completeness"]
PROT_QC_FILTER     = LOCKED["prot_qc_filter"]
HC_MODE            = LOCKED["hc_mode"]
K_SELECTION_METHOD = LOCKED["k_selection_method"]
CLUSTER_ORDER_METHOD = LOCKED["cluster_order_method"]
N_SVD              = LOCKED["n_svd_components"]
PANEL_SVD_NC       = LOCKED["panel_svd_components"]
CFG_PROT_TARGET_N  = int(CFG.get("prot_target_feature_count", 1463))

# ── Proteomics panel paths ─────────────────────────────────────────────────
_DEFAULT_PANELS = {
    "oncology":        r"S:/AMP-PD/releases_2023_v4release_1027_proteomics-PLA-PPEA-D03_olink-explore_protein-expression_PLA-PPEA-D03_oncology.csv",
    "neurology":       r"S:/AMP-PD/releases_2023_v4release_1027_proteomics-PLA-PPEA-D03_olink-explore_protein-expression_PLA-PPEA-D03_neurology.csv",
    "inflammation":    r"S:/AMP-PD/releases_2023_v4release_1027_proteomics-PLA-PPEA-D03_olink-explore_protein-expression_PLA-PPEA-D03_inflammation.csv",
    "cardiometabolic": r"S:/AMP-PD/releases_2023_v4release_1027_proteomics-PLA-PPEA-D03_olink-explore_protein-expression_PLA-PPEA-D03_cardiometabolic.csv",
}
PROTEOMICS_PANELS: Dict[str, str] = CFG.get("proteomics_panels", _DEFAULT_PANELS)

# ── RNA config ──────────────────────────────────────────────────────────────
RNA_COMPLETENESS_THRESHOLD = CFG.get("rna_completeness_threshold", 0.80)
RNA_TARGET_N_GENES = CFG.get("rna_target_n_genes", 5000)
RNA_SVD_NC = CFG.get("rna_svd_components", 64)
RNA_EXCLUDE_BATCHES = CFG.get("rna_exclude_batches",
    ["PP-43", "PP-70", "PP-71", "PP-74", "PP-75"])

# ── Manifest / meta paths ──────────────────────────────────────────────────
MANIFEST_PATH = TAB / "feature_manifest.json"
META_PATH     = TAB / "zmeta.json"

# ── Determinism ────────────────────────────────────────────────────────────
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ── CLI flags ──────────────────────────────────────────────────────────────
_ap = argparse.ArgumentParser(description="PD-Deep Precision Suite v2.1")
_ap.add_argument("--skip_robustness", action="store_true", default=False)
try:
    FLAGS = _ap.parse_args([])
except SystemExit:
    FLAGS = _ap.parse_args()


def print_banner():
    """Print startup banner with configuration summary."""
    print("=" * 70)
    print("PD-Deep Precision Suite v2.1  (PROT + RNA . RidgeSVD . Reviewer Fixes)")
    print("=" * 70)
    print(f"  TRAIN cohort prefix  : {TRAIN_PREFIX}")
    print(f"  TEST  cohort prefix  : {TEST_PREFIX}")
    print(f"  Proteomics panels    : {list(PROTEOMICS_PANELS.keys())}")
    print(f"  QC filter            : Cumulative_QC == {PROT_QC_FILTER}")
    print(f"  Protein completeness : >={PROT_COMPLETENESS_THRESHOLD*100:.0f}%")
    print(f"  HC anchoring mode    : {HC_MODE}")
    print(f"  SVD components       : {N_SVD} (global), {PANEL_SVD_NC} (per-panel)")
    print(f"  K selection          : min BIC (sil>0, AMI>={LOCKED['ami_threshold']})")
    print(f"  Prediction clamp     : [{Y_LO}, {Y_HI}]")
    print(f"  Primary estimand     : {LOCKED['primary_estimand']}")
    print("=" * 70)

"""
§0 — Paths, configuration, locked analysis plan, and constants.

All tunable parameters and pre-registered settings live here so that
every downstream module imports a single source of truth.

Resolution order for settings:
    1. command-line flags   (python run.py --data_dir S:/AMP-PD ...)
    2. config.yaml          (--config path, default: ./config.yaml)
    3. built-in defaults    (below)
"""

from __future__ import annotations

import os
import sys
import json
import random
import argparse
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── CLI flags ──────────────────────────────────────────────────────────────
# parse_known_args() so the package also imports cleanly inside Jupyter
# (which injects its own argv) and ignores flags it does not know about.
_ap = argparse.ArgumentParser(
    description="PD-Deep Precision Suite v2.1",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
_ap.add_argument("--config", default="config.yaml",
                 help="YAML config file (relative to CWD or project root)")
_ap.add_argument("--data_dir", default=None,
                 help="Folder with the raw AMP-PD release files "
                      "(overrides config.yaml data_dir)")
_ap.add_argument("--out_dir", default=None,
                 help="Where to write results/ (default: <project>/results)")
_ap.add_argument("--skip_robustness", action="store_true", default=False,
                 help="Skip the (slow) robustness package")
_ap.add_argument("--skip_assembly", action="store_true", default=False,
                 help="Reuse an existing results/tables/clinical_unified.csv")
_ap.add_argument("--force_assembly", action="store_true", default=False,
                 help="Rebuild clinical_unified.csv even if it exists")
_ap.add_argument("--no_rna", action="store_true", default=False,
                 help="Disable the optional RNA modality")
_ap.add_argument("--report_only", action="store_true", default=False,
                 help="Only (re)generate SUMMARY_REPORT.md from a previous run")
FLAGS, _UNKNOWN_ARGS = _ap.parse_known_args()


# ── Optional YAML config ───────────────────────────────────────────────────
def _locate_config(p: str) -> Optional[Path]:
    cands = [Path(p), Path.cwd() / p, PROJECT_ROOT / p]
    for c in cands:
        if c.is_file():
            return c.resolve()
    return None


CFG_PATH = _locate_config(FLAGS.config)
CFG: Dict[str, Any] = {}
if CFG_PATH is not None:
    import yaml
    with open(CFG_PATH, "r", encoding="utf-8") as fh:
        CFG = yaml.safe_load(fh) or {}
else:
    print(f"[Config] No config file found for '{FLAGS.config}' -- "
          f"using built-in defaults")


def _cfg(key: str, default=None):
    v = CFG.get(key)
    return default if v is None else v


# ── Directory layout ───────────────────────────────────────────────────────
DATA_DIR = Path(FLAGS.data_dir or _cfg("data_dir", "S:/AMP-PD"))
OUT = Path(FLAGS.out_dir or _cfg("out_dir", PROJECT_ROOT / "results"))
ROOT = OUT.parent
TAB  = OUT / "tables"
FIG  = OUT / "figures"
ROB  = OUT / "robustness"
for p in (OUT, TAB, FIG, ROB):
    p.mkdir(parents=True, exist_ok=True)

# ── Locked analysis plan (frozen for manuscript) ────────────────────────────
LOCKED: Dict[str, Any] = {
    "train_prefix":       str(_cfg("train_prefix", "PP-")),
    "test_prefix":        str(_cfg("test_prefix", "PD-")),
    "id_key":             "participant_id",
    "cv_group_col":       "patno",
    "n_cv_folds":         5,
    "exclude_batches":    [],
    "prot_completeness":  float(_cfg("prot_completeness", 0.50)),
    "prot_qc_filter":     str(_cfg("prot_qc_filter", "PASS")),
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
    "n_svd_components":   int(_cfg("n_svd_components", 128)),
    "ridge_alphas":       "logspace(-3, 3, 15)",
    "enet_l1_ratios":     [0.1, 0.5, 0.7, 0.9, 0.95],
    # Panel-aware SVD (v2.1)
    "panel_svd_components": int(_cfg("panel_svd_components", 32)),
    # Subtyping & robustness
    "boot_B":             int(_cfg("boot_B", 50)),
    "boot_frac":          0.80,
    "ensemble_seeds":     [42, 777, 2027],
    # MSI_U
    "msi_u_n_components": 64,
    # Coefficient stability (v2.1)
    "coef_boot_B":        int(_cfg("coef_boot_B", 100)),
    "coef_boot_frac":     0.80,
    # RNA modality (optional)
    "rna_svd_components": int(_cfg("rna_svd_components", 64)),
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
CFG_PROT_TARGET_N  = int(_cfg("prot_target_feature_count", 1463))

# ── Feature selection (§2a) ────────────────────────────────────────────────
FEATURE_SELECTION  = str(_cfg("feature_selection", "mad_corr_cap"))   # | variance_topn
PROT_MIN_OBS_FRAC  = float(_cfg("prot_min_obs_frac", 0.30))
PROT_MIN_MAD       = float(_cfg("prot_min_mad", 0.01))
PROT_CORR_THRESH   = float(_cfg("prot_corr_thresh", 0.95))
PROT_FEATURE_CAP   = int(_cfg("prot_feature_cap", 1168))

# ── Population for the severity model ──────────────────────────────────────
# "all"     : every TRAIN/TEST row with UPDRS (PD cases + healthy controls)
# "pd_only" : PD cases only.  The validity package always reports the other.
SEVERITY_POPULATION = str(_cfg("severity_population", "all")).lower()

# ── Resampling / inference knobs ───────────────────────────────────────────
N_PERMUTATIONS  = int(_cfg("n_permutations", 100))
STABILITY_B     = int(_cfg("stability_B", 200))
PAIRED_BOOT_B   = int(_cfg("paired_boot_B", 1000))
PLR_BOOT_B      = int(_cfg("participant_boot_B", 2000))
CUMULATIVE_K_GRID = list(_cfg("cumulative_k_grid",
    [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]))
PROGRESSION_MIN_VISITS       = int(_cfg("progression_min_visits", 2))
PROGRESSION_MIN_SPAN_MONTHS  = float(_cfg("progression_min_span_months", 3))
TTE_ENDPOINTS = _cfg("tte_endpoints", None)     # optional explicit column mapping

# ── Raw AMP-PD release files ───────────────────────────────────────────────
RELEASE_PREFIX = str(_cfg("release_prefix", "releases_2023_v4release_1027"))


def _data_path(name: str) -> str:
    """Resolve a file name against DATA_DIR unless it is already absolute."""
    p = Path(name)
    return str(p if p.is_absolute() else DATA_DIR / p)


_DEFAULT_CLINICAL_FILES = {
    "updrs_i":      f"{RELEASE_PREFIX}_clinical_MDS_UPDRS_Part_I.csv",
    "updrs_ii":     f"{RELEASE_PREFIX}_clinical_MDS_UPDRS_Part_II.csv",
    "updrs_iii":    f"{RELEASE_PREFIX}_clinical_MDS_UPDRS_Part_III.csv",
    "updrs_iv":     f"{RELEASE_PREFIX}_clinical_MDS_UPDRS_Part_IV.csv",
    "upsit":        f"{RELEASE_PREFIX}_clinical_UPSIT.csv",
    "demographics": "Demographics.csv",
    "case_control": f"{RELEASE_PREFIX}_amp_pd_case_control.csv",
    # optional — silently skipped when absent
    "med_history":  f"{RELEASE_PREFIX}_clinical_PD_Medical_History.csv",
    "datscan":      f"{RELEASE_PREFIX}_clinical_DaTSCAN_SBR.csv",
    "time_to_event": "endpoints_time_to_event.csv",
}
CLINICAL_FILES: Dict[str, str] = {
    k: _data_path(v) for k, v in
    {**_DEFAULT_CLINICAL_FILES, **(_cfg("clinical_files", {}) or {})}.items()
}
CASE_CONTROL_PATH = _data_path(_cfg("case_control_path",
                                    CLINICAL_FILES["case_control"]))
# keep legacy key available to modules that still read CFG directly
CFG["case_control_path"] = CASE_CONTROL_PATH

# Assembly options
ASSEMBLY: Dict[str, Any] = {
    "updrs3_prefer_off_state": bool(_cfg("updrs3_prefer_off_state", True)),
    "updrs_part_iv_missing_as_zero": bool(_cfg("updrs_part_iv_missing_as_zero", True)),
    "keep_other_cohorts": bool(_cfg("keep_other_cohorts", False)),
    "upsit_fill_from_baseline": bool(_cfg("upsit_fill_from_baseline", True)),
}

# ── Proteomics panel paths ─────────────────────────────────────────────────
PROTEOMICS_TISSUE = str(_cfg("proteomics_tissue", "PLA")).upper()   # PLA | CSF
_PANEL_NAMES = list(_cfg("proteomics_panel_names",
                         ["oncology", "neurology", "inflammation",
                          "cardiometabolic"]))
_DEFAULT_PANELS = {
    name: (f"{RELEASE_PREFIX}_proteomics-{PROTEOMICS_TISSUE}-PPEA-D03_"
           f"olink-explore_protein-expression_{PROTEOMICS_TISSUE}-PPEA-D03_"
           f"{name}.csv")
    for name in _PANEL_NAMES
}
PROTEOMICS_PANELS: Dict[str, str] = {
    k: _data_path(v) for k, v in
    (_cfg("proteomics_panels", None) or _DEFAULT_PANELS).items()
}

# ── RNA config (optional modality; off unless rna_path is given) ───────────
RNA_PATH: Optional[str] = None
if not FLAGS.no_rna and _cfg("rna_path"):
    RNA_PATH = _data_path(str(_cfg("rna_path")))
RNA_COMPLETENESS_THRESHOLD = float(_cfg("rna_completeness_threshold", 0.80))
RNA_TARGET_N_GENES = int(_cfg("rna_target_n_genes", 5000))
RNA_SVD_NC = int(_cfg("rna_svd_components", 64))
RNA_EXCLUDE_BATCHES = list(_cfg("rna_exclude_batches",
    ["PP-43", "PP-70", "PP-71", "PP-74", "PP-75"]))
RNA_LOG1P = _cfg("rna_log1p", "auto")          # auto | true | false

# ── Manifest / meta paths ──────────────────────────────────────────────────
MANIFEST_PATH = TAB / "feature_manifest.json"
META_PATH     = TAB / "zmeta.json"

# ── Determinism ────────────────────────────────────────────────────────────
SEED = int(_cfg("seed", 42))
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)


def print_banner():
    """Print startup banner with configuration summary."""
    print("=" * 70)
    print("PD-Deep Precision Suite v2.1  (PROT + RNA . RidgeSVD . Reviewer Fixes)")
    print("=" * 70)
    print(f"  Config file          : {CFG_PATH or '(defaults)'}")
    print(f"  Data dir             : {DATA_DIR}")
    print(f"  Output dir           : {OUT}")
    print(f"  TRAIN cohort prefix  : {TRAIN_PREFIX}")
    print(f"  TEST  cohort prefix  : {TEST_PREFIX}")
    print(f"  Proteomics tissue    : {PROTEOMICS_TISSUE}")
    print(f"  Proteomics panels    : {list(PROTEOMICS_PANELS.keys())}")
    print(f"  QC filter            : Cumulative_QC == {PROT_QC_FILTER}")
    print(f"  Protein completeness : >={PROT_COMPLETENESS_THRESHOLD*100:.0f}%")
    print(f"  HC anchoring mode    : {HC_MODE}")
    print(f"  RNA modality         : {RNA_PATH or 'disabled'}")
    print(f"  SVD components       : {N_SVD} (global), {PANEL_SVD_NC} (per-panel)")
    print(f"  K selection          : min BIC (sil>0, AMI>={LOCKED['ami_threshold']})")
    print(f"  Prediction clamp     : [{Y_LO}, {Y_HI}]")
    print(f"  Primary estimand     : {LOCKED['primary_estimand']}")
    print(f"  Severity population  : {SEVERITY_POPULATION}")
    print(f"  Feature selection    : {FEATURE_SELECTION} "
          f"(obs>={PROT_MIN_OBS_FRAC:.0%}, MAD>={PROT_MIN_MAD}, |r|<={PROT_CORR_THRESH}, "
          f"cap={PROT_FEATURE_CAP})")
    print(f"  Permutations / stability B : {N_PERMUTATIONS} / {STABILITY_B}")
    print(f"  Robustness           : {'SKIPPED' if FLAGS.skip_robustness else 'enabled'}")
    print("=" * 70)

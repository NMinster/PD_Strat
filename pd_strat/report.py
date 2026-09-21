"""
§13 — Human-readable summary report.

Builds ``results/SUMMARY_REPORT.md`` from the in-memory run summary
(``utils.summary``) plus the CSV tables written by each stage, and prints
the same text to the console.  Can also be regenerated after the fact from
``results/tables/summary.json`` via ``python run.py --report_only``.
"""

from __future__ import annotations

import json
import datetime as _dt
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import (
    OUT, TAB, FIG, ROB, DATA_DIR, CFG_PATH, TRAIN_PREFIX, TEST_PREFIX,
    PROTEOMICS_PANELS, PROTEOMICS_TISSUE, PROT_COMPLETENESS_THRESHOLD,
    PROT_QC_FILTER, N_SVD, PANEL_SVD_NC, RNA_PATH, SEED,
)

REPORT_PATH = OUT / "SUMMARY_REPORT.md"

_METRIC_ORDER = ["n", "spearman", "pearson", "mae", "rmse", "r2",
                 "cal_slope", "cal_intercept"]


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  formatting helpers                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _f(x: Any, nd: int = 3) -> str:
    """Format a scalar for a table cell."""
    if x is None:
        return "–"
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return "–"
        return f"{x:.{nd}f}"
    s = str(x).replace("|", "\\|").replace("\n", " ")
    return s if len(s) <= 160 else s[:157] + "..."


def _is_scalar(x: Any) -> bool:
    return x is None or isinstance(x, (str, int, float, bool,
                                       np.integer, np.floating, np.bool_))


def _md_table(rows: List[Dict[str, Any]], cols: Optional[List[str]] = None,
              index_name: str = "") -> List[str]:
    if not rows:
        return ["_(no data)_", ""]
    if cols is None:
        seen: Dict[str, None] = {}
        for r in rows:
            for k in r:
                if k != "_name":
                    seen.setdefault(k, None)
        cols = list(seen)
    head = ([index_name] if index_name else []) + cols
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join(["---"] * len(head)) + "|"]
    for r in rows:
        cells = ([str(r.get("_name", ""))] if index_name else [])
        cells += [_f(r.get(c)) for c in cols]
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    return out


def _metrics_table(d: Dict[str, Dict[str, Any]], index_name="model") -> List[str]:
    """Render {name: {metric: value}} as a table."""
    rows = []
    for name, mets in (d or {}).items():
        if isinstance(mets, dict):
            # bootstrap CI keys are reported in their own section
            rows.append({"_name": name,
                         **{k: v for k, v in mets.items() if "_ci_" not in k}})
    cols = [c for c in _METRIC_ORDER
            if any(c in r for r in rows)]
    extra = sorted({k for r in rows for k in r
                    if k not in cols and k != "_name" and _is_scalar(r[k])})
    return _md_table(rows, cols + extra, index_name)


def _ci_table(d: Dict[str, Dict[str, Dict[str, float]]]) -> List[str]:
    """Render {split: {metric: {point, lo, hi}}}."""
    rows = []
    for split, mets in (d or {}).items():
        if not isinstance(mets, dict):
            continue
        for metric, v in mets.items():
            if isinstance(v, dict) and "point" in v:
                rows.append({"_name": f"{split} / {metric}",
                             "estimate": v.get("point"),
                             "95% CI low": v.get("lo"),
                             "95% CI high": v.get("hi")})
    return _md_table(rows, ["estimate", "95% CI low", "95% CI high"], "split / metric")


def _generic(obj: Any, depth: int = 0, max_depth: int = 3) -> List[str]:
    """Best-effort rendering of nested dicts (used for robustness etc.)."""
    lines: List[str] = []
    if not isinstance(obj, dict):
        return [f"{obj}", ""]
    scalars = {k: v for k, v in obj.items() if _is_scalar(v)}
    nested  = {k: v for k, v in obj.items() if not _is_scalar(v)}
    if scalars:
        lines += _md_table([{"_name": k, "value": v} for k, v in scalars.items()],
                           ["value"], "key")
    for k, v in nested.items():
        if depth >= max_depth:
            lines.append(f"- **{k}**: _(nested, see summary.json)_")
            continue
        lines.append(f"{'#' * min(6, 3 + depth)} {k}")
        lines.append("")
        if isinstance(v, dict):
            if v and all(isinstance(x, dict) for x in v.values()) \
                    and all(all(_is_scalar(y) for y in x.values()) for x in v.values()):
                lines += _metrics_table(v, index_name="")
            else:
                lines += _generic(v, depth + 1, max_depth)
        elif isinstance(v, list):
            if v and all(isinstance(x, dict) for x in v):
                lines += _md_table(v[:50])
            else:
                lines.append(", ".join(_f(x) for x in v[:50]))
                lines.append("")
        else:
            lines.append(f"{v}")
            lines.append("")
    return lines


def _csv_table(path: Path, max_rows: int = 30) -> List[str]:
    if not path.exists():
        return [f"_({path.name} not found)_", ""]
    try:
        df = pd.read_csv(path)
    except Exception as e:  # pragma: no cover
        return [f"_(could not read {path.name}: {e})_", ""]
    if df.empty:
        return ["_(empty)_", ""]
    df = df.head(max_rows)
    rows = df.to_dict("records")          # keeps per-column dtypes
    return _md_table(rows, list(df.columns))


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  report builder                                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def build_report(summary: Dict[str, Any]) -> str:
    L: List[str] = []
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    L += [f"# PD-Deep Precision Suite v{summary.get('version', '2.1')} — Summary Report",
          "", f"_Generated {now}_", ""]

    # ── 1. Run configuration ────────────────────────────────────────────
    L += ["## 1. Run configuration", ""]
    L += _md_table([
        {"_name": "Config file", "value": str(CFG_PATH or "(defaults)")},
        {"_name": "Data directory", "value": str(DATA_DIR)},
        {"_name": "Output directory", "value": str(OUT)},
        {"_name": "TRAIN / TEST prefix", "value": f"{TRAIN_PREFIX} / {TEST_PREFIX}"},
        {"_name": "Proteomics", "value": f"{PROTEOMICS_TISSUE}: "
                                         f"{', '.join(PROTEOMICS_PANELS)}"},
        {"_name": "QC filter", "value": f"Cumulative_QC == {PROT_QC_FILTER}"},
        {"_name": "Protein completeness", "value": f">= {PROT_COMPLETENESS_THRESHOLD:.0%}"},
        {"_name": "SVD components", "value": f"{N_SVD} global / {PANEL_SVD_NC} per panel"},
        {"_name": "RNA modality", "value": RNA_PATH or "disabled"},
        {"_name": "Primary model", "value": summary.get("primary_model", "–")},
        {"_name": "Seed", "value": SEED},
    ], ["value"], "setting")

    # ── 2. Sample flow ──────────────────────────────────────────────────
    L += ["## 2. Sample flow", ""]
    L += _csv_table(TAB / "sample_flow_table.csv")
    us = summary.get("updrs_scale")
    if isinstance(us, dict):
        L += ["**UPDRS scale (TRAIN, usable rows)**", ""]
        L += _md_table([us], ["min", "max", "mean", "std", "median", "iqr"])

    # ── 3. Severity prediction ──────────────────────────────────────────
    L += ["## 3. Severity prediction (MDS-UPDRS total)", ""]
    L += [f"Primary model selected on external TEST Spearman ρ: "
          f"**{summary.get('primary_model', '–')}**", ""]
    L += ["### 3a. Cross-validated OOF (TRAIN, participant-grouped 5-fold)", ""]
    L += _metrics_table(summary.get("oof_models", {}))
    L += ["### 3b. External TEST (fit on all TRAIN)", ""]
    L += _metrics_table(summary.get("test_raw", {}))
    cal = summary.get("calibration")
    if isinstance(cal, dict):
        L += ["### 3c. Calibration", ""]
        L += _md_table([{"_name": "OOF linear recalibration",
                         "slope": cal.get("oof_slope"),
                         "intercept": cal.get("oof_intercept")}],
                       ["slope", "intercept"], "fit")
        if isinstance(cal.get("recal_test"), dict) and cal["recal_test"]:
            L += ["Recalibrated TEST metrics:", ""]
            rt = cal["recal_test"]
            if all(isinstance(v, dict) for v in rt.values()):
                L += _metrics_table(rt)
            else:
                L += _md_table([rt], None)
    if summary.get("bootstrap_ci"):
        L += ["### 3d. Bootstrap 95% CIs (participant-level resampling)", ""]
        L += _ci_table(summary["bootstrap_ci"])
    if summary.get("participant_level"):
        L += ["### 3e. Participant-level estimand (one prediction per participant)", ""]
        L += _metrics_table(summary["participant_level"], index_name="split")
        if summary.get("participant_ci"):
            L += _ci_table(summary["participant_ci"])
    ig = summary.get("incremental_gain")
    if ig:
        L += ["### 3f. Incremental gain over covariates (OOF)", ""]
        L += _metrics_table(ig)
    sb = summary.get("severity_bands")
    if sb:
        L += ["### 3g. Severity-band classification / decision analysis", ""]
        L += _generic(sb, depth=1)

    # ── 4. RNA / fusion ─────────────────────────────────────────────────
    if summary.get("rna_oof") or summary.get("late_fusion"):
        L += ["## 4. RNA modality and late fusion", ""]
        if summary.get("rna_oof"):
            L += ["**RNA-only OOF**", ""] + _metrics_table(summary["rna_oof"])
        if summary.get("rna_test"):
            L += ["**RNA-only TEST**", ""] + _metrics_table(summary["rna_test"])
        if summary.get("late_fusion"):
            L += ["**Protein + RNA late fusion**", ""] + _generic(summary["late_fusion"], 1)
    else:
        L += ["## 4. RNA modality", "", "_RNA modality not run (disabled or "
              "insufficient samples)._", ""]

    # ── 5. MSI_U + subtypes ─────────────────────────────────────────────
    L += ["## 5. Unsupervised severity index (MSI_U) and molecular subtypes", ""]
    L += _md_table([
        {"_name": "MSI_U ρ vs UPDRS (TRAIN)", "value": summary.get("msi_u_rho_train")},
        {"_name": "MSI_U ρ vs UPDRS (TEST)", "value": summary.get("msi_u_rho_test")},
        {"_name": "MSI_U variance explained (PC1)", "value": summary.get("msi_u_var_pc1")},
        {"_name": "MSI_U n TRAIN / TEST",
         "value": f"{summary.get('msi_u_n_train', '–')} / {summary.get('msi_u_n_test', '–')}"},
        {"_name": "Subtype K", "value": summary.get("subtype_K")},
        {"_name": "K selection", "value": summary.get("subtype_method")},
        {"_name": "η² (UPDRS | subtype)", "value": summary.get("eta2_updrs")},
        {"_name": "η² (UPSIT | subtype)", "value": summary.get("eta2_upsit")},
    ], ["value"], "quantity")
    K = summary.get("subtype_K")
    if K is not None:
        L += ["**K-selection grid (TRAIN-PD)**", ""]
        L += _csv_table(TAB / "subtype_grid_TRAINPD.csv")
        L += [f"**Cluster summary (K={K})**", ""]
        L += _csv_table(TAB / f"subtypes_TRAINPD_K{K}_summary.csv")

    # ── 6. Confounding ──────────────────────────────────────────────────
    conf = summary.get("msi_u_confounding")
    if conf:
        L += ["## 6. Confounding audit (sex / site / age)", ""]
        L += _generic(conf, depth=1)

    # ── 7. Robustness ───────────────────────────────────────────────────
    rob = summary.get("robustness")
    L += ["## 7. Robustness package", ""]
    if rob:
        L += _generic(rob, depth=1)
    else:
        L += ["_Robustness package skipped (`--skip_robustness`) or failed._", ""]

    # ── 8. Outputs ──────────────────────────────────────────────────────
    L += ["## 8. Output files", ""]
    for name, d in (("tables", TAB), ("figures", FIG), ("robustness", ROB)):
        files = sorted(p.name for p in d.glob("*") if p.is_file())
        L += [f"**{name}/** ({len(files)} files): " +
              (", ".join(f"`{f}`" for f in files) if files else "_none_"), ""]
    L += [f"Full machine-readable summary: `{TAB / 'summary.json'}`", ""]
    return "\n".join(L)


def write_summary_report(summary: Optional[Dict[str, Any]] = None,
                         echo: bool = True) -> Path:
    """Write SUMMARY_REPORT.md (and print it)."""
    if summary is None:
        p = TAB / "summary.json"
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found -- run the pipeline first")
        summary = json.load(open(p, encoding="utf-8"))
    text = build_report(summary)
    REPORT_PATH.write_text(text, encoding="utf-8")
    if echo:
        print("\n" + text)
    print(f"\n[Report] written to {REPORT_PATH}")
    return REPORT_PATH

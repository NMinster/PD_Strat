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

    # ── 3h-3m. Validity package ─────────────────────────────────────────
    val = summary.get("validity")
    if val:
        L += ["### 3h. Severity or diagnosis? (HC-vs-PD discrimination, within-PD ρ)", ""]
        L += [f"Primary population: **{val.get('primary_population', 'all')}**. "
              "`rho_within_pd` is the rank correlation restricted to PD rows; "
              "`auroc_pd_vs_hc` is how well the *same* score separates PD from HC.", ""]
        svd = val.get("severity_vs_diagnosis", {})
        if svd:
            L += _md_table([{"_name": k, **v} for k, v in svd.items()],
                           ["n_pd_participants", "rho_all", "rho_within_pd",
                            "rho_within_pd_participant", "rho_within_hc",
                            "auroc_pd_vs_hc_participant", "mean_pred_hc", "mean_pred_pd",
                            "n_pd_rows", "n_hc_rows"], "split")
        for key, title in (("refit_pd_only", "Model refit on PD cases only"),
                           ("refit_all", "Model refit on all rows (PD + HC)")):
            rf = val.get(key)
            if rf and "OOF_row" in rf:
                L += [f"**{title}** (TRAIN {rf['n_train_participants']} participants / "
                      f"{rf['n_train_rows']} rows; TEST {rf.get('n_test_rows', 0)} rows)", ""]
                tbl = {}
                for s in ("OOF", "TEST"):
                    if f"{s}_row" in rf:
                        tbl[f"{s} row-level"] = rf[f"{s}_row"]
                    if f"{s}_participant" in rf:
                        tbl[f"{s} participant-mean"] = rf[f"{s}_participant"]
                L += _metrics_table(tbl, index_name="level")
                ci = {s: rf[f"{s}_participant_ci"] for s in ("OOF", "TEST")
                      if f"{s}_participant_ci" in rf}
                if ci:
                    L += _ci_table(ci)
        ec = val.get("extended_covariates")
        if ec:
            L += ["### 3i. Extended covariate adjustment", ""]
            L += [f"Covariates: {', '.join(ec.get('covariates', []))}", ""]
            for s in ("OOF", "TEST"):
                if s in ec:
                    L += [f"**{s}** (partial ρ of proteomics given covariates = "
                          f"{_f(ec[s].get('partial_rho_prot_given_cov'))})", ""]
                    L += _metrics_table({k: v for k, v in ec[s].items() if isinstance(v, dict)})
            if ec.get("strata"):
                L += ["**Strata**", ""]
                L += _md_table([{"_name": k, "n": v["n"], "rho": v["rho"]}
                                for k, v in ec["strata"].items()], ["n", "rho"], "stratum")
        if val.get("cross_endpoint"):
            L += ["### 3j. Cross-endpoint validation (participant-level Spearman ρ, 95% CI)", ""]
            ce = pd.DataFrame(val["cross_endpoint"])
            ce = ce[ce["population"] == "PD"] if "population" in ce else ce
            L += _md_table(ce.to_dict("records"),
                           ["split", "endpoint", "n_participants", "rho_participant",
                            "ci_lo_participant", "ci_hi_participant", "n_rows", "rho_rows"])
        es = val.get("error_structure")
        if es:
            L += ["### 3k. Error structure and range shift", ""]
            L += _metrics_table({k: v for k, v in es.items() if isinstance(v, dict)},
                                index_name="split")
            if es.get("table"):
                L += _md_table(es["table"], ["split", "stratum", "bin", "n", "mae", "bias"])
        tr = val.get("target_recalibration")
        if tr:
            L += ["### 3l. Recalibration in the target cohort (cross-fitted within TEST)", ""]
            L += [tr.get("note", ""), ""]
            L += _metrics_table({k: {kk: vv for kk, vv in v.items() if kk != "participant"}
                                 for k, v in tr.items() if isinstance(v, dict)})
        if val.get("model_comparison"):
            L += ["### 3m. Paired participant-level model comparison (Δρ = comparator − primary)", ""]
            L += _md_table(val["model_comparison"],
                           ["split", "comparator", "n_participants", "rho_primary",
                            "rho_comparator", "delta_rho", "delta_ci_lo", "delta_ci_hi",
                            "p_two_sided"])

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

    # ── 5b. Progression ─────────────────────────────────────────────────
    pr = summary.get("progression")
    if pr:
        L += ["## 5b. Progression and prognostic value (PD participants)", ""]
        for s in ("TRAIN", "TEST"):
            if f"{s}_slope" in pr and pr[f"{s}_slope"].get("n_participants", 0) >= 20:
                d = pr[f"{s}_slope"]
                L += [f"**{s}** — n = {d['n_participants']} participants with slopes "
                      f"(median span {_f(pr.get(f'{s}_slope_summary', {}).get('median_span_months'), 1)} months)", ""]
                L += _md_table([
                    {"_name": "ρ(baseline UPDRS, slope)", "value": d.get("rho_baselineUPDRS_slope")},
                    {"_name": "ρ(baseline proteomic severity, slope)", "value": d.get("rho_pred0_slope"),
                     "95% CI": f"[{_f(d.get('rho_pred0_slope_ci', [np.nan]*2)[0])}, {_f(d.get('rho_pred0_slope_ci', [np.nan]*2)[1])}]"},
                    {"_name": "ρ(residual proteomic severity | baseline UPDRS, slope)",
                     "value": d.get("rho_pred0_resid_slope"),
                     "95% CI": f"[{_f(d.get('rho_pred0_resid_slope_ci', [np.nan]*2)[0])}, {_f(d.get('rho_pred0_resid_slope_ci', [np.nan]*2)[1])}]"},
                ], ["value", "95% CI"], "association")
                if "ols_slope_on_pred0_and_baseline" in d:
                    L += ["Combined OLS (HC3 SE), slope ~ z(pred0) + z(baseline UPDRS):", ""]
                    L += _md_table([d["ols_slope_on_pred0_and_baseline"]])
            if f"{s}_mixed" in pr:
                mm = {k: v for k, v in pr[f"{s}_mixed"].items() if isinstance(v, dict) and "beta_pred0_x_year" in v}
                if mm:
                    L += [f"**{s}** mixed model UPDRS ~ pred0_z × years (+ baseline UPDRS) + (1 | participant)", ""]
                    L += _md_table([{"_name": k, **{kk: vv for kk, vv in v.items() if kk != 'formula'}}
                                    for k, v in mm.items()], None, "model")
        if pr.get("cox"):
            L += ["**Time-to-event (Cox, HR per SD of baseline proteomic severity)**", ""]
            L += _md_table(pr["cox"], ["split", "endpoint", "model", "n", "n_events",
                                       "HR_per_SD_pred0", "HR_ci_lo", "HR_ci_hi", "p", "harrell_C"])
        if pr.get("subtype_progression"):
            L += ["**Progression by molecular subtype (TRAIN)**", ""]
            L += _generic(pr["subtype_progression"], depth=2)

    # ── 6. Confounding ──────────────────────────────────────────────────
    conf = summary.get("msi_u_confounding")
    if conf:
        L += ["## 6. Confounding audit (sex / site / age)", ""]
        L += _generic(conf, depth=1)

    # ── 7. Robustness ───────────────────────────────────────────────────
    rob = summary.get("robustness")
    L += ["## 7. Robustness package", ""]
    if rob:
        if "perm_empirical_p" in rob:
            L += [f"Permutation null (n = {rob.get('perm_n')}): mean ρ = {_f(rob.get('perm_rho_mean'))}, "
                  f"max = {_f(rob.get('perm_rho_max'))}; observed OOF ρ = "
                  f"{_f(summary.get('oof_spearman'))} → empirical p = {_f(rob['perm_empirical_p'], 4)}.", ""]
        L += _generic({k: v for k, v in rob.items() if k != "perm_rhos"}, depth=1)
    else:
        L += ["_Robustness package skipped (`--skip_robustness`) or failed._", ""]

    # ── 7b. Panel reduction ─────────────────────────────────────────────
    prd = summary.get("panel_reduction")
    if prd:
        L += ["## 7b. Reduced panel: stability selection and nested cumulative curve", ""]
        L += [f"k* = **{prd.get('k_star')}** proteins ({prd.get('k_star_rule')}); "
              f"OOF ρ at k* = {_f(prd.get('oof_rho_at_k_star'))} vs full {_f(prd.get('oof_rho_full'))}; "
              f"TEST ρ at k* = {_f(prd.get('test_rho_at_k_star'))} vs full {_f(prd.get('test_rho_full'))}.", ""]
        L += [f"Stable proteins (inclusion ≥ 80% at k = 50): {prd.get('n_stable_k50_freq80')}; "
              f"sign consistency all / top-20 / top-40 = {_f(prd.get('mean_sign_consistency_all'))} / "
              f"{_f(prd.get('mean_sign_consistency_top20'))} / {_f(prd.get('mean_sign_consistency_top40'))}.", ""]
        if prd.get("stable_set_test"):
            L += ["Stability-selected set on TEST:", ""] + _md_table([prd["stable_set_test"]])
        L += _csv_table(ROB / "cumulative_importance.csv")
        L += ["Top 20 by stability-weighted importance:", ""]
        L += _csv_table(ROB / "stability_selection.csv", max_rows=20)

    # ── 7c. Confirmatory ────────────────────────────────────────────────
    cf = summary.get("confirmatory")
    if cf:
        L += ["## 7c. Confirmatory protein analysis (locked list, TEST replication)", ""]
        L += ["TRAIN p-values are descriptive (proteins were selected on TRAIN); "
              "the confirmatory evidence is TEST replication.", ""]
        if cf.get("severity"):
            L += ["**Cross-sectional severity**", ""] + _md_table([cf["severity"]])
        if cf.get("progression"):
            L += ["**Baseline protein × time (adjusted for baseline UPDRS)**", ""] + _md_table([cf["progression"]])
        L += ["Top 15 proteins (sorted by TEST p):", ""]
        L += _csv_table(ROB / "confirmatory_severity.csv", max_rows=15)

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

#!/usr/bin/env python3
"""
PD-Deep Precision Suite v2.1 — main orchestrator.

Imports and runs every pipeline stage in order:
  §0  Configuration + banner
  §1  Clinical data loading
  §2  Proteomics + RNA loading
  §3  Feature alignment (aligned matrices, SVD)
  §4  Targets, masks, CV setup
  §5  OOF severity, covariate baselines, TEST evaluation
  §6  Calibration, bootstrap CIs, participant-level eval
  §6c Severity band classification, decile / quintile analysis
  §6d RNA severity + late fusion
  §7  MSI_U unsupervised severity index + GMM subtyping
  §9  Confounding audit
  §10 Robustness package
  §12 Publication figures
  §13 Final summary
"""

from __future__ import annotations

import json
import traceback

import numpy as np
import pandas as pd

from .config import (
    print_banner, TAB, FLAGS, SEED, LOCKED,
)
from .utils import summary, summary_update, save_flow_table
from .data_loading import (
    load_clinical, build_cohort_masks, load_proteomics,
    completeness_audit, load_rna,
)
from .features import build_aligned_matrices
from .severity_model import (
    setup_targets_and_cv, run_primary_oof,
    run_covariate_baselines, run_test_evaluation,
)
from .calibration import (
    run_calibration, run_participant_level_eval,
)
from .clinical_analysis import run_severity_band_analysis
from .rna_pipeline import run_rna_pipeline
from .subtyping import run_msi_u_and_subtyping
from .confounding import run_confounding_audit
from .robustness import run_robustness
from .figures import run_figures


def main():
    """Run the full PD-Deep Precision Suite pipeline."""

    # ── §0  Banner ────────────────────────────────────────────────────
    print_banner()

    # ── §1  Clinical data ─────────────────────────────────────────────
    clin, id_key, SEX_COL, AGE_COL, SITE_COL = load_clinical()
    cohort = build_cohort_masks(clin)
    is_train = cohort["is_train"]
    is_test  = cohort["is_test"]
    is_pd    = cohort["is_pd"]
    is_pd_train = cohort["is_pd_train"]
    is_pd_test  = cohort["is_pd_test"]

    # ── §2  Proteomics ────────────────────────────────────────────────
    z_prot, panel_protein_map = load_proteomics(
        clin, is_train, is_test)
    completeness_audit(z_prot, clin, is_train)

    # ── §2c RNA ───────────────────────────────────────────────────────
    z_rna, HAS_RNA = load_rna(clin, is_train, z_prot)

    # ── §3  Feature alignment ─────────────────────────────────────────
    feat = build_aligned_matrices(z_prot, z_rna, HAS_RNA,
                                  panel_protein_map)
    X_prot = feat["X_prot"]
    M_prot = feat["M_prot"]
    X_rna  = feat["X_rna"]
    M_rna  = feat["M_rna"]
    d_prot = feat["d_prot"]
    d_rna  = feat["d_rna"]
    prot_cols     = feat["prot_cols"]
    prot_cols_set = feat["prot_cols_set"]
    PANEL_COL_INDICES = feat["PANEL_COL_INDICES"]
    USE_PANEL_AWARE   = feat["USE_PANEL_AWARE"]

    # ── §4  Targets, masks, CV ────────────────────────────────────────
    tv = setup_targets_and_cv(clin, z_prot, M_prot, is_train, is_test)
    y_all        = tv["y_all"]
    train_idx    = tv["train_idx"]
    test_idx     = tv["test_idx"]
    y_tr         = tv["y_tr"]
    has_any_omics  = tv["has_any_omics"]
    prot_ok_train  = tv["prot_ok_train"]
    prot_ok_test   = tv["prot_ok_test"]
    train_idx_y  = tv["train_idx_y"]
    y_tr_nonan   = tv["y_tr_nonan"]
    groups_all   = tv["groups_all"]
    groups_train = tv["groups_train"]
    gkf          = tv["gkf"]
    test_idx_omics = tv["test_idx_omics"]
    y_te_full    = tv["y_te_full"]
    has_test_y   = tv["has_test_y"]

    # ── §5  OOF severity ──────────────────────────────────────────────
    baseline_oof, _oof_preds = run_primary_oof(
        X_prot, M_prot, y_all, train_idx_y, groups_train, gkf,
        PANEL_COL_INDICES, USE_PANEL_AWARE)

    # ── §5b Covariate baselines ───────────────────────────────────────
    covariate_results = run_covariate_baselines(
        clin, X_prot, M_prot, y_all,
        train_idx_y, groups_train, gkf,
        PANEL_COL_INDICES, USE_PANEL_AWARE)

    # ── §5c TEST evaluation + model selection ─────────────────────────
    (test_results, _test_preds, PRIMARY_LABEL,
     oof_pred, rho_oof, oof_mae, oof_rmse,
     test_pred) = run_test_evaluation(
        X_prot, M_prot, y_all, train_idx_y,
        test_idx, test_idx_omics, y_te_full, has_test_y,
        prot_ok_test, PANEL_COL_INDICES, USE_PANEL_AWARE,
        baseline_oof, _oof_preds)

    rho_test = test_results.get(PRIMARY_LABEL, {}).get("spearman", np.nan)

    # ── §6  Calibration + bootstrap CIs ───────────────────────────────
    (cal_slope, cal_intercept, recal_test_results,
     all_metrics, ci_results) = run_calibration(
        oof_pred, y_all, train_idx_y,
        test_pred, prot_ok_test,
        y_te_full, has_test_y,
        oof_mae, PRIMARY_LABEL,
        baseline_oof, test_results, covariate_results,
        clin.index, test_idx_omics)

    # ── §6b Participant-level evaluation ──────────────────────────────
    plr_results, plr_ci = run_participant_level_eval(
        clin, y_all, train_idx_y, oof_pred,
        test_idx_omics, test_pred, prot_ok_test,
        y_te_full, has_test_y)

    # ── §6c Severity band + decile analysis ───────────────────────────
    severity_band_results = run_severity_band_analysis(
        clin, y_all, train_idx_y, oof_pred,
        test_idx_omics, test_pred, prot_ok_test,
        y_te_full, has_test_y)

    # ── §6d RNA pipeline + late fusion ────────────────────────────────
    rna_results = run_rna_pipeline(
        clin, X_rna, M_rna, d_rna, HAS_RNA,
        X_prot, M_prot,
        y_all, train_idx_y, test_idx, test_idx_omics,
        groups_all, is_train,
        prot_ok_test,
        oof_pred, test_pred, rho_oof, rho_test,
        PANEL_COL_INDICES, USE_PANEL_AWARE)

    # ── §7-8 MSI_U + GMM subtyping ───────────────────────────────────
    sub_results = run_msi_u_and_subtyping(
        clin, z_prot, X_prot, M_prot,
        y_all, tv["upsit_all"],
        is_pd_train, is_pd_test)
    K = sub_results["K"]

    # ── §9  Confounding audit ─────────────────────────────────────────
    run_confounding_audit(sub_results)

    # ── §10 Robustness ────────────────────────────────────────────────
    if not FLAGS.skip_robustness:
        try:
            run_robustness(
                clin, z_prot, X_prot, M_prot, y_all,
                train_idx, test_idx, train_idx_y, y_tr_nonan,
                groups_all, groups_train, gkf,
                test_idx_omics, y_te_full, has_test_y, prot_ok_test,
                prot_ok_train, has_any_omics,
                PANEL_COL_INDICES, USE_PANEL_AWARE,
                PRIMARY_LABEL, baseline_oof, oof_pred,
                rho_oof, rho_test, d_prot, prot_cols, prot_cols_set,
                sub_results)
        except Exception as e:
            print(f"[Robustness] WARN: {e}")
            traceback.print_exc()
    else:
        print("\n[Robustness SKIPPED per --skip_robustness flag]")

    # ── §12 Figures ───────────────────────────────────────────────────
    try:
        run_figures(
            oof_pred=oof_pred,
            y_tr=y_all[train_idx_y],
            train_idx_y=train_idx_y,
            test_pred=test_pred,
            test_idx=test_idx,
            test_idx_omics=test_idx_omics,
            y_all=y_all,
            prot_ok_test=prot_ok_test,
            K=K,
            labs_trpd=sub_results["labs_trpd"],
            Zs_trpd=sub_results["Zs"],
            pd_ids_clean=sub_results["pd_ids_clean"],
            y_pd_clean=sub_results["y_pd_clean"],
            Mr=M_rna,
            Mp=M_prot,
            z_rna=z_rna if HAS_RNA else None,
            z_prot=z_prot,
            d_rna=d_rna,
            d_prot=d_prot,
            has_any_omics=has_any_omics,
        )
    except Exception as e:
        print(f"[Figures] WARN: {e}")
        traceback.print_exc()

    # ── §13 Flow table + final summary ────────────────────────────────
    save_flow_table()

    print(f"\n{'=' * 70}")
    print("PD-Deep Precision Suite v2.1 — COMPLETE")
    print(f"{'=' * 70}")
    for k in ("n_train", "n_test", "d_rna", "d_prot",
              "oof_spearman", "primary_model",
              "subtype_K", "eta2_updrs", "eta2_upsit"):
        print(f"  {k}: {summary.get(k)}")
    print(f"\n  Outputs in: {TAB.parent}/")

    # Save final summary JSON
    try:
        with open(TAB / "summary.json", "w") as fh:
            json.dump(
                {k: (v if not isinstance(v, np.floating) else float(v))
                 for k, v in summary.items()},
                fh, indent=2, default=str)
    except Exception:
        pass

    return summary


if __name__ == "__main__":
    main()

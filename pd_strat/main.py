#!/usr/bin/env python3
"""
PD-Deep Precision Suite v2.1 — main orchestrator.

Runs every pipeline stage in order:
  §0  Configuration + banner
  §0b Assemble clinical_unified.csv from the raw AMP-PD files
  §1  Clinical data loading
  §2  Proteomics + RNA loading
  §3  Feature alignment (aligned matrices, SVD)
  §4  Targets, masks, CV setup
  §5  OOF severity, TEST evaluation + model selection, covariate baselines
  §6  Calibration, bootstrap CIs, participant-level eval
  §6c Severity band classification, decile / quintile analysis
  §6d RNA severity + late fusion (optional)
  §7  MSI_U unsupervised severity index + GMM subtyping
  §9  Confounding audit
  §10 Robustness package (optional, slow)
  §12 Publication figures
  §13 Sample-flow table + SUMMARY_REPORT.md
"""

from __future__ import annotations

import sys
import time
import traceback

import numpy as np

from .config import (print_banner, FLAGS, TAB, SEVERITY_POPULATION,
                     PROT_COMPLETENESS_THRESHOLD)
from .utils import summary, summary_update, save_flow_table


def _stage(name: str, fn, *args, optional: bool = False, **kwargs):
    """Run one stage with timing; optional stages log-and-continue on error."""
    t0 = time.time()
    try:
        out = fn(*args, **kwargs)
        print(f"  [{name}] done in {time.time() - t0:.1f}s")
        return out
    except Exception as e:
        if not optional:
            raise
        print(f"  [{name}] WARN: stage failed and was skipped: {e}")
        traceback.print_exc()
        return None


def main():
    """Run the full PD-Deep Precision Suite pipeline."""
    from .report import write_summary_report

    if FLAGS.report_only:
        write_summary_report(None)
        return None

    from .assemble_clinical import assemble_clinical
    from .data_loading import (
        load_clinical, build_cohort_masks, load_proteomics,
        completeness_audit, load_rna,
    )
    from .features import build_aligned_matrices
    from .severity_model import (
        setup_targets_and_cv, run_primary_oof,
        run_covariate_baselines, run_test_evaluation,
    )
    from .calibration import run_calibration, run_participant_level_eval
    from .clinical_analysis import run_severity_band_analysis
    from .rna_pipeline import run_rna_pipeline
    from .subtyping import run_msi_u_and_subtyping
    from .confounding import run_confounding_audit
    from .robustness import run_robustness
    from .figures import run_figures, run_extended_figures
    from .validity import run_validity
    from .progression import run_progression
    from .panel_reduction import run_panel_reduction
    from .confirmatory import run_confirmatory

    t_start = time.time()

    # ── §0  Banner ────────────────────────────────────────────────────
    print_banner()

    # ── §0b Assemble clinical table from raw AMP-PD files ─────────────
    if not FLAGS.skip_assembly:
        _stage("assembly", assemble_clinical, force=FLAGS.force_assembly)

    # ── §1  Clinical data ─────────────────────────────────────────────
    clin, id_key, SEX_COL, AGE_COL, SITE_COL = load_clinical()
    cohort = build_cohort_masks(clin)
    is_train    = cohort["is_train"]
    is_test     = cohort["is_test"]
    is_pd_train = cohort["is_pd_train"]
    is_pd_test  = cohort["is_pd_test"]
    summary_update({
        "n_rows": int(len(clin)),
        "n_participants": int(clin.index.nunique()),
        "n_train": int(is_train.sum()), "n_test": int(is_test.sum()),
    })

    # ── §2  Proteomics ────────────────────────────────────────────────
    print(f"\n{'=' * 60}\nPROTEOMICS LOADING\n{'=' * 60}")
    z_prot, panel_protein_map = load_proteomics(clin, is_train)
    if z_prot.shape[1] == 0:
        raise RuntimeError(
            "No proteomics features loaded -- check 'proteomics_panels' / "
            "'data_dir' in config.yaml (see the [WARN] lines above).")
    completeness_audit(z_prot, is_train, is_test, panel_protein_map)

    # ── §2c RNA (optional) ────────────────────────────────────────────
    z_rna, HAS_RNA = load_rna(clin, is_train)

    # ── §3  Feature alignment ─────────────────────────────────────────
    feat = build_aligned_matrices(z_prot, z_rna, clin.index,
                                  panel_protein_map, HAS_RNA)
    X_prot, M_prot = feat["X_prot"], feat["M_prot"]
    X_rna,  M_rna  = feat["X_rna"],  feat["M_rna"]
    d_prot, d_rna  = feat["d_prot"], feat["d_rna"]
    prot_cols, prot_cols_set = feat["prot_cols"], feat["prot_cols_set"]
    PANEL_COL_INDICES = feat["PANEL_COL_INDICES"]
    USE_PANEL_AWARE   = feat["USE_PANEL_AWARE"]
    summary_update({"d_prot": int(d_prot), "d_rna": int(d_rna),
                    "panel_aware": bool(USE_PANEL_AWARE)})

    # ── §4  Targets, masks, CV ────────────────────────────────────────
    if SEVERITY_POPULATION == "pd_only":
        print(f"\n[Population] severity model restricted to PD cases "
              f"(severity_population=pd_only)")
        is_train_sev = is_train & cohort["is_pd_flag"]
        is_test_sev = is_test & cohort["is_pd_flag"]
    else:
        is_train_sev, is_test_sev = is_train, is_test
    tv = setup_targets_and_cv(clin, z_prot, M_prot, is_train_sev, is_test_sev)
    if SEVERITY_POPULATION == "pd_only":
        # index sets for the all-population sensitivity refit in the validity package
        comp_all = z_prot.notna().mean(axis=1).values
        ok_all = (comp_all >= PROT_COMPLETENESS_THRESHOLD) & (M_prot.sum(1) > 0) \
                 & np.isfinite(tv["y_all"])
        tv["train_idx_y_all"] = np.where(is_train.values & ok_all)[0]
        tv["test_idx_omics_all"] = np.where(is_test.values & ok_all)[0]
    y_all, upsit_all = tv["y_all"], tv["upsit_all"]
    train_idx, test_idx = tv["train_idx"], tv["test_idx"]
    has_any_omics = tv["has_any_omics"]
    prot_ok_train, prot_ok_test = tv["prot_ok_train"], tv["prot_ok_test"]
    train_idx_y, y_tr_nonan = tv["train_idx_y"], tv["y_tr_nonan"]
    groups_all, groups_train, gkf = tv["groups_all"], tv["groups_train"], tv["gkf"]
    test_idx_omics = tv["test_idx_omics"]
    y_te_full, has_test_y = tv["y_te_full"], tv["has_test_y"]
    UPDRS_STATS = tv["UPDRS_STATS"]

    if len(train_idx_y) < 20:
        raise RuntimeError(
            f"Only {len(train_idx_y)} TRAIN rows have both UPDRS and "
            f">= {tv['prot_ok_train'].mean():.0%} proteomics completeness -- "
            "cannot train. Check cohort prefixes and the completeness audit.")

    # ── §5a OOF severity (all candidates) ─────────────────────────────
    baseline_oof, _oof_preds, oof_ridge, oof_mono_ridge = run_primary_oof(
        X_prot, M_prot, y_all, train_idx_y, groups_train, gkf,
        PANEL_COL_INDICES, USE_PANEL_AWARE, UPDRS_STATS)

    # ── §5c TEST evaluation + data-driven model selection ─────────────
    (test_results, _test_preds, PRIMARY_LABEL,
     oof_pred, rho_oof, oof_mae, oof_rmse, test_pred) = run_test_evaluation(
        X_prot, M_prot, y_all, train_idx_y,
        test_idx, test_idx_omics, y_te_full, has_test_y, prot_ok_test,
        PANEL_COL_INDICES, USE_PANEL_AWARE, baseline_oof, _oof_preds)
    rho_test = test_results.get(PRIMARY_LABEL, {}).get("spearman", np.nan)
    summary_update({"oof_spearman": float(rho_oof) if np.isfinite(rho_oof) else None,
                    "test_spearman": float(rho_test) if np.isfinite(rho_test) else None})

    # ── §5b Covariate baselines (needs PRIMARY_LABEL) ─────────────────
    covariate_results = _stage(
        "covariate_baselines", run_covariate_baselines,
        clin, X_prot, M_prot, y_all, train_idx_y, y_tr_nonan,
        groups_train, gkf, PANEL_COL_INDICES, USE_PANEL_AWARE,
        baseline_oof, PRIMARY_LABEL, optional=True) or {}

    # ── §6  Calibration + bootstrap CIs ───────────────────────────────
    _stage("calibration", run_calibration,
           oof_pred, y_all, train_idx_y, test_pred, prot_ok_test,
           y_te_full, has_test_y, oof_mae, PRIMARY_LABEL,
           baseline_oof, test_results, covariate_results,
           clin.index, test_idx_omics, optional=True)

    # ── §6b Participant-level evaluation ──────────────────────────────
    _stage("participant_level", run_participant_level_eval,
           clin, y_all, train_idx_y, oof_pred,
           test_idx_omics, test_pred, prot_ok_test,
           y_te_full, has_test_y, optional=True)

    # ── §6c Severity band + decile analysis ───────────────────────────
    _stage("severity_bands", run_severity_band_analysis,
           clin, y_all, train_idx_y, oof_pred,
           test_idx_omics, test_pred, prot_ok_test,
           y_te_full, has_test_y, optional=True)

    # ── §6e Validity package ──────────────────────────────────────────
    _stage("validity", run_validity,
           clin, z_prot, X_prot, M_prot, y_all, tv, cohort,
           PANEL_COL_INDICES, USE_PANEL_AWARE, PRIMARY_LABEL,
           oof_pred, test_pred, _oof_preds, _test_preds,
           severity_population=SEVERITY_POPULATION, optional=True)

    # ── §6d RNA pipeline + late fusion ────────────────────────────────
    if HAS_RNA:
        _stage("rna_pipeline", run_rna_pipeline,
               clin, X_rna, M_rna, d_rna, HAS_RNA, X_prot, M_prot,
               y_all, train_idx_y, test_idx, test_idx_omics,
               groups_all, is_train, prot_ok_test,
               oof_pred, test_pred, rho_oof, rho_test,
               PANEL_COL_INDICES, USE_PANEL_AWARE, optional=True)
    else:
        print("\n[RNA] modality disabled or unavailable -- fusion skipped")

    # ── §7-8 MSI_U + GMM subtyping ───────────────────────────────────
    sub_results = run_msi_u_and_subtyping(
        clin, z_prot, X_prot, M_prot, y_all, upsit_all,
        is_pd_train, is_pd_test)
    K = sub_results["K"]

    # ── §9  Confounding audit ─────────────────────────────────────────
    _stage("confounding", run_confounding_audit, sub_results, optional=True)

    # ── §6f Progression / prognostic value ────────────────────────────
    _stage("progression", run_progression,
           clin, y_all, train_idx_y, oof_pred, test_idx_omics,
           test_pred[prot_ok_test] if len(test_idx_omics) else np.array([]),
           cohort, sub_results, optional=True)

    # ── §10 Robustness ────────────────────────────────────────────────
    if FLAGS.skip_robustness:
        print("\n[Robustness SKIPPED per --skip_robustness]")
    else:
        _stage("robustness", run_robustness,
               clin, z_prot, X_prot, M_prot, y_all,
               train_idx, test_idx, train_idx_y, y_tr_nonan,
               groups_all, groups_train, gkf,
               test_idx_omics, y_te_full, has_test_y, prot_ok_test,
               prot_ok_train, has_any_omics,
               PANEL_COL_INDICES, USE_PANEL_AWARE,
               PRIMARY_LABEL, baseline_oof, oof_pred,
               rho_oof, rho_test, d_prot, prot_cols, prot_cols_set,
               sub_results, optional=True)

        # ── §11 Panel reduction (stability selection, nested cumulative) ─
        _stage("panel_reduction", run_panel_reduction,
               clin, X_prot, y_all, train_idx_y, groups_train, gkf,
               test_idx_omics, y_te_full, prot_cols, PANEL_COL_INDICES,
               optional=True)

        # ── §10n Confirmatory protein analysis with TEST replication ──
        _stage("confirmatory", run_confirmatory,
               clin, z_prot, y_all, train_idx_y, test_idx_omics, prot_cols,
               cohort, optional=True)

    # ── §12 Figures ───────────────────────────────────────────────────
    _stage("figures", run_figures,
           oof_pred=oof_pred, y_tr=y_all[train_idx_y], train_idx_y=train_idx_y,
           test_pred=test_pred, test_idx=test_idx, test_idx_omics=test_idx_omics,
           y_all=y_all, prot_ok_test=prot_ok_test,
           K=K, labs_trpd=sub_results["labs_trpd"], Zs_trpd=sub_results["Zs"],
           pd_ids_clean=sub_results["pd_ids_clean"],
           y_pd_clean=sub_results["y_pd_clean"],
           upsit_pd_train=None, Mr=M_rna, Mp=M_prot,
           z_rna=z_rna if HAS_RNA else None, z_prot=z_prot,
           d_rna=d_rna, d_prot=d_prot, has_any_omics=has_any_omics,
           optional=True)
    _stage("extended_figures", run_extended_figures, optional=True)

    # ── §13 Flow table + summary report ───────────────────────────────
    save_flow_table()
    summary_update({"runtime_sec": round(time.time() - t_start, 1)})
    write_summary_report(summary)

    print(f"\n{'=' * 70}")
    print(f"PD-Deep Precision Suite v2.1 -- COMPLETE "
          f"({(time.time() - t_start) / 60:.1f} min)")
    print(f"{'=' * 70}")
    for k in ("n_train", "n_test", "d_prot", "d_rna", "primary_model",
              "oof_spearman", "test_spearman", "subtype_K",
              "eta2_updrs", "eta2_upsit"):
        print(f"  {k:<16}: {summary.get(k)}")
    svd = summary.get("validity", {}).get("severity_vs_diagnosis", {})
    for split in ("OOF", "TEST"):
        if split in svd:
            print(f"  {split:<4} within-PD rho (participant): "
                  f"{svd[split].get('rho_within_pd_participant')}  "
                  f"AUROC PD vs HC: {svd[split].get('auroc_pd_vs_hc_participant')}")
    print(f"\n  Outputs : {TAB.parent}")
    print(f"  Report  : {TAB.parent / 'SUMMARY_REPORT.md'}")
    return summary


if __name__ == "__main__":
    main()

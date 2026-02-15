"""
§6–6b — Calibration (Platt-style linear recalibration), bootstrap CIs,
and participant-level evaluation (primary estimand).
"""

from __future__ import annotations

from typing import Dict, Any, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

from .config import TAB, Y_LO, Y_HI, SEED
from .utils import (
    map_participant_id, full_metrics, spearman_np, bootstrap_ci,
    summary_update,
)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  §6  CALIBRATION                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def run_calibration(oof_pred, y_all, train_idx_y,
                    test_pred, prot_ok_test,
                    y_te_full, has_test_y,
                    oof_mae, PRIMARY_LABEL,
                    baseline_oof, test_results, covariate_results,
                    clin_index, test_idx_omics):
    """Platt-style recalibration + bootstrap CIs.

    Returns (cal_slope, cal_intercept, recal_test_results,
             all_metrics, ci_results).
    """
    print(f"\n{'=' * 60}")
    print("CALIBRATION: Platt-style linear recalibration + calibration curves")
    print(f"{'=' * 60}")

    cal_slope, cal_intercept = np.nan, np.nan
    recal_test_results: Dict[str, Any] = {}

    m_oof = np.isfinite(oof_pred) & np.isfinite(y_all[train_idx_y])
    if m_oof.sum() >= 20:
        cal_coefs = np.polyfit(oof_pred[m_oof], y_all[train_idx_y][m_oof], 1)
        cal_slope, cal_intercept = float(cal_coefs[0]), float(cal_coefs[1])
        print(f"  TRAIN OOF calibration: slope={cal_slope:.4f}, "
              f"intercept={cal_intercept:.2f}")

        oof_recal = np.clip(cal_slope * oof_pred + cal_intercept,
                            Y_LO, Y_HI).astype(np.float32)
        mets_oof_recal = full_metrics(oof_recal, y_all[train_idx_y])
        print(f"  OOF after recalibration: MAE={mets_oof_recal['mae']:.2f} "
              f"(before: {oof_mae:.2f})")

        # Apply to TEST
        te_pred_raw = test_pred[prot_ok_test]
        te_pred_finite = np.isfinite(te_pred_raw)
        if np.isfinite(cal_slope) and te_pred_finite.any():
            te_pred_recal = np.clip(
                cal_slope * te_pred_raw + cal_intercept,
                Y_LO, Y_HI).astype(np.float32)
            mets_recal = full_metrics(te_pred_recal, y_te_full)
            if mets_recal["n"] >= 5:
                cal2 = np.polyfit(
                    te_pred_recal[np.isfinite(te_pred_recal) & has_test_y],
                    y_te_full[np.isfinite(te_pred_recal) & has_test_y], 1)
                mets_recal["post_recal_slope"] = float(cal2[0])
                mets_recal["post_recal_intercept"] = float(cal2[1])
                recal_test_results = mets_recal
                print(f"\n  Recalibrated TEST (n={mets_recal['n']}): "
                      f"rho_s={mets_recal['spearman']:.3f}, "
                      f"MAE={mets_recal['mae']:.2f}, R2={mets_recal['r2']:.3f}")

        # Binned calibration curve data
        def _calibration_bins(pred, true, n_bins=10):
            m = np.isfinite(pred) & np.isfinite(true)
            p, t = pred[m], true[m]
            if len(p) < n_bins:
                return None
            bins = np.percentile(p, np.linspace(0, 100, n_bins + 1))
            rows = []
            for i in range(n_bins):
                lo, hi = bins[i], bins[i + 1]
                mask = (p >= lo) & (p <= hi) if i == n_bins - 1 else (p >= lo) & (p < hi)
                if mask.sum() > 0:
                    rows.append({"bin": i, "pred_mean": float(p[mask].mean()),
                                 "obs_mean": float(t[mask].mean()),
                                 "obs_std": float(t[mask].std()),
                                 "n": int(mask.sum())})
            return pd.DataFrame(rows)

        cal_bins_oof = _calibration_bins(oof_pred, y_all[train_idx_y])
        if cal_bins_oof is not None:
            cal_bins_oof.to_csv(TAB / "calibration_bins_oof.csv", index=False)
        if te_pred_finite.any() and has_test_y.any():
            cal_bins_raw = _calibration_bins(te_pred_raw, y_te_full)
            if cal_bins_raw is not None:
                cal_bins_raw.to_csv(TAB / "calibration_bins_test_raw.csv", index=False)
            if np.isfinite(cal_slope):
                cal_bins_recal = _calibration_bins(te_pred_recal, y_te_full)
                if cal_bins_recal is not None:
                    cal_bins_recal.to_csv(TAB / "calibration_bins_test_recal.csv", index=False)
    else:
        print("  [SKIP] Insufficient OOF predictions for calibration")

    summary_update({"calibration": {
        "oof_slope": cal_slope, "oof_intercept": cal_intercept,
        "recal_test": recal_test_results,
    }})

    # Combined metrics table
    all_metrics = {}
    for name, mets in baseline_oof.items():
        all_metrics[f"{name}_OOF"] = mets
    for k, v in test_results.items():
        all_metrics[f"{k}_TEST_raw"] = v
    if recal_test_results:
        all_metrics[f"{PRIMARY_LABEL}_TEST_recalibrated"] = recal_test_results
    for name, mets in covariate_results.items():
        all_metrics[f"baseline_{name}_OOF"] = mets

    # Bootstrap 95% CIs
    print(f"\n{'=' * 60}")
    print("BOOTSTRAP 95% CIs (participant-level, B=2000)")
    print(f"{'=' * 60}")

    ci_results: Dict[str, Dict] = {}
    if len(train_idx_y) >= 20 and oof_pred.size > 0:
        ci_oof = bootstrap_ci(train_idx_y, oof_pred, y_all[train_idx_y],
                              clin_index)
        ci_results["OOF"] = ci_oof
        for metric in ("spearman", "pearson", "mae", "rmse", "r2"):
            d = ci_oof[metric]
            print(f"  OOF {metric:>8s}: {d['point']:.3f} "
                  f"[{d['lo']:.3f}, {d['hi']:.3f}]")
            all_metrics.setdefault(f"{PRIMARY_LABEL}_OOF", {})
            all_metrics[f"{PRIMARY_LABEL}_OOF"][f"{metric}_ci_lo"] = d["lo"]
            all_metrics[f"{PRIMARY_LABEL}_OOF"][f"{metric}_ci_hi"] = d["hi"]

    if test_idx_omics.size > 0 and has_test_y.sum() >= 5:
        te_pred_ci = test_pred[prot_ok_test]
        ci_test = bootstrap_ci(test_idx_omics, te_pred_ci, y_te_full,
                               clin_index)
        ci_results["TEST"] = ci_test
        for metric in ("spearman", "pearson", "mae", "rmse", "r2"):
            d = ci_test[metric]
            print(f"  TEST {metric:>8s}: {d['point']:.3f} "
                  f"[{d['lo']:.3f}, {d['hi']:.3f}]")
            all_metrics.setdefault(f"{PRIMARY_LABEL}_TEST_raw", {})
            all_metrics[f"{PRIMARY_LABEL}_TEST_raw"][f"{metric}_ci_lo"] = d["lo"]
            all_metrics[f"{PRIMARY_LABEL}_TEST_raw"][f"{metric}_ci_hi"] = d["hi"]

    ci_rows = []
    for split, ci_dict in ci_results.items():
        for metric, vals in ci_dict.items():
            ci_rows.append({"split": split, "metric": metric, **vals})
    if ci_rows:
        pd.DataFrame(ci_rows).to_csv(TAB / "bootstrap_ci.csv", index=False)

    summary_update({"bootstrap_ci": ci_results})
    pd.DataFrame(all_metrics).T.to_csv(TAB / "all_severity_metrics.csv")
    print(f"  Saved: {TAB}/all_severity_metrics.csv")

    return cal_slope, cal_intercept, recal_test_results, all_metrics, ci_results


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  §6b  PARTICIPANT-LEVEL EVALUATION                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def participant_level_metrics(clin, positions, predictions, y_vals, tag):
    """Aggregate to one prediction per participant, return metrics."""
    pids = np.array([map_participant_id(str(x))
                     for x in clin.index[positions]])
    vcs = clin["visit_code"].iloc[positions].values

    df = pd.DataFrame({"pid": pids, "vc": vcs, "pred": predictions, "y": y_vals})
    df = df.dropna(subset=["pred", "y"])
    results = {}

    # Track 1 (PRIMARY): Participant-mean
    agg = df.groupby("pid").agg(pred=("pred", "mean"), y=("y", "mean")).reset_index()
    if len(agg) >= 5:
        mets_agg = full_metrics(agg["pred"].values, agg["y"].values)
        results[f"{tag}_participant_mean"] = mets_agg
        print(f"  * {tag} Participant-mean (PRIMARY): "
              f"n={mets_agg['n']}, rho_s={mets_agg['spearman']:.3f}, "
              f"MAE={mets_agg['mae']:.2f}, R2={mets_agg['r2']:.3f}")

    # Track 2: Baseline-only (M0)
    m0_df = df[df["vc"].astype(str) == "M0"]
    if len(m0_df) >= 5:
        m0_agg = m0_df.groupby("pid").agg(
            pred=("pred", "mean"), y=("y", "mean")).reset_index()
        mets_m0 = full_metrics(m0_agg["pred"].values, m0_agg["y"].values)
        results[f"{tag}_baseline_M0"] = mets_m0
        print(f"    {tag} Baseline (M0): n={mets_m0['n']}, "
              f"rho_s={mets_m0['spearman']:.3f}")

    # Track 3 (secondary): Row-level
    mets_row = full_metrics(predictions, y_vals)
    results[f"{tag}_row_level"] = mets_row
    print(f"    {tag} Row-level (secondary): n={mets_row['n']}, "
          f"rho_s={mets_row['spearman']:.3f}")

    return results


def participant_level_bootstrap_ci(clin, positions, predictions, y_vals,
                                    n_boot=2000, seed=SEED):
    """Bootstrap 95% CI on participant-mean metrics."""
    pids = np.array([map_participant_id(str(x))
                     for x in clin.index[positions]])
    df = pd.DataFrame({"pid": pids, "pred": predictions, "y": y_vals})
    df = df.dropna(subset=["pred", "y"])
    agg = df.groupby("pid").agg(pred=("pred", "mean"), y=("y", "mean")).reset_index()

    if len(agg) < 10:
        return {}

    rng = np.random.default_rng(seed)
    boot_metrics = {"spearman": [], "pearson": [], "mae": [], "rmse": [], "r2": []}

    for _ in range(n_boot):
        bs_idx = rng.choice(len(agg), size=len(agg), replace=True)
        p_ = agg["pred"].values[bs_idx]
        t_ = agg["y"].values[bs_idx]
        m_ = np.isfinite(p_) & np.isfinite(t_)
        if m_.sum() < 5:
            continue
        p_b, t_b = p_[m_], t_[m_]
        boot_metrics["spearman"].append(float(spearmanr(p_b, t_b).statistic))
        boot_metrics["pearson"].append(float(pearsonr(p_b, t_b).statistic))
        boot_metrics["mae"].append(float(np.mean(np.abs(p_b - t_b))))
        boot_metrics["rmse"].append(float(np.sqrt(np.mean((p_b - t_b)**2))))
        ss_res = np.sum((t_b - p_b)**2)
        ss_tot = np.sum((t_b - t_b.mean())**2)
        boot_metrics["r2"].append(float(1 - ss_res / max(ss_tot, 1e-12)))

    point = full_metrics(agg["pred"].values, agg["y"].values)
    result = {}
    for metric in boot_metrics:
        vals = np.array(boot_metrics[metric])
        if len(vals) < 100:
            result[metric] = {"point": point.get(metric, np.nan),
                              "lo": np.nan, "hi": np.nan}
        else:
            result[metric] = {
                "point": point.get(metric, np.nan),
                "lo": float(np.percentile(vals, 2.5)),
                "hi": float(np.percentile(vals, 97.5)),
            }
    return result


def run_participant_level_eval(clin, y_all, train_idx_y, oof_pred,
                                test_idx_omics, test_pred, prot_ok_test,
                                y_te_full, has_test_y):
    """Run participant-level evaluation for OOF and TEST."""
    print(f"\n{'=' * 60}")
    print("PRIMARY: PARTICIPANT-LEVEL EVALUATION")
    print(f"{'=' * 60}")

    plr_results = {}
    plr_ci = {}

    if len(train_idx_y) >= 20 and oof_pred.size > 0:
        plr_oof = participant_level_metrics(
            clin, train_idx_y, oof_pred, y_all[train_idx_y], "OOF")
        plr_results.update(plr_oof)
        ci = participant_level_bootstrap_ci(
            clin, train_idx_y, oof_pred, y_all[train_idx_y])
        plr_ci["OOF_participant_mean"] = ci
        if ci:
            print(f"  OOF participant-mean 95% CI:")
            for m in ("spearman", "mae"):
                d = ci.get(m, {})
                print(f"    {m}: {d.get('point',0):.3f} "
                      f"[{d.get('lo',0):.3f}, {d.get('hi',0):.3f}]")

    if test_idx_omics.size > 0 and has_test_y.sum() >= 5:
        te_pred = test_pred[prot_ok_test]
        plr_test = participant_level_metrics(
            clin, test_idx_omics, te_pred, y_te_full, "TEST")
        plr_results.update(plr_test)
        ci_te = participant_level_bootstrap_ci(
            clin, test_idx_omics, te_pred, y_te_full)
        plr_ci["TEST_participant_mean"] = ci_te
        if ci_te:
            print(f"  TEST participant-mean 95% CI:")
            for m in ("spearman", "mae"):
                d = ci_te.get(m, {})
                print(f"    {m}: {d.get('point',0):.3f} "
                      f"[{d.get('lo',0):.3f}, {d.get('hi',0):.3f}]")

    pd.DataFrame(plr_results).T.to_csv(TAB / "participant_level_metrics.csv")

    plr_ci_rows = []
    for split, ci_dict in plr_ci.items():
        for metric, vals in ci_dict.items():
            plr_ci_rows.append({"split": split, "metric": metric, **vals})
    if plr_ci_rows:
        pd.DataFrame(plr_ci_rows).to_csv(
            TAB / "participant_level_bootstrap_ci.csv", index=False)

    summary_update({"participant_level": plr_results,
                    "participant_ci": plr_ci})
    return plr_results, plr_ci

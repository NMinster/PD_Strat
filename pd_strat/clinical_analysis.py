"""
§6c — Severity band classification, decile analysis, and quintile separation.
"""

from __future__ import annotations

import traceback
from typing import Dict, Any

import numpy as np
import pandas as pd

from .config import TAB, Y_LO, Y_HI
from .utils import map_participant_id, full_metrics, summary_update

SEVERITY_BANDS = {
    "mild_moderate": 32,
    "moderate_severe": 58,
}


def _severity_band_analysis(positions, predictions, y_vals, tag):
    """Compute severity band classification metrics."""
    from sklearn.metrics import roc_auc_score, average_precision_score

    m = np.isfinite(predictions) & np.isfinite(y_vals)
    if m.sum() < 20:
        print(f"  {tag}: insufficient data ({m.sum()} valid)")
        return {}

    pred, true = predictions[m], y_vals[m]
    results = {}

    for band_name, threshold in SEVERITY_BANDS.items():
        y_binary = (true > threshold).astype(int)
        n_pos, n_neg = y_binary.sum(), len(y_binary) - y_binary.sum()
        if n_pos < 5 or n_neg < 5:
            continue
        try:
            auroc = roc_auc_score(y_binary, pred)
            auprc = average_precision_score(y_binary, pred)
            prevalence = n_pos / len(y_binary)
            results[f"{band_name}_auroc"] = auroc
            results[f"{band_name}_auprc"] = auprc
            results[f"{band_name}_prevalence"] = prevalence
            results[f"{band_name}_n_pos"] = n_pos
            results[f"{band_name}_n_neg"] = n_neg
            print(f"  {tag} {band_name} (>{threshold}): "
                  f"AUROC={auroc:.3f}, AUPRC={auprc:.3f} "
                  f"(prev={prevalence:.2f})")
        except Exception as e:
            print(f"  {tag} {band_name}: failed ({e})")

    # 3-class classification
    band_true = np.where(true <= SEVERITY_BANDS["mild_moderate"], 0,
                np.where(true <= SEVERITY_BANDS["moderate_severe"], 1, 2))
    band_pred = np.where(pred <= SEVERITY_BANDS["mild_moderate"], 0,
                np.where(pred <= SEVERITY_BANDS["moderate_severe"], 1, 2))
    try:
        from sklearn.metrics import cohen_kappa_score, accuracy_score
        kappa = cohen_kappa_score(band_true, band_pred, weights="linear")
        acc = accuracy_score(band_true, band_pred)
        results["3class_kappa_linear"] = kappa
        results["3class_accuracy"] = acc
        print(f"  {tag} 3-class: kappa_linear={kappa:.3f}, accuracy={acc:.2f}")
        for bname, bval in [("mild", 0), ("moderate", 1), ("severe", 2)]:
            in_band = band_true == bval
            if in_band.sum() > 0:
                results[f"recall_{bname}"] = float((band_pred[in_band] == bval).mean())
    except Exception:
        pass

    return results


def _decile_analysis(positions, predictions, y_vals, tag):
    """Bin predictions into deciles + top-vs-bottom quintile separation."""
    m = np.isfinite(predictions) & np.isfinite(y_vals)
    if m.sum() < 20:
        return None, {}

    pred, true = predictions[m], y_vals[m]
    results = {}

    n_bins = min(10, len(pred) // 5)
    if n_bins < 3:
        return None, results

    pctiles = np.percentile(pred, np.linspace(0, 100, n_bins + 1))
    rows = []
    for i in range(n_bins):
        lo_, hi_ = pctiles[i], pctiles[i + 1]
        mask = (pred >= lo_) & (pred <= hi_) if i == n_bins - 1 else (pred >= lo_) & (pred < hi_)
        if mask.sum() > 0:
            rows.append({
                "decile": i + 1,
                "pred_mean": float(pred[mask].mean()),
                "pred_lo": float(lo_), "pred_hi": float(hi_),
                "obs_mean": float(true[mask].mean()),
                "obs_std": float(true[mask].std()),
                "obs_median": float(np.median(true[mask])),
                "n": int(mask.sum()),
            })
    df_decile = pd.DataFrame(rows)

    q20 = np.percentile(pred, 20)
    q80 = np.percentile(pred, 80)
    bottom_q = true[pred <= q20]
    top_q = true[pred >= q80]
    if len(bottom_q) >= 3 and len(top_q) >= 3:
        mean_diff = top_q.mean() - bottom_q.mean()
        pooled_sd = np.sqrt(
            (bottom_q.var() * (len(bottom_q) - 1) +
             top_q.var() * (len(top_q) - 1)) /
            (len(bottom_q) + len(top_q) - 2))
        cohens_d = mean_diff / max(pooled_sd, 1e-6)
        results["quintile_top_mean"] = float(top_q.mean())
        results["quintile_bottom_mean"] = float(bottom_q.mean())
        results["quintile_mean_diff"] = float(mean_diff)
        results["quintile_cohens_d"] = float(cohens_d)
        results["quintile_top_n"] = len(top_q)
        results["quintile_bottom_n"] = len(bottom_q)
        print(f"  {tag} Q5 vs Q1: diff={mean_diff:.1f}, d={cohens_d:.2f}")

    return df_decile, results


def run_severity_band_analysis(clin, y_all, train_idx_y, oof_pred,
                                test_idx_omics, test_pred, prot_ok_test,
                                y_te_full, has_test_y):
    """Run severity band classification + decile analyses."""
    print(f"\n{'=' * 60}")
    print("SEVERITY BAND CLASSIFICATION + CLINICAL DECISION ANALYSES")
    print(f"{'=' * 60}")

    severity_band_results: Dict[str, Any] = {}
    try:
        # OOF
        if len(train_idx_y) >= 20 and oof_pred.size > 0:
            sb_oof = _severity_band_analysis(
                train_idx_y, oof_pred, y_all[train_idx_y], "OOF")
            severity_band_results["OOF"] = sb_oof
            df_dec, dec = _decile_analysis(
                train_idx_y, oof_pred, y_all[train_idx_y], "OOF")
            severity_band_results["OOF_decile"] = dec
            if df_dec is not None:
                df_dec.to_csv(TAB / "decile_analysis_oof.csv", index=False)

        # TEST
        if test_idx_omics.size > 0 and has_test_y.sum() >= 5:
            te_pred = test_pred[prot_ok_test]
            sb_test = _severity_band_analysis(
                test_idx_omics, te_pred, y_te_full, "TEST")
            severity_band_results["TEST"] = sb_test
            df_dec_t, dec_t = _decile_analysis(
                test_idx_omics, te_pred, y_te_full, "TEST")
            severity_band_results["TEST_decile"] = dec_t
            if df_dec_t is not None:
                df_dec_t.to_csv(TAB / "decile_analysis_test.csv", index=False)

        # Participant-level
        print(f"\n  --- Participant-level severity bands ---")
        for p_tag, p_pos, p_preds, p_y in [
            ("OOF", train_idx_y, oof_pred, y_all[train_idx_y]),
            ("TEST", test_idx_omics,
             test_pred[prot_ok_test] if test_idx_omics.size > 0 else np.array([]),
             y_te_full),
        ]:
            if len(p_pos) < 20:
                continue
            pids = np.array([map_participant_id(str(x))
                             for x in clin.index[p_pos]])
            df_sb = pd.DataFrame({"pid": pids, "pred": p_preds, "y": p_y})
            df_sb = df_sb.dropna(subset=["pred", "y"])
            agg = df_sb.groupby("pid").agg(
                pred=("pred", "mean"), y=("y", "mean")).reset_index()
            if len(agg) >= 20:
                sb_plr = _severity_band_analysis(
                    np.arange(len(agg)), agg["pred"].values,
                    agg["y"].values, f"{p_tag}_participant")
                severity_band_results[f"{p_tag}_participant"] = sb_plr
                _, dec_plr = _decile_analysis(
                    np.arange(len(agg)), agg["pred"].values,
                    agg["y"].values, f"{p_tag}_participant")
                severity_band_results[f"{p_tag}_participant_decile"] = dec_plr

        # Save
        sb_flat = {}
        for k, v in severity_band_results.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    sb_flat[f"{k}_{k2}"] = v2
        pd.DataFrame([sb_flat]).T.to_csv(TAB / "severity_bands.csv")
        summary_update({"severity_bands": severity_band_results})

    except Exception as e:
        print(f"  [WARN] Severity band analysis failed: {e}")
        traceback.print_exc()

    return severity_band_results

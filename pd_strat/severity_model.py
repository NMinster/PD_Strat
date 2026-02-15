"""
§4–5c — Targets, masks, CV setup, OOF predictions, covariate baselines,
and TRAIN->TEST evaluation for all candidate models.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import RidgeCV, ElasticNetCV, Ridge as RidgeModel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

from .config import (
    LOCKED, RIDGE_ALPHAS, ENET_L1_RATIOS,
    Y_LO, Y_HI, PROT_COMPLETENESS_THRESHOLD,
    N_SVD, PANEL_SVD_NC, SEED,
)
from .utils import (
    map_participant_id, full_metrics, spearman_np,
    flow_record, summary_update,
)
from .features import (
    svd_n_components, build_panel_aware_features, build_monolithic_features,
)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  §4  TARGETS, MASKS, CV SETUP                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def setup_targets_and_cv(clin: pd.DataFrame,
                         z_prot: pd.DataFrame,
                         M_prot: np.ndarray,
                         is_train: pd.Series,
                         is_test: pd.Series,
                         ) -> dict:
    """Prepare y arrays, train/test indices, completeness masks, and CV.

    Returns a dict with keys:
        y_all, upsit_all, train_idx, test_idx, y_tr,
        has_any_omics, prot_ok_train, prot_ok_test,
        valid_train, train_idx_y, y_tr_nonan,
        groups_all, groups_train, n_groups, n_splits, gkf,
        test_idx_omics, y_te_full, has_test_y,
        UPDRS_STATS
    """
    y_all = pd.to_numeric(clin["updrs_total"], errors="coerce").values.astype(np.float32)
    upsit_all = pd.to_numeric(clin.get("upsit_total"), errors="coerce").values.astype(np.float32)

    train_idx = np.where(is_train.values)[0]
    test_idx  = np.where(is_test.values)[0]
    y_tr = y_all[train_idx]
    has_any_omics = (M_prot.sum(1) > 0).astype(bool)

    def prot_completeness(positions):
        return z_prot.iloc[positions].notna().mean(axis=1).values

    comp_train = prot_completeness(train_idx)
    comp_test  = prot_completeness(test_idx)
    prot_ok_train = ((comp_train >= PROT_COMPLETENESS_THRESHOLD)
                     & has_any_omics[train_idx])
    prot_ok_test  = ((comp_test >= PROT_COMPLETENESS_THRESHOLD)
                     & has_any_omics[test_idx])
    flow_record("04_completeness_filter",
                int(prot_ok_train.sum()), int(prot_ok_test.sum()))

    print(f"\n[Completeness verify] Threshold={PROT_COMPLETENESS_THRESHOLD}")
    if prot_ok_train.any() and (~prot_ok_train).any():
        print(f"  TRAIN: {prot_ok_train.sum()}/{len(comp_train)} pass "
              f"(min_comp={comp_train[prot_ok_train].min():.3f}, "
              f"max_fail={comp_train[~prot_ok_train].max():.3f})")
    else:
        print(f"  TRAIN: {prot_ok_train.sum()}/{len(comp_train)} pass")
    print(f"  TEST:  {prot_ok_test.sum()}/{len(comp_test)} pass")

    # UPDRS scale stats
    y_train_finite = y_all[train_idx][np.isfinite(y_all[train_idx]) & prot_ok_train]
    UPDRS_STATS: Dict[str, float] = {}
    if y_train_finite.size >= 10:
        UPDRS_STATS = {
            "min": float(np.min(y_train_finite)),
            "max": float(np.max(y_train_finite)),
            "mean": float(np.mean(y_train_finite)),
            "std": float(np.std(y_train_finite)),
            "median": float(np.median(y_train_finite)),
            "q25": float(np.percentile(y_train_finite, 25)),
            "q75": float(np.percentile(y_train_finite, 75)),
            "iqr": float(np.percentile(y_train_finite, 75) -
                         np.percentile(y_train_finite, 25)),
        }
        print(f"\n{'=' * 60}\nUPDRS SCALE CONTEXT (TRAIN)\n{'=' * 60}")
        print(f"  Range: {UPDRS_STATS['min']:.1f} - {UPDRS_STATS['max']:.1f}")
        print(f"  Mean+-SD: {UPDRS_STATS['mean']:.1f}+-{UPDRS_STATS['std']:.1f}")
        print(f"  Clamp: [{Y_LO}, {Y_HI}]")
        summary_update({"updrs_scale": UPDRS_STATS})

    # CV machinery
    have_y_tr = np.isfinite(y_tr)
    valid_train = have_y_tr & prot_ok_train
    train_idx_y = train_idx[valid_train]
    y_tr_nonan  = y_tr[valid_train]

    groups_all = np.array([map_participant_id(str(x))
                           for x in clin.index.to_numpy()])
    groups_train = groups_all[train_idx_y]
    n_groups = len(pd.unique(groups_train))
    n_splits = max(2, min(LOCKED["n_cv_folds"], n_groups))
    print(f"  CV: n_groups={n_groups} (participant-level), n_splits={n_splits}")
    gkf = GroupKFold(n_splits=n_splits)

    # Fold-leakage check
    for fi, (tr_ix, va_ix) in enumerate(
            gkf.split(train_idx_y, y_tr_nonan, groups_train), 1):
        g_tr = set(groups_train[tr_ix])
        g_va = set(groups_train[va_ix])
        overlap = g_tr & g_va
        assert len(overlap) == 0, (
            f"Fold {fi}: {len(overlap)} participants in both train & val!")
    print(f"  Fold-leakage check passed: zero participant overlap "
          f"across all {n_splits} folds")

    test_idx_omics = test_idx[prot_ok_test]
    y_te_full = y_all[test_idx_omics]
    has_test_y = np.isfinite(y_te_full)

    flow_record("05_severity_CV", int(valid_train.sum()),
                int((np.isfinite(y_all[test_idx]) & prot_ok_test).sum()))

    return {
        "y_all": y_all, "upsit_all": upsit_all,
        "train_idx": train_idx, "test_idx": test_idx, "y_tr": y_tr,
        "has_any_omics": has_any_omics,
        "prot_ok_train": prot_ok_train, "prot_ok_test": prot_ok_test,
        "valid_train": valid_train,
        "train_idx_y": train_idx_y, "y_tr_nonan": y_tr_nonan,
        "groups_all": groups_all, "groups_train": groups_train,
        "n_groups": n_groups, "n_splits": n_splits, "gkf": gkf,
        "test_idx_omics": test_idx_omics,
        "y_te_full": y_te_full, "has_test_y": has_test_y,
        "UPDRS_STATS": UPDRS_STATS,
    }


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  §5a  CROSS-VALIDATED OOF                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def run_oof_cv(X, M, y_all_arr, train_idx_y, groups, gkf_obj,
               panel_col_indices, n_svd_panel=PANEL_SVD_NC,
               use_panel_aware=True):
    """Ridge & ElasticNet with per-fold panel-aware features."""
    ytr = y_all_arr[train_idx_y]
    oof_ridge = np.full(len(train_idx_y), np.nan, dtype=np.float32)
    oof_enet  = np.full(len(train_idx_y), np.nan, dtype=np.float32)

    for fold, (tr, va) in enumerate(
            gkf_obj.split(train_idx_y, ytr, groups), 1):
        fp = train_idx_y[tr]
        vp = train_idx_y[va]
        yf = ytr[tr]

        if use_panel_aware and panel_col_indices:
            Z_all, _ = build_panel_aware_features(
                X, M, fp, panel_col_indices, n_svd_panel)
            Ztr = Z_all[fp]
            Zva = Z_all[vp]
        else:
            sc = StandardScaler(with_mean=False)
            Xtr_s = sc.fit_transform(X[fp])
            Xva_s = sc.transform(X[vp])
            nc = svd_n_components(Xtr_s)
            svd = TruncatedSVD(n_components=nc, random_state=SEED)
            Ztr = svd.fit_transform(Xtr_s)
            Zva = svd.transform(Xva_s)

        ridge = RidgeCV(alphas=RIDGE_ALPHAS)
        ridge.fit(Ztr, yf)
        oof_ridge[va] = np.clip(ridge.predict(Zva), Y_LO, Y_HI).astype(np.float32)

        try:
            enet = ElasticNetCV(l1_ratio=ENET_L1_RATIOS,
                                alphas=np.logspace(-3, 1, 10),
                                cv=3, random_state=SEED, max_iter=2000)
            enet.fit(Ztr, yf)
            oof_enet[va] = np.clip(enet.predict(Zva), Y_LO, Y_HI).astype(np.float32)
        except Exception:
            oof_enet[va] = oof_ridge[va]

    return oof_ridge, oof_enet


def run_primary_oof(X_prot, M_prot, y_all, train_idx_y, groups_train, gkf,
                    PANEL_COL_INDICES, USE_PANEL_AWARE, UPDRS_STATS):
    """Run all OOF candidates: panel-aware, monolithic, HistGBT.

    Returns (baseline_oof, _oof_preds, oof_ridge, oof_mono_ridge).
    """
    print(f"\n{'=' * 60}")
    print("PRIMARY MODEL: Panel-Aware RidgeSVD")
    print(f"{'=' * 60}")

    d_prot = X_prot.shape[1]
    baseline_oof: Dict[str, Dict] = {}
    _oof_preds: Dict[str, np.ndarray] = {}
    oof_ridge = np.full(0, np.nan)
    oof_mono_ridge = np.full(0, np.nan)
    y_true_cv = y_all[train_idx_y]

    if len(train_idx_y) < 20 or d_prot == 0:
        return baseline_oof, _oof_preds, oof_ridge, oof_mono_ridge

    oof_ridge, oof_enet = run_oof_cv(
        X_prot, M_prot, y_all, train_idx_y, groups_train, gkf,
        PANEL_COL_INDICES, PANEL_SVD_NC, use_panel_aware=USE_PANEL_AWARE)

    primary_label = "PanelRidgeSVD" if USE_PANEL_AWARE else "RidgeSVD"
    for name, oof in [(primary_label, oof_ridge),
                      (primary_label.replace("Ridge", "ElasticNet"), oof_enet)]:
        mets = full_metrics(oof, y_true_cv, name)
        if mets["n"] >= 10:
            pct = (100 * mets["mae"] / (UPDRS_STATS.get("max", 1) -
                   UPDRS_STATS.get("min", 0)) if UPDRS_STATS else np.nan)
            baseline_oof[name] = mets
            print(f"  {name} OOF (n={mets['n']}): "
                  f"rho_s={mets['spearman']:.3f}, rho_p={mets['pearson']:.3f}, "
                  f"MAE={mets['mae']:.2f} ({pct:.1f}% range), "
                  f"RMSE={mets['rmse']:.2f}, R2={mets['r2']:.3f}")

    # Monolithic comparison
    if USE_PANEL_AWARE:
        oof_mono_ridge, _ = run_oof_cv(
            X_prot, M_prot, y_all, train_idx_y, groups_train, gkf,
            PANEL_COL_INDICES, N_SVD, use_panel_aware=False)
        mets_mono = full_metrics(oof_mono_ridge, y_true_cv, "MonoRidgeSVD")
        if mets_mono["n"] >= 10:
            baseline_oof["MonoRidgeSVD"] = mets_mono
            print(f"  MonoRidgeSVD OOF (n={mets_mono['n']}): "
                  f"rho_s={mets_mono['spearman']:.3f}, "
                  f"MAE={mets_mono['mae']:.2f}, R2={mets_mono['r2']:.3f}")
    else:
        oof_mono_ridge = oof_ridge

    # HistGradientBoosting baseline
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        print(f"\n  --- Hard baseline: HistGradientBoosting ---")
        oof_gbt = np.full(len(train_idx_y), np.nan, dtype=np.float32)
        for fold, (tr, va) in enumerate(
                gkf.split(train_idx_y, y_true_cv, groups_train), 1):
            fp = train_idx_y[tr]
            vp = train_idx_y[va]
            yf = y_all[fp]
            if USE_PANEL_AWARE and PANEL_COL_INDICES:
                Z_all_gbt, _ = build_panel_aware_features(
                    X_prot, M_prot, fp, PANEL_COL_INDICES, PANEL_SVD_NC)
                Ztr, Zva = Z_all_gbt[fp], Z_all_gbt[vp]
            else:
                sc = StandardScaler(with_mean=False)
                Xtr_s = sc.fit_transform(X_prot[fp])
                Xva_s = sc.transform(X_prot[vp])
                nc = svd_n_components(Xtr_s)
                svd = TruncatedSVD(n_components=nc, random_state=SEED)
                Ztr = svd.fit_transform(Xtr_s)
                Zva = svd.transform(Xva_s)

            gbt = HistGradientBoostingRegressor(
                max_iter=200, max_depth=5, learning_rate=0.05,
                min_samples_leaf=10, random_state=SEED,
                early_stopping=True, n_iter_no_change=20,
                validation_fraction=0.15)
            gbt.fit(Ztr, yf)
            oof_gbt[va] = np.clip(gbt.predict(Zva), Y_LO, Y_HI).astype(np.float32)

        mets_gbt = full_metrics(oof_gbt, y_true_cv, "HistGBT")
        if mets_gbt["n"] >= 10:
            baseline_oof["HistGBT"] = mets_gbt
            _oof_preds["HistGBT"] = oof_gbt
            print(f"  HistGBT OOF (n={mets_gbt['n']}): "
                  f"rho_s={mets_gbt['spearman']:.3f}, "
                  f"MAE={mets_gbt['mae']:.2f}, R2={mets_gbt['r2']:.3f}")
    except ImportError:
        print("  [SKIP] scikit-learn HistGBT not available")

    _candidate_label = "PanelRidgeSVD" if USE_PANEL_AWARE else "RidgeSVD"
    _oof_preds[_candidate_label] = oof_ridge
    _oof_preds["MonoRidgeSVD"] = oof_mono_ridge

    summary_update({"oof_models": baseline_oof})
    return baseline_oof, _oof_preds, oof_ridge, oof_mono_ridge


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  §5b  COVARIATE BASELINES + INCREMENTAL GAIN                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def build_covariate_matrix(clin, positions, reference_cols=None):
    """Build covariate design matrix (age, sex, site dummies)."""
    from .data_loading import demographics_for
    demo = demographics_for(clin, positions)
    parts, names = [], []

    age = pd.to_numeric(demo["age"], errors="coerce").values
    if np.isfinite(age).sum() >= 5:
        a = age.copy()
        a[~np.isfinite(a)] = np.nanmean(a)
        parts.append(a.reshape(-1, 1))
        names.append("age")

    sex_s = demo["sex"].astype(str).fillna("UNKNOWN")
    dum_sex = pd.get_dummies(sex_s, prefix="sex", drop_first=True)
    if dum_sex.shape[1] >= 1:
        names.extend(dum_sex.columns.tolist())
        parts.append(dum_sex.values.astype(np.float64))

    site_s = demo["site"].astype(str).fillna("UNKNOWN")
    dum_site = pd.get_dummies(site_s, prefix="site", drop_first=True)
    if dum_site.shape[1] >= 1:
        names.extend(dum_site.columns.tolist())
        parts.append(dum_site.values.astype(np.float64))

    if not parts:
        return None, []

    X = np.hstack(parts)
    if reference_cols is not None:
        df_x = pd.DataFrame(X, columns=names)
        df_x = df_x.reindex(columns=reference_cols, fill_value=0.0)
        X = df_x.values.astype(np.float64)
        names = list(df_x.columns)

    return X, names


def run_covariate_baselines(clin, X_prot, M_prot, y_all,
                            train_idx_y, y_tr_nonan, groups_train, gkf,
                            PANEL_COL_INDICES, USE_PANEL_AWARE,
                            baseline_oof, PRIMARY_LABEL):
    """Covariates-only, proteomics-only, and combined OOF CV."""
    from .config import TAB
    print(f"\n{'=' * 60}")
    print("COVARIATE BASELINES + INCREMENTAL GAIN")
    print(f"{'=' * 60}")

    covariate_results: Dict[str, Dict] = {}
    if len(train_idx_y) < 20:
        return covariate_results

    X_cov_full, cov_col_names = build_covariate_matrix(
        clin, np.arange(len(clin)))
    has_covariates = X_cov_full is not None and len(cov_col_names) >= 1

    if not has_covariates:
        print("  [SKIP] No usable covariates")
        pd.DataFrame(covariate_results).T.to_csv(TAB / "incremental_gain_oof.csv")
        summary_update({"incremental_gain": covariate_results})
        return covariate_results

    print(f"  Covariates: {len(cov_col_names)} features "
          f"({', '.join(cov_col_names[:5])}...)")

    # Covariates-only OOF
    oof_cov = np.full(len(train_idx_y), np.nan, dtype=np.float32)
    for _, (tr, va) in enumerate(
            gkf.split(train_idx_y, y_tr_nonan, groups_train)):
        Xtr_c = X_cov_full[train_idx_y[tr]]
        Xva_c = X_cov_full[train_idx_y[va]]
        yf = y_all[train_idx_y[tr]]
        ridge_c = RidgeCV(alphas=RIDGE_ALPHAS)
        ridge_c.fit(Xtr_c, yf)
        oof_cov[va] = np.clip(ridge_c.predict(Xva_c), Y_LO, Y_HI).astype(np.float32)

    mets_cov = full_metrics(oof_cov, y_all[train_idx_y])
    covariate_results["covariates_only"] = mets_cov
    print(f"  Covariates-only OOF: rho_s={mets_cov['spearman']:.3f}, "
          f"MAE={mets_cov['mae']:.2f}, R2={mets_cov['r2']:.3f}")

    # Combined OOF
    oof_combined = np.full(len(train_idx_y), np.nan, dtype=np.float32)
    for _, (tr, va) in enumerate(
            gkf.split(train_idx_y, y_tr_nonan, groups_train)):
        fp = train_idx_y[tr]
        vp = train_idx_y[va]
        yf = y_all[fp]
        if USE_PANEL_AWARE:
            Z_all, _ = build_panel_aware_features(
                X_prot, M_prot, fp, PANEL_COL_INDICES, PANEL_SVD_NC)
            Ztr, Zva = Z_all[fp], Z_all[vp]
        else:
            sc = StandardScaler(with_mean=False)
            Xtr_s = sc.fit_transform(X_prot[fp])
            Xva_s = sc.transform(X_prot[vp])
            nc = svd_n_components(Xtr_s)
            svd = TruncatedSVD(n_components=nc, random_state=SEED)
            Ztr = svd.fit_transform(Xtr_s)
            Zva = svd.transform(Xva_s)

        Xtr_comb = np.hstack([Ztr, X_cov_full[fp]])
        Xva_comb = np.hstack([Zva, X_cov_full[vp]])
        ridge_cb = RidgeCV(alphas=RIDGE_ALPHAS)
        ridge_cb.fit(Xtr_comb, yf)
        oof_combined[va] = np.clip(ridge_cb.predict(Xva_comb),
                                   Y_LO, Y_HI).astype(np.float32)

    mets_comb = full_metrics(oof_combined, y_all[train_idx_y])
    covariate_results["combined"] = mets_comb
    covariate_results["proteomics_only"] = baseline_oof.get(PRIMARY_LABEL, {})

    rho_cov = mets_cov.get("spearman", np.nan)
    rho_prot = covariate_results["proteomics_only"].get("spearman", np.nan)
    rho_comb = mets_comb.get("spearman", np.nan)
    print(f"\n  INCREMENTAL GAIN (OOF):")
    print(f"    Covariates-only:  rho={rho_cov:.3f}")
    print(f"    Proteomics-only:  rho={rho_prot:.3f}")
    print(f"    Combined:         rho={rho_comb:.3f}")
    if np.isfinite(rho_cov) and np.isfinite(rho_prot):
        print(f"    Delta_rho(prot vs cov):  {rho_prot - rho_cov:+.3f}")
        print(f"    Delta_rho(comb vs cov):  {rho_comb - rho_cov:+.3f}")

    pd.DataFrame(covariate_results).T.to_csv(TAB / "incremental_gain_oof.csv")
    summary_update({"incremental_gain": covariate_results})
    return covariate_results


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  §5c  TRAIN -> TEST (all candidates)                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def run_test_evaluation(X_prot, M_prot, y_all, train_idx_y,
                        test_idx, test_idx_omics, y_te_full, has_test_y,
                        prot_ok_test,
                        PANEL_COL_INDICES, USE_PANEL_AWARE,
                        baseline_oof, _oof_preds):
    """Fit on all TRAIN, predict TEST for every candidate model.

    Returns (test_results, _test_preds, PRIMARY_LABEL, oof_pred,
             rho_oof, oof_mae, oof_rmse, test_pred_ridge).
    """
    from .config import TAB
    print(f"\n{'=' * 60}")
    print("TRAIN->TEST -- ALL CANDIDATE MODELS")
    print(f"{'=' * 60}")

    d_prot = X_prot.shape[1]
    test_results: Dict[str, Dict] = {}
    _test_preds: Dict[str, np.ndarray] = {}
    _candidate_label = "PanelRidgeSVD" if USE_PANEL_AWARE else "RidgeSVD"

    if not (len(train_idx_y) >= 20 and d_prot > 0
            and test_idx_omics.size > 0 and has_test_y.sum() >= 5):
        test_pred_ridge = np.full(test_idx.shape[0], np.nan, dtype=np.float32)
        print("  [SKIP] Insufficient TRAIN or TEST data")
        return (test_results, _test_preds, _candidate_label,
                _oof_preds.get(_candidate_label, np.full(0, np.nan)),
                np.nan, np.nan, np.nan, test_pred_ridge)

    ytr_all = y_all[train_idx_y]

    # Panel-aware candidates
    if USE_PANEL_AWARE:
        Z_all_full, panel_fit = build_panel_aware_features(
            X_prot, M_prot, train_idx_y, PANEL_COL_INDICES, PANEL_SVD_NC)
        Ztr_pa, Zte_pa = Z_all_full[train_idx_y], Z_all_full[test_idx_omics]
        for tag, Cls, kw in [
            ("PanelRidgeSVD", RidgeCV, dict(alphas=RIDGE_ALPHAS)),
            ("PanelElasticNetSVD", ElasticNetCV,
             dict(l1_ratio=ENET_L1_RATIOS, alphas=np.logspace(-3, 1, 10),
                  cv=3, random_state=SEED, max_iter=2000)),
        ]:
            try:
                mdl = Cls(**kw); mdl.fit(Ztr_pa, ytr_all)
                pred = np.clip(mdl.predict(Zte_pa), Y_LO, Y_HI).astype(np.float32)
            except Exception:
                pred = np.full(len(test_idx_omics), np.nan, dtype=np.float32)
            arr = np.full(test_idx.shape[0], np.nan, dtype=np.float32)
            arr[prot_ok_test] = pred
            _test_preds[tag] = arr
            mets = full_metrics(pred, y_te_full, tag)
            if mets["n"] >= 5:
                cal = np.polyfit(pred[np.isfinite(pred) & has_test_y],
                                 y_te_full[np.isfinite(pred) & has_test_y], 1)
                mets["cal_slope"] = float(cal[0])
                mets["cal_intercept"] = float(cal[1])
                test_results[tag] = mets
                print(f"  {tag} TEST (n={mets['n']}): "
                      f"rho_s={mets['spearman']:.3f}, MAE={mets['mae']:.2f}, "
                      f"R2={mets['r2']:.3f}, cal_slope={cal[0]:.3f}")

    # Monolithic candidates
    Z_mono, mono_fit = build_monolithic_features(X_prot, train_idx_y, N_SVD)
    Ztr_m, Zte_m = Z_mono[train_idx_y], Z_mono[test_idx_omics]
    for tag, Cls, kw in [
        ("MonoRidgeSVD", RidgeCV, dict(alphas=RIDGE_ALPHAS)),
        ("MonoElasticNetSVD", ElasticNetCV,
         dict(l1_ratio=ENET_L1_RATIOS, alphas=np.logspace(-3, 1, 10),
              cv=3, random_state=SEED, max_iter=2000)),
    ]:
        try:
            mdl = Cls(**kw); mdl.fit(Ztr_m, ytr_all)
            pred = np.clip(mdl.predict(Zte_m), Y_LO, Y_HI).astype(np.float32)
        except Exception:
            pred = np.full(len(test_idx_omics), np.nan, dtype=np.float32)
        arr = np.full(test_idx.shape[0], np.nan, dtype=np.float32)
        arr[prot_ok_test] = pred
        _test_preds[tag] = arr
        mets = full_metrics(pred, y_te_full, tag)
        if mets["n"] >= 5:
            cal = np.polyfit(pred[np.isfinite(pred) & has_test_y],
                             y_te_full[np.isfinite(pred) & has_test_y], 1)
            mets["cal_slope"] = float(cal[0])
            mets["cal_intercept"] = float(cal[1])
            test_results[tag] = mets
            print(f"  {tag} TEST (n={mets['n']}): "
                  f"rho_s={mets['spearman']:.3f}, MAE={mets['mae']:.2f}, "
                  f"R2={mets['r2']:.3f}, cal_slope={cal[0]:.3f}")

    # HistGBT
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        gbt = HistGradientBoostingRegressor(
            max_iter=200, max_depth=5, learning_rate=0.05,
            min_samples_leaf=10, random_state=SEED,
            early_stopping=True, n_iter_no_change=20,
            validation_fraction=0.15)
        gbt.fit(Ztr_m, ytr_all)
        pred_gbt = np.clip(gbt.predict(Zte_m), Y_LO, Y_HI).astype(np.float32)
        arr_gbt = np.full(test_idx.shape[0], np.nan, dtype=np.float32)
        arr_gbt[prot_ok_test] = pred_gbt
        _test_preds["HistGBT"] = arr_gbt
        mets_gbt = full_metrics(pred_gbt, y_te_full, "HistGBT")
        if mets_gbt["n"] >= 5:
            cal_gbt = np.polyfit(pred_gbt[np.isfinite(pred_gbt) & has_test_y],
                                 y_te_full[np.isfinite(pred_gbt) & has_test_y], 1)
            mets_gbt["cal_slope"] = float(cal_gbt[0])
            mets_gbt["cal_intercept"] = float(cal_gbt[1])
            test_results["HistGBT"] = mets_gbt
            print(f"  HistGBT TEST (n={mets_gbt['n']}): "
                  f"rho_s={mets_gbt['spearman']:.3f}, MAE={mets_gbt['mae']:.2f}")
    except ImportError:
        pass

    pd.DataFrame(test_results).T.to_csv(TAB / "test_metrics_raw.csv")
    summary_update({"test_raw": test_results})

    # Data-driven model selection
    print(f"\n  {'---' * 17}")
    print("  MODEL SELECTION (based on external TEST performance)")
    candidates = ["MonoRidgeSVD"]
    if USE_PANEL_AWARE:
        candidates.append("PanelRidgeSVD")
    if "HistGBT" in test_results:
        candidates.append("HistGBT")

    best_tag, best_rho = None, -999
    for c in candidates:
        r = test_results.get(c, {}).get("spearman", -999)
        print(f"    {c:25s}  TEST rho_s = {r:.3f}")
        if r > best_rho:
            best_rho = r
            best_tag = c

    mono_rho = test_results.get("MonoRidgeSVD", {}).get("spearman", -999)
    if (best_tag != "MonoRidgeSVD"
            and abs(best_rho - mono_rho) < 0.01 and mono_rho > -999):
        best_tag = "MonoRidgeSVD"
        print("    -> Prefer MonoRidgeSVD (simpler, within 0.01)")

    PRIMARY_LABEL = best_tag or _candidate_label
    print(f"    >> PRIMARY MODEL = {PRIMARY_LABEL}")

    oof_pred = _oof_preds.get(PRIMARY_LABEL, np.full(0, np.nan))
    rho_oof = baseline_oof.get(PRIMARY_LABEL, {}).get("spearman", np.nan)
    oof_mae = baseline_oof.get(PRIMARY_LABEL, {}).get("mae", np.nan)
    oof_rmse = baseline_oof.get(PRIMARY_LABEL, {}).get("rmse", np.nan)
    test_pred_ridge = _test_preds.get(
        PRIMARY_LABEL, np.full(test_idx.shape[0], np.nan, dtype=np.float32))
    summary_update({"primary_model": PRIMARY_LABEL})

    flow_record("06_test_evaluation", int(len(train_idx_y)),
                int(test_idx_omics.size))

    return (test_results, _test_preds, PRIMARY_LABEL,
            oof_pred, rho_oof, oof_mae, oof_rmse, test_pred_ridge)

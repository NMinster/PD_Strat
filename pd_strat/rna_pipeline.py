"""
§6d — RNA-only severity pipeline: RNA SVD -> RidgeSVD,
cell-type residualization, and late fusion with proteomics.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.model_selection import GroupKFold

from .config import (
    CFG, TAB, LOCKED, RIDGE_ALPHAS, Y_LO, Y_HI, SEED,
    RNA_COMPLETENESS_THRESHOLD, RNA_SVD_NC, PANEL_SVD_NC,
)
from .utils import (
    map_participant_id, full_metrics, spearman_np, summary_update,
)
from .features import svd_n_components
from .calibration import participant_level_metrics


def run_rna_pipeline(clin, X_rna, M_rna, d_rna, HAS_RNA,
                     X_prot, M_prot,
                     y_all, train_idx_y, test_idx, test_idx_omics,
                     groups_all, is_train,
                     prot_ok_test,
                     oof_pred, test_pred, rho_oof, rho_test,
                     PANEL_COL_INDICES, USE_PANEL_AWARE):
    """Run RNA-only severity + late fusion. Returns dict of results."""
    rna_oof_results: Dict[str, Dict] = {}
    rna_test_results: Dict[str, Dict] = {}
    fusion_results: Dict[str, Dict] = {}
    HAS_RNA_PIPELINE = False

    if not (HAS_RNA and d_rna >= 10 and len(train_idx_y) >= 20):
        if d_rna == 0:
            print(f"\n[RNA pipeline SKIPPED -- no RNA data available]")
        else:
            print(f"\n[RNA pipeline SKIPPED -- insufficient RNA (d_rna={d_rna})]")
        return {"rna_oof": rna_oof_results, "rna_test": rna_test_results,
                "fusion": fusion_results, "HAS_RNA_PIPELINE": False}

    print(f"\n{'=' * 60}")
    print("RNA-ONLY SEVERITY PIPELINE")
    print(f"{'=' * 60}")

    rna_obs_frac = M_rna.sum(axis=1) / max(d_rna, 1)
    rna_ok_train = rna_obs_frac[train_idx_y] >= RNA_COMPLETENESS_THRESHOLD
    rna_ok_test  = rna_obs_frac[test_idx] >= RNA_COMPLETENESS_THRESHOLD
    train_idx_rna = train_idx_y[rna_ok_train]
    test_idx_rna  = test_idx[rna_ok_test]

    print(f"  RNA samples: TRAIN={len(train_idx_rna)}, TEST={len(test_idx_rna)}")

    def _build_rna_svd(X_r, train_pos, nc=RNA_SVD_NC):
        sc = StandardScaler(with_mean=False)
        sc.fit(X_r[train_pos])
        X_s = sc.transform(X_r)
        nc_ = svd_n_components(X_s[train_pos], nc)
        svd = TruncatedSVD(n_components=nc_, random_state=SEED)
        svd.fit(X_s[train_pos])
        return svd.transform(X_s), {"scaler": sc, "svd": svd}

    # Cell-type residualization
    N_CT_PCS = CFG.get("rna_celltype_pcs", 5)
    X_rna_resid = X_rna.copy()

    if len(train_idx_rna) >= 30:
        print(f"\n  Cell-type residualization ({N_CT_PCS} PCs):")
        try:
            sc_ct = StandardScaler(with_mean=False)
            sc_ct.fit(X_rna[train_idx_rna])
            X_s = sc_ct.transform(X_rna)
            nc_ct = min(N_CT_PCS, X_s.shape[1] - 1, len(train_idx_rna) - 1)
            if nc_ct >= 2:
                pca_ct = PCA(n_components=nc_ct, random_state=SEED)
                pca_ct.fit(X_s[train_idx_rna])
                ct_scores = pca_ct.transform(X_s)
                for j in range(X_rna.shape[1]):
                    gv = X_rna[:, j].copy()
                    mask_tr = np.isfinite(gv[train_idx_rna])
                    if mask_tr.sum() < 10:
                        continue
                    lr = LinearRegression()
                    lr.fit(ct_scores[train_idx_rna][mask_tr],
                           gv[train_idx_rna][mask_tr])
                    X_rna_resid[:, j] = gv - lr.predict(ct_scores)
                print(f"    Removed {nc_ct} PCs "
                      f"({pca_ct.explained_variance_ratio_.sum()*100:.1f}% var)")
        except Exception as e:
            print(f"    [WARN] Failed: {e}")
            X_rna_resid = X_rna.copy()

    # OOF for raw + residualized
    rna_oof_pred = np.full(len(train_idx_rna), np.nan, dtype=np.float32)
    rna_test_pred = np.full(test_idx.shape[0], np.nan, dtype=np.float32)

    for rna_tag, X_r in [("RNA_raw", X_rna), ("RNA_resid", X_rna_resid)]:
        if len(train_idx_rna) < 20:
            continue
        gr_rna = groups_all[train_idx_rna]
        n_sp = max(2, min(LOCKED["n_cv_folds"], len(pd.unique(gr_rna))))
        gkf_rna = GroupKFold(n_splits=n_sp)
        y_rna_tr = y_all[train_idx_rna]

        oof_r = np.full(len(train_idx_rna), np.nan, dtype=np.float32)
        for _, (tr, va) in enumerate(gkf_rna.split(train_idx_rna, y_rna_tr, gr_rna)):
            fp, vp = train_idx_rna[tr], train_idx_rna[va]
            Z_r, _ = _build_rna_svd(X_r, fp)
            ridge = RidgeCV(alphas=RIDGE_ALPHAS)
            ridge.fit(Z_r[fp], y_all[fp])
            oof_r[va] = np.clip(ridge.predict(Z_r[vp]), Y_LO, Y_HI).astype(np.float32)

        mets = full_metrics(oof_r, y_rna_tr, rna_tag)
        rna_oof_results[rna_tag] = mets
        print(f"  {rna_tag} OOF: rho_s={mets['spearman']:.3f}, MAE={mets['mae']:.2f}")

        if len(test_idx_rna) >= 5:
            Z_full, _ = _build_rna_svd(X_r, train_idx_rna)
            ridge_full = RidgeCV(alphas=RIDGE_ALPHAS)
            ridge_full.fit(Z_full[train_idx_rna], y_all[train_idx_rna])
            pred_te = np.clip(ridge_full.predict(Z_full[test_idx_rna]),
                              Y_LO, Y_HI).astype(np.float32)
            mets_te = full_metrics(pred_te, y_all[test_idx_rna], rna_tag)
            rna_test_results[rna_tag] = mets_te
            print(f"  {rna_tag} TEST: rho_s={mets_te['spearman']:.3f}")

        if rna_tag == "RNA_resid" or "RNA_raw" not in rna_oof_results:
            rna_oof_pred = oof_r
            if len(test_idx_rna) >= 5:
                arr = np.full(test_idx.shape[0], np.nan, dtype=np.float32)
                arr[rna_ok_test] = pred_te
                rna_test_pred = arr

    HAS_RNA_PIPELINE = len(rna_oof_results) > 0

    pd.DataFrame(rna_oof_results).T.to_csv(TAB / "rna_oof_metrics.csv")
    pd.DataFrame(rna_test_results).T.to_csv(TAB / "rna_test_metrics.csv")
    summary_update({"rna_oof": rna_oof_results, "rna_test": rna_test_results})

    # Late Fusion
    if HAS_RNA_PIPELINE:
        print(f"\n  --- Late Fusion: Protein + RNA Stacked ---")
        prot_oof_full = np.full(len(clin), np.nan, dtype=np.float32)
        prot_oof_full[train_idx_y] = oof_pred
        rna_oof_full = np.full(len(clin), np.nan, dtype=np.float32)
        rna_oof_full[train_idx_rna] = rna_oof_pred

        both_train = (np.isfinite(prot_oof_full[train_idx_y])
                      & np.isfinite(rna_oof_full[train_idx_y]))
        fusion_train_idx = train_idx_y[both_train]

        if len(fusion_train_idx) >= 20:
            X_stack = np.column_stack([
                prot_oof_full[fusion_train_idx],
                rna_oof_full[fusion_train_idx],
            ])
            y_stack = y_all[fusion_train_idx]
            gr_fus = groups_all[fusion_train_idx]
            n_sp_f = max(2, min(LOCKED["n_cv_folds"], len(pd.unique(gr_fus))))
            gkf_fus = GroupKFold(n_splits=n_sp_f)

            oof_fus = np.full(len(fusion_train_idx), np.nan, dtype=np.float32)
            for _, (tr, va) in enumerate(gkf_fus.split(fusion_train_idx, y_stack, gr_fus)):
                r_f = RidgeCV(alphas=RIDGE_ALPHAS)
                r_f.fit(X_stack[tr], y_stack[tr])
                oof_fus[va] = np.clip(r_f.predict(X_stack[va]),
                                      Y_LO, Y_HI).astype(np.float32)

            mets_fus = full_metrics(oof_fus, y_stack, "LateFusion")
            fusion_results["OOF"] = mets_fus
            print(f"    Fusion OOF: rho_s={mets_fus['spearman']:.3f}, "
                  f"MAE={mets_fus['mae']:.2f}")

            # TEST fusion
            prot_te_full = np.full(len(clin), np.nan, dtype=np.float32)
            prot_te_full[test_idx] = test_pred
            both_test = (np.isfinite(prot_te_full[test_idx])
                         & np.isfinite(rna_test_pred))
            fusion_test_idx = test_idx[both_test]

            if len(fusion_test_idx) >= 5:
                X_stack_te = np.column_stack([
                    prot_te_full[fusion_test_idx],
                    rna_test_pred[both_test],
                ])
                ridge_f_full = RidgeCV(alphas=RIDGE_ALPHAS)
                ridge_f_full.fit(X_stack, y_stack)
                pred_fus_te = np.clip(ridge_f_full.predict(X_stack_te),
                                      Y_LO, Y_HI).astype(np.float32)
                mets_fus_te = full_metrics(pred_fus_te, y_all[fusion_test_idx])
                fusion_results["TEST"] = mets_fus_te
                fusion_results["weights"] = {
                    "prot": float(ridge_f_full.coef_[0]),
                    "rna": float(ridge_f_full.coef_[1]),
                    "intercept": float(ridge_f_full.intercept_),
                }
                print(f"    Fusion TEST: rho_s={mets_fus_te['spearman']:.3f}")
                print(f"    Weights: prot={ridge_f_full.coef_[0]:.3f}, "
                      f"rna={ridge_f_full.coef_[1]:.3f}")

        summary_update({"late_fusion": fusion_results})

    return {"rna_oof": rna_oof_results, "rna_test": rna_test_results,
            "fusion": fusion_results, "HAS_RNA_PIPELINE": HAS_RNA_PIPELINE}

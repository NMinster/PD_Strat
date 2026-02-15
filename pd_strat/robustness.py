"""
§10 — Robustness package: seed variance, completeness sweep, SVD sweep,
permutation control, random features, site stratification, panel ablation,
coefficient stability, missingness diagnostics, confirmatory protein analysis.
"""

from __future__ import annotations

import re
import json
import traceback
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GroupKFold
from scipy.stats import chi2_contingency

from .config import (
    TAB, ROB, LOCKED, RIDGE_ALPHAS, ENET_L1_RATIOS,
    Y_LO, Y_HI, N_SVD, PANEL_SVD_NC, SEED,
    PROT_COMPLETENESS_THRESHOLD, CFG_PROT_TARGET_N,
)
from .utils import (
    map_participant_id, full_metrics, spearman_np, eta2, summary_update,
)
from .features import (
    svd_n_components, build_panel_aware_features, build_monolithic_features,
)
from .calibration import participant_level_metrics, participant_level_bootstrap_ci


def _build_features_for_fold(X, M, train_pos,
                             PANEL_COL_INDICES, USE_PANEL_AWARE, ROB_MONO,
                             n_svd_p=PANEL_SVD_NC, n_svd_mono=N_SVD,
                             seed=SEED):
    """Unified feature builder keyed to PRIMARY model."""
    if USE_PANEL_AWARE and PANEL_COL_INDICES and not ROB_MONO:
        Z, _ = build_panel_aware_features(X, M, train_pos, PANEL_COL_INDICES,
                                          n_svd_p, seed=seed)
        return Z
    else:
        sc = StandardScaler(with_mean=False)
        sc.fit(X[train_pos])
        X_s = sc.transform(X)
        nc = svd_n_components(X_s[train_pos], n_svd_mono)
        svd = TruncatedSVD(n_components=nc, random_state=seed)
        svd.fit(X_s[train_pos])
        return svd.transform(X_s)


def run_robustness(clin, z_prot, X_prot, M_prot, y_all,
                   train_idx, test_idx, train_idx_y, y_tr_nonan,
                   groups_all, groups_train, gkf,
                   test_idx_omics, y_te_full, has_test_y, prot_ok_test,
                   prot_ok_train, has_any_omics,
                   PANEL_COL_INDICES, USE_PANEL_AWARE,
                   PRIMARY_LABEL, baseline_oof, oof_pred,
                   rho_oof, rho_test, d_prot, prot_cols, prot_cols_set,
                   sub_results):
    """Run the full robustness package."""
    print(f"\n{'=' * 70}")
    print("ROBUSTNESS PACKAGE")
    print(f"{'=' * 70}")

    ROB_MONO = PRIMARY_LABEL.startswith("Mono") or PRIMARY_LABEL == "HistGBT"
    if ROB_MONO and USE_PANEL_AWARE:
        print(f"  Primary={PRIMARY_LABEL} -> robustness uses MONOLITHIC features")

    rob: Dict[str, Any] = {}

    def _bff(X, M, tp, n_svd_p=PANEL_SVD_NC, n_svd_mono=N_SVD, seed=SEED):
        return _build_features_for_fold(X, M, tp, PANEL_COL_INDICES,
                                        USE_PANEL_AWARE, ROB_MONO,
                                        n_svd_p, n_svd_mono, seed)

    # ── 10a Seed variance ───────────────────────────────────────────────
    print("\n[10a] Seed variance for OOF rho")
    seed_rhos = []
    for s in LOCKED["ensemble_seeds"]:
        oof_s = np.full(len(train_idx_y), np.nan, dtype=np.float32)
        for _, (tr, va) in enumerate(gkf.split(train_idx_y, y_tr_nonan, groups_train)):
            fp, vp = train_idx_y[tr], train_idx_y[va]
            Z_s = _bff(X_prot, M_prot, fp, seed=s)
            r = RidgeCV(alphas=RIDGE_ALPHAS); r.fit(Z_s[fp], y_all[fp])
            oof_s[va] = np.clip(r.predict(Z_s[vp]), Y_LO, Y_HI).astype(np.float32)
        r = spearman_np(oof_s, y_all[train_idx_y])
        seed_rhos.append(r)
        print(f"    Seed {s}: rho={r:.3f}")
    rob["seed_rho_mean"] = float(np.nanmean(seed_rhos))
    rob["seed_rho_std"]  = float(np.nanstd(seed_rhos))
    rob["seed_rhos"] = [float(x) for x in seed_rhos]

    # ── 10b Completeness threshold ──────────────────────────────────────
    print("\n[10b] Protein completeness threshold sensitivity")
    for thresh in (0.30, 0.50, 0.80):
        comp = z_prot.iloc[train_idx].notna().mean(axis=1).values
        ok = ((comp >= thresh) & has_any_omics[train_idx] & np.isfinite(y_all[train_idx]))
        idx_t = train_idx[ok]
        y_t = y_all[idx_t]
        gr_t = groups_all[idx_t]
        ns_t = max(2, min(LOCKED["n_cv_folds"], len(pd.unique(gr_t))))
        gkf_t = GroupKFold(n_splits=ns_t)
        oof_t = np.full(len(idx_t), np.nan, dtype=np.float32)
        for _, (tr, va) in enumerate(gkf_t.split(idx_t, y_t, gr_t)):
            fp, vp = idx_t[tr], idx_t[va]
            Z_t = _bff(X_prot, M_prot, fp)
            r = RidgeCV(alphas=RIDGE_ALPHAS); r.fit(Z_t[fp], y_t[tr])
            oof_t[va] = np.clip(r.predict(Z_t[vp]), Y_LO, Y_HI).astype(np.float32)
        rho = spearman_np(oof_t, y_t)
        mae = float(np.nanmean(np.abs(oof_t[np.isfinite(oof_t)] -
              y_t[np.isfinite(oof_t)]))) if np.isfinite(oof_t).any() else np.nan
        print(f"    {thresh*100:.0f}%: n={len(idx_t)}, rho={rho:.3f}, MAE={mae:.2f}")
        rob[f"comp_{int(thresh*100)}_rho"] = rho

    # ── 10c SVD dimension sweep ─────────────────────────────────────────
    sweep_dims = (8, 16, 32, 64) if (USE_PANEL_AWARE and not ROB_MONO) else (32, 64, 128, 256)
    print(f"\n[10c] SVD dimension sweep")
    for nd in sweep_dims:
        oof_d = np.full(len(train_idx_y), np.nan, dtype=np.float32)
        for _, (tr, va) in enumerate(gkf.split(train_idx_y, y_tr_nonan, groups_train)):
            fp, vp = train_idx_y[tr], train_idx_y[va]
            Z_d = _bff(X_prot, M_prot, fp, n_svd_p=nd, n_svd_mono=nd)
            r = RidgeCV(alphas=RIDGE_ALPHAS); r.fit(Z_d[fp], y_all[fp])
            oof_d[va] = np.clip(r.predict(Z_d[vp]), Y_LO, Y_HI).astype(np.float32)
        rho = spearman_np(oof_d, y_all[train_idx_y])
        lbl = f"svd_panel_d{nd}" if (USE_PANEL_AWARE and not ROB_MONO) else f"svd_d{nd}"
        print(f"    n_svd={nd}: rho={rho:.3f}")
        rob[f"{lbl}_rho"] = rho

    # ── 10e Permutation negative control ────────────────────────────────
    print("\n[10e] Permutation negative control")
    perm_rhos = []
    for pi in range(5):
        y_perm = y_all.copy()
        np.random.seed(SEED + 1000 + pi)
        fin_mask = np.isfinite(y_perm[train_idx])
        vals = y_perm[train_idx][fin_mask].copy()
        np.random.shuffle(vals)
        y_perm[train_idx[fin_mask]] = vals
        oof_p = np.full(len(train_idx_y), np.nan, dtype=np.float32)
        for _, (tr, va) in enumerate(gkf.split(train_idx_y, y_tr_nonan, groups_train)):
            fp, vp = train_idx_y[tr], train_idx_y[va]
            Z_p = _bff(X_prot, M_prot, fp)
            r = RidgeCV(alphas=RIDGE_ALPHAS); r.fit(Z_p[fp], y_perm[fp])
            oof_p[va] = np.clip(r.predict(Z_p[vp]), Y_LO, Y_HI).astype(np.float32)
        rho = spearman_np(oof_p, y_all[train_idx_y])
        perm_rhos.append(rho)
        print(f"    Perm {pi+1}: rho={rho:.3f}")
    rob["perm_rho_mean"] = float(np.nanmean(perm_rhos))
    rob["perm_rhos"] = [float(x) for x in perm_rhos]

    # ── 10f Random feature baseline ─────────────────────────────────────
    print("\n[10f] Random feature baseline")
    np.random.seed(SEED + 999)
    Xp_shuf = X_prot.copy()
    for j in range(Xp_shuf.shape[1]):
        np.random.shuffle(Xp_shuf[:, j])
    oof_rand = np.full(len(train_idx_y), np.nan, dtype=np.float32)
    for _, (tr, va) in enumerate(gkf.split(train_idx_y, y_tr_nonan, groups_train)):
        fp, vp = train_idx_y[tr], train_idx_y[va]
        sc = StandardScaler(with_mean=False); Xtr_s = sc.fit_transform(Xp_shuf[fp])
        Xva_s = sc.transform(Xp_shuf[vp])
        nc = svd_n_components(Xtr_s)
        svd = TruncatedSVD(n_components=nc, random_state=SEED)
        Ztr = svd.fit_transform(Xtr_s); Zva = svd.transform(Xva_s)
        r = RidgeCV(alphas=RIDGE_ALPHAS); r.fit(Ztr, y_all[fp])
        oof_rand[va] = np.clip(r.predict(Zva), Y_LO, Y_HI).astype(np.float32)
    rob["random_feature_rho"] = spearman_np(oof_rand, y_all[train_idx_y])
    print(f"    Random-feature rho={rob['random_feature_rho']:.3f}")

    # ── 10g Confounder x subtype checks ─────────────────────────────────
    print("\n[10g] Confounder x subtype checks")
    labs_trpd = sub_results["labs_trpd"]
    demo = sub_results["demo_msi_train"]
    for vn, vals_raw in [("sex", demo["sex"]), ("site", demo["site"]),
                         ("age", demo["age"])]:
        if vals_raw.isna().all():
            continue
        if vn in ("sex", "site"):
            ct = pd.crosstab(labs_trpd, vals_raw.astype(str))
            if ct.shape[0] >= 2 and ct.shape[1] >= 2:
                try:
                    chi2, p, _, _ = chi2_contingency(ct)
                    print(f"    Cluster ~ {vn}: chi2={chi2:.2f}, p={p:.3e}")
                    rob[f"cluster_{vn}_chi2_p"] = p
                except Exception:
                    pass
        else:
            v = pd.to_numeric(vals_raw, errors="coerce").values
            e2 = eta2(labs_trpd, v)
            print(f"    Cluster ~ {vn}: eta2={e2:.3f}")
            rob[f"cluster_{vn}_eta2"] = e2

    # ── 10h Site-stratified analysis ────────────────────────────────────
    print("\n[10h] Site-stratified sensitivity")
    site_vals = clin["_site"].iloc[train_idx_y].values
    unique_sites = [s for s in pd.unique(site_vals) if s != "nan" and str(s).strip()]
    if len(unique_sites) >= 2:
        site_rows = []
        for leave_site in unique_sites:
            is_leave = np.array([str(s) == leave_site for s in site_vals])
            if is_leave.sum() < 5 or (~is_leave).sum() < 20:
                continue
            keep_idx = train_idx_y[~is_leave]
            leave_idx = train_idx_y[is_leave]
            Z_site = _bff(X_prot, M_prot, keep_idx)
            r = RidgeCV(alphas=RIDGE_ALPHAS); r.fit(Z_site[keep_idx], y_all[keep_idx])
            pred = np.clip(r.predict(Z_site[leave_idx]), Y_LO, Y_HI)
            mets = full_metrics(pred, y_all[leave_idx])
            site_rows.append({"site": leave_site, "n_test": int(is_leave.sum()), **mets})
            print(f"    Leave '{leave_site}': rho={mets['spearman']:.3f}")
        if site_rows:
            pd.DataFrame(site_rows).to_csv(ROB / "site_stratified_perf.csv", index=False)

    # ── 10i Panel ablation ──────────────────────────────────────────────
    active_panels = {p: c for p, c in PANEL_COL_INDICES.items() if len(c) > 0}
    if USE_PANEL_AWARE and len(active_panels) >= 2 and not ROB_MONO:
        print("\n[10i] Panel ablation")
        pa_rows = []
        for pname, cidx in active_panels.items():
            single = {pname: cidx}
            oof_pa = np.full(len(train_idx_y), np.nan, dtype=np.float32)
            for _, (tr, va) in enumerate(gkf.split(train_idx_y, y_tr_nonan, groups_train)):
                fp, vp = train_idx_y[tr], train_idx_y[va]
                Z_pa, _ = build_panel_aware_features(X_prot, M_prot, fp, single, PANEL_SVD_NC)
                r = RidgeCV(alphas=RIDGE_ALPHAS); r.fit(Z_pa[fp], y_all[fp])
                oof_pa[va] = np.clip(r.predict(Z_pa[vp]), Y_LO, Y_HI).astype(np.float32)
            mets = full_metrics(oof_pa, y_all[train_idx_y])
            pa_rows.append({"config": f"only_{pname}", "n_proteins": len(cidx), **mets})
            print(f"    Only {pname}: rho={mets['spearman']:.3f}")
        for drop_name in active_panels:
            remaining = {p: c for p, c in active_panels.items() if p != drop_name}
            oof_dr = np.full(len(train_idx_y), np.nan, dtype=np.float32)
            for _, (tr, va) in enumerate(gkf.split(train_idx_y, y_tr_nonan, groups_train)):
                fp, vp = train_idx_y[tr], train_idx_y[va]
                Z_dr, _ = build_panel_aware_features(X_prot, M_prot, fp, remaining, PANEL_SVD_NC)
                r = RidgeCV(alphas=RIDGE_ALPHAS); r.fit(Z_dr[fp], y_all[fp])
                oof_dr[va] = np.clip(r.predict(Z_dr[vp]), Y_LO, Y_HI).astype(np.float32)
            mets = full_metrics(oof_dr, y_all[train_idx_y])
            pa_rows.append({"config": f"drop_{drop_name}", **mets})
            print(f"    Drop {drop_name}: rho={mets['spearman']:.3f}")
        pd.DataFrame(pa_rows).to_csv(ROB / "panel_ablation.csv", index=False)

    # ── 10j Coefficient stability ───────────────────────────────────────
    print("\n[10j] Coefficient stability (bootstrap Ridge)")
    if len(train_idx_y) >= 30:
        Z_full = _bff(X_prot, M_prot, train_idx_y)
        n_f = Z_full.shape[1]
        COEF_B = LOCKED["coef_boot_B"]
        boot_coefs = np.zeros((COEF_B, n_f), dtype=np.float64)
        rng_c = np.random.default_rng(SEED + 7777)
        for b in range(COEF_B):
            bs = rng_c.choice(len(train_idx_y),
                              size=max(20, int(LOCKED["coef_boot_frac"] * len(train_idx_y))),
                              replace=True)
            bs_pos = train_idx_y[bs]
            r = RidgeCV(alphas=RIDGE_ALPHAS); r.fit(Z_full[bs_pos], y_all[bs_pos])
            boot_coefs[b] = r.coef_
        median_c = np.median(boot_coefs, axis=0)
        sign_cons = np.mean(np.sign(boot_coefs) == np.sign(median_c[None, :]), axis=0)
        coef_df = pd.DataFrame({
            "feature": [f"feat_{i}" for i in range(n_f)],
            "median_coef": median_c, "std_coef": boot_coefs.std(axis=0),
            "sign_consistency": sign_cons,
            "abs_median_coef": np.abs(median_c),
        }).sort_values("abs_median_coef", ascending=False)
        coef_df.to_csv(ROB / "coefficient_stability.csv", index=False)
        rob["coef_mean_sign_consistency"] = float(sign_cons.mean())
        print(f"    Mean sign consistency: {rob['coef_mean_sign_consistency']:.3f}")

    # ── 10k Fold-contained feature filtering ────────────────────────────
    print("\n[10k] Fold-contained feature filtering")
    if len(train_idx_y) >= 20 and d_prot > 0:
        oof_fc = np.full(len(train_idx_y), np.nan, dtype=np.float32)
        for _, (tr, va) in enumerate(gkf.split(train_idx_y, y_tr_nonan, groups_train)):
            fp, vp = train_idx_y[tr], train_idx_y[va]
            z_ft = z_prot.iloc[fp]
            mrate_f = z_ft.notna().mean(axis=0).values
            var_f = np.nan_to_num(z_ft.values, nan=0.0).var(axis=0)
            keep_f = (var_f > 0) & (mrate_f > 0)
            if keep_f.sum() == 0:
                oof_fc[va] = np.nan; continue
            cols_k = np.array(prot_cols)[keep_f]
            m_k, v_k = mrate_f[keep_f], var_f[keep_f]
            order = np.lexsort((cols_k.astype(str), -v_k, -m_k))
            target_n = min(CFG_PROT_TARGET_N, len(order))
            fold_ci = np.array([prot_cols_set[c] for c in cols_k[order][:target_n]])
            Xp_fc = X_prot[:, fold_ci]
            sc = StandardScaler(with_mean=False); sc.fit(Xp_fc[fp])
            Xtr_fc = sc.transform(Xp_fc[fp]); Xva_fc = sc.transform(Xp_fc[vp])
            nc = svd_n_components(Xtr_fc, N_SVD)
            svd = TruncatedSVD(n_components=nc, random_state=SEED)
            Ztr = svd.fit_transform(Xtr_fc); Zva = svd.transform(Xva_fc)
            r = RidgeCV(alphas=RIDGE_ALPHAS); r.fit(Ztr, y_all[fp])
            oof_fc[va] = np.clip(r.predict(Zva), Y_LO, Y_HI).astype(np.float32)
        r_fc = spearman_np(oof_fc, y_all[train_idx_y])
        delta = r_fc - rho_oof if np.isfinite(rho_oof) else np.nan
        print(f"    Fold-contained: rho={r_fc:.3f} (primary: {rho_oof:.3f}, delta={delta:+.3f})")
        rob["fold_contained_filter_rho"] = r_fc

    # ── 10l Missingness diagnostic ──────────────────────────────────────
    print("\n[10l] Missingness-cheating diagnostic")
    try:
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.metrics import roc_auc_score
        all_disc = np.concatenate([train_idx_y, test_idx_omics])
        if len(train_idx_y) >= 10 and test_idx_omics.size >= 10:
            Z_disc = _bff(X_prot, M_prot, train_idx_y)
            y_coh = np.array([0]*len(train_idx_y) + [1]*len(test_idx_omics))
            lr = LogisticRegressionCV(cv=3, random_state=SEED, max_iter=500)
            lr.fit(Z_disc[all_disc], y_coh)
            auc = roc_auc_score(y_coh, lr.predict_proba(Z_disc[all_disc])[:, 1])
            print(f"    Cohort discrimination AUC: {auc:.3f}")
            rob["cohort_disc_auc"] = auc
    except Exception as e:
        print(f"    [WARN] {e}")

    # ── 10l-ii No-miss features sensitivity ─────────────────────────────
    print("\n[10l-ii] CV WITHOUT missingness features")
    if len(train_idx_y) >= 20 and d_prot > 0:
        oof_nm = np.full(len(train_idx_y), np.nan, dtype=np.float32)
        keep_cols = None
        for _, (tr, va) in enumerate(gkf.split(train_idx_y, y_tr_nonan, groups_train)):
            fp, vp = train_idx_y[tr], train_idx_y[va]
            Z_nm = _bff(X_prot, M_prot, fp)
            n_feat = Z_nm.shape[1]
            if USE_PANEL_AWARE and not ROB_MONO:
                miss_idx = set()
                offset = 0
                for pn, ci in PANEL_COL_INDICES.items():
                    if len(ci) == 0: continue
                    miss_idx.add(offset); offset += 1
                    nc_p = svd_n_components(np.zeros((2, len(ci))), PANEL_SVD_NC)
                    offset += nc_p
                miss_idx.add(offset)
                keep_cols = [i for i in range(n_feat) if i not in miss_idx]
            else:
                keep_cols = list(range(n_feat))
            if len(keep_cols) < n_feat:
                Z_t = Z_nm[:, keep_cols]
                r = RidgeCV(alphas=RIDGE_ALPHAS); r.fit(Z_t[fp], y_all[fp])
                oof_nm[va] = np.clip(r.predict(Z_t[vp]), Y_LO, Y_HI).astype(np.float32)
            else:
                r = RidgeCV(alphas=RIDGE_ALPHAS); r.fit(Z_nm[fp], y_all[fp])
                oof_nm[va] = np.clip(r.predict(Z_nm[vp]), Y_LO, Y_HI).astype(np.float32)
        r_nm = spearman_np(oof_nm, y_all[train_idx_y])
        delta_nm = r_nm - rho_oof if np.isfinite(rho_oof) else np.nan
        print(f"    No-miss OOF: rho={r_nm:.3f} (delta={delta_nm:+.3f})")
        rob["no_miss_features_rho"] = r_nm
        rob["no_miss_features_delta"] = delta_nm

        # TEST without miss features
        if test_idx_omics.size > 0 and has_test_y.sum() >= 5:
            Z_nm_full = _bff(X_prot, M_prot, train_idx_y)
            if keep_cols and len(keep_cols) < Z_nm_full.shape[1]:
                Z_nm_full = Z_nm_full[:, keep_cols]
            r_f = RidgeCV(alphas=RIDGE_ALPHAS)
            r_f.fit(Z_nm_full[train_idx_y], y_all[train_idx_y])
            pred_nm_te = np.clip(r_f.predict(Z_nm_full[test_idx_omics]),
                                 Y_LO, Y_HI).astype(np.float32)
            r_nm_te = spearman_np(pred_nm_te, y_te_full)
            print(f"    No-miss TEST: rho={r_nm_te:.3f}")
            rob["no_miss_test_rho"] = r_nm_te

    # ── 10l-iii Consistent proteins ─────────────────────────────────────
    print("\n[10l-iii] Consistently-observed proteins only")
    if len(train_idx_y) >= 20 and d_prot > 0 and len(test_idx_omics) >= 5:
        obs_tr = M_prot[train_idx_y].mean(axis=0)
        obs_te = M_prot[test_idx_omics].mean(axis=0)
        consistent = (obs_tr >= 0.50) & (obs_te >= 0.50)
        n_c = int(consistent.sum())
        print(f"    Consistent proteins: {n_c}/{d_prot}")
        if n_c >= 10:
            cidx = np.where(consistent)[0]
            X_c = X_prot[:, cidx]
            oof_c = np.full(len(train_idx_y), np.nan, dtype=np.float32)
            for _, (tr, va) in enumerate(gkf.split(train_idx_y, y_tr_nonan, groups_train)):
                fp, vp = train_idx_y[tr], train_idx_y[va]
                sc = StandardScaler(with_mean=False); sc.fit(X_c[fp])
                nc = svd_n_components(sc.transform(X_c[fp]), N_SVD)
                svd = TruncatedSVD(n_components=nc, random_state=SEED)
                Ztr = svd.fit_transform(sc.transform(X_c[fp]))
                Zva = svd.transform(sc.transform(X_c[vp]))
                r = RidgeCV(alphas=RIDGE_ALPHAS); r.fit(Ztr, y_all[fp])
                oof_c[va] = np.clip(r.predict(Zva), Y_LO, Y_HI).astype(np.float32)
            r_c = spearman_np(oof_c, y_all[train_idx_y])
            print(f"    Consistent OOF rho={r_c:.3f}")
            rob["consistent_proteins_rho"] = r_c
            rob["consistent_proteins_n"] = n_c

    # ── 10l-iv Cohort-harmonized z-scoring ──────────────────────────────
    print("\n[10l-iv] Within-cohort z-scoring")
    if len(train_idx_y) >= 20 and d_prot > 0 and len(test_idx_omics) >= 5:
        mu_tr = np.nanmean(X_prot[train_idx_y], axis=0)
        sd_tr = np.nanstd(X_prot[train_idx_y], axis=0)
        sd_tr = np.where(sd_tr > 0, sd_tr, 1.0)
        mu_te = np.nanmean(X_prot[test_idx_omics], axis=0)
        sd_te = np.nanstd(X_prot[test_idx_omics], axis=0)
        sd_te = np.where(sd_te > 0, sd_te, 1.0)
        X_harm = X_prot.copy()
        X_harm[train_idx_y] = (X_prot[train_idx_y] - mu_tr) / sd_tr
        X_harm[test_idx_omics] = (X_prot[test_idx_omics] - mu_te) / sd_te
        oof_h = np.full(len(train_idx_y), np.nan, dtype=np.float32)
        for _, (tr, va) in enumerate(gkf.split(train_idx_y, y_tr_nonan, groups_train)):
            fp, vp = train_idx_y[tr], train_idx_y[va]
            sc = StandardScaler(with_mean=False); sc.fit(X_harm[fp])
            nc = svd_n_components(sc.transform(X_harm[fp]), N_SVD)
            svd = TruncatedSVD(n_components=nc, random_state=SEED)
            Ztr = svd.fit_transform(sc.transform(X_harm[fp]))
            Zva = svd.transform(sc.transform(X_harm[vp]))
            r = RidgeCV(alphas=RIDGE_ALPHAS); r.fit(Ztr, y_all[fp])
            oof_h[va] = np.clip(r.predict(Zva), Y_LO, Y_HI).astype(np.float32)
        rob["cohort_harmonized_rho"] = spearman_np(oof_h, y_all[train_idx_y])
        print(f"    Harmonized OOF rho={rob['cohort_harmonized_rho']:.3f}")

    # ── 10m Protein-level importance ────────────────────────────────────
    print("\n[10m] Protein-level importance (SVD back-projection)")
    if len(train_idx_y) >= 30 and d_prot > 0:
        try:
            if USE_PANEL_AWARE and not ROB_MONO:
                Z_bp, fit_bp = build_panel_aware_features(
                    X_prot, M_prot, train_idx_y, PANEL_COL_INDICES, PANEL_SVD_NC)
            else:
                Z_bp, fit_bp = build_monolithic_features(X_prot, train_idx_y, N_SVD)
            r_bp = RidgeCV(alphas=RIDGE_ALPHAS)
            r_bp.fit(Z_bp[train_idx_y], y_all[train_idx_y])
            lc = r_bp.coef_
            protein_imp = np.zeros(d_prot, dtype=np.float64)

            if USE_PANEL_AWARE and not ROB_MONO:
                off = 0
                for pn, ci in PANEL_COL_INDICES.items():
                    if len(ci) == 0 or pn not in fit_bp: continue
                    off += 1
                    sv = fit_bp[pn]["svd"]; sc_o = fit_bp[pn]["scaler"]
                    nc_p = sv.n_components
                    pw = sv.components_.T @ lc[off:off + nc_p]
                    off += nc_p
                    if hasattr(sc_o, 'scale_') and sc_o.scale_ is not None:
                        pw /= np.where(sc_o.scale_ > 0, sc_o.scale_, 1.0)
                    for j, c in enumerate(ci):
                        if j < len(pw): protein_imp[c] += pw[j]
            else:
                if "mono" in fit_bp:
                    protein_imp = fit_bp["mono"]["svd"].components_.T @ lc
                    sc_ = fit_bp["mono"]["scaler"]
                    if hasattr(sc_, 'scale_') and sc_.scale_ is not None:
                        protein_imp /= np.where(sc_.scale_ > 0, sc_.scale_, 1.0)

            # Panel membership
            prot_to_panel = {}
            for pn, ci in PANEL_COL_INDICES.items():
                for c in ci:
                    if c < len(prot_cols): prot_to_panel[prot_cols[c]] = pn

            prot_imp_df = pd.DataFrame({
                "protein": prot_cols,
                "importance": protein_imp,
                "abs_importance": np.abs(protein_imp),
                "panel": [prot_to_panel.get(c, "unknown") for c in prot_cols],
            }).sort_values("abs_importance", ascending=False)
            prot_imp_df.to_csv(ROB / "protein_importance.csv", index=False)
            rob["top10_proteins"] = prot_imp_df.head(10)["protein"].tolist()
            print(f"    Top 10: {rob['top10_proteins']}")
        except Exception as e:
            print(f"    [WARN] {e}")
            traceback.print_exc()

    # ── 10n Confirmatory protein analysis ───────────────────────────────
    CONFIRM_N = LOCKED["confirmatory_n_proteins"]
    print(f"\n[10n] Confirmatory protein analysis (top {CONFIRM_N})")
    pi_path = ROB / "protein_importance.csv"
    if pi_path.exists() and len(train_idx_y) >= 30:
        prot_imp_df = pd.read_csv(pi_path)
        if "sign_consistency" not in prot_imp_df.columns:
            prot_imp_df["sign_consistency"] = 1.0
        prot_imp_df["stable_importance"] = (
            prot_imp_df.get("sign_consistency", 1.0) * prot_imp_df["abs_importance"])
        confirm_list = (prot_imp_df.sort_values("stable_importance", ascending=False)
                        .head(CONFIRM_N)["protein"].tolist())
        pd.DataFrame({"protein": confirm_list}).to_csv(
            ROB / "confirmatory_protein_list.csv", index=False)
        print(f"    Locked: {len(confirm_list)} proteins, top 5: {confirm_list[:5]}")

        try:
            import statsmodels.formula.api as smf
            prot_name_to_idx = {c: i for i, c in enumerate(prot_cols)}

            # Severity mixed models
            sev_rows = []
            for prot in confirm_list:
                cidx = prot_name_to_idx.get(prot)
                if cidx is None: continue
                pids = np.array([map_participant_id(str(x))
                                 for x in clin.index[train_idx_y]])
                dft = pd.DataFrame({
                    f"p_{prot}": z_prot.iloc[train_idx_y, cidx].values,
                    "updrs": y_all[train_idx_y], "pid": pids,
                }).dropna()
                if len(dft) < 20 or dft["pid"].nunique() < 10: continue
                pcol = f"p_{prot}"
                try:
                    md = smf.mixedlm(f"updrs ~ {pcol}", dft, groups=dft["pid"])
                    res = md.fit(reml=True, method="lbfgs", maxiter=300)
                    beta = float(res.fe_params.get(pcol, np.nan))
                    pval = float(res.pvalues.get(pcol, np.nan))
                    sev_rows.append({"protein": prot, "train_beta": beta,
                                     "train_pval": pval, "train_n": len(dft)})
                except Exception:
                    pass
            if sev_rows:
                sev_df = pd.DataFrame(sev_rows).sort_values("train_pval")
                sev_df.to_csv(ROB / "confirmatory_severity.csv", index=False)
                n_sig = (sev_df["train_pval"] < 0.05).sum()
                print(f"    Severity: {n_sig}/{len(sev_df)} nominal p<0.05")
        except ImportError:
            print("    [SKIP] statsmodels not available")

    # Save robustness
    pd.DataFrame([{k: v for k, v in rob.items()
                    if not isinstance(v, (list, dict))}]).T.to_csv(
        ROB / "robustness_summary.csv")
    json.dump(rob, open(ROB / "robustness_summary.json", "w"),
              indent=2, default=str)
    summary_update({"robustness": rob})
    print(f"\n  Saved: {ROB}/")
    return rob

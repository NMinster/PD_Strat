"""
§7–8 — MSI_U unsupervised severity index and GMM-based subtyping.
"""

from __future__ import annotations

import re
import json
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score

from .config import (
    TAB, LOCKED, SEED,
    PROT_COMPLETENESS_THRESHOLD, K_SELECTION_METHOD, CLUSTER_ORDER_METHOD,
)
from .utils import (
    map_participant_id, spearman_np, eta2, flow_record, summary_update,
)
from .features import svd_n_components
from .data_loading import demographics_for


def _has_valid_prot(z_prot, M_prot, positions, min_comp=PROT_COMPLETENESS_THRESHOLD):
    """Boolean mask: which positions have sufficient proteomics data."""
    if z_prot.empty or len(positions) == 0:
        return np.zeros(len(positions), dtype=bool)
    comp = z_prot.iloc[positions].notna().mean(axis=1).values
    has = M_prot[positions].sum(axis=1) > 0
    return (comp >= min_comp) & has


def select_one_row_per_participant(clin, z_prot, M_prot, positions,
                                    prefer_visit="M0",
                                    require_prot=True,
                                    min_comp=PROT_COMPLETENESS_THRESHOLD):
    """Select one row per participant (earliest visit with proteomics)."""
    if len(positions) == 0:
        return positions

    pids = np.array([str(x) for x in clin.index[positions]])
    vcs  = clin["visit_code"].iloc[positions].values
    prot_ok = (_has_valid_prot(z_prot, M_prot, positions, min_comp)
               if require_prot else np.ones(len(positions), dtype=bool))

    df = pd.DataFrame({"pos": positions, "pid": pids, "vc": vcs,
                        "prot_ok": prot_ok})
    df["vc_str"] = df["vc"].fillna("").astype(str)

    def _month(vc):
        if pd.isna(vc) or str(vc).strip() == "":
            return 9999
        m_obj = re.match(r"M(\d+)", str(vc))
        return int(m_obj.group(1)) if m_obj else 9999

    df["month"]   = df["vc"].apply(_month)
    df["is_pref"] = (df["vc_str"] == prefer_visit).astype(int)
    n_before = df["pid"].nunique()

    df_prot = df[df["prot_ok"]].copy()
    df_prot = (df_prot
               .sort_values(["pid", "is_pref", "month"],
                            ascending=[True, False, True])
               .drop_duplicates("pid", keep="first"))

    n_after = len(df_prot)
    n_m0 = int((df_prot["vc_str"] == prefer_visit).sum())
    print(f"  [Unique] {n_after} participants from {len(positions)} rows "
          f"({n_m0} at {prefer_visit}, "
          f"{n_before - n_after} dropped)")
    return df_prot["pos"].values


def _safe_gmm(Z, K, seed, reg0=1e-3):
    for ct in ("full", "tied", "diag"):
        try:
            g = GaussianMixture(n_components=K, covariance_type=ct,
                                reg_covar=reg0, random_state=seed, n_init=3)
            g.fit(Z)
            return g, "gmm"
        except Exception:
            continue
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=K, n_init=10, random_state=seed)
    km.fit(Z)
    return km, "kmeans"


def run_msi_u_and_subtyping(clin, z_prot, X_prot, M_prot,
                             y_all, upsit_all,
                             is_pd_train, is_pd_test):
    """Run unsupervised MSI_U index and GMM subtyping.

    Returns dict with MSI_U and subtyping results.
    """
    print(f"\n{'=' * 60}")
    print("MSI_U: UNSUPERVISED SEVERITY INDEX (SVD of z_prot, TRAIN-fit)")
    print(f"{'=' * 60}")

    pd_tr_all = np.where(is_pd_train.values)[0]
    pd_tr_uniq = select_one_row_per_participant(
        clin, z_prot, M_prot, pd_tr_all,
        LOCKED["subtype_prefer_visit"], True, PROT_COMPLETENESS_THRESHOLD)
    y_msi_tr = y_all[pd_tr_uniq]
    msi_tr_ok = np.isfinite(y_msi_tr)
    msi_train_pos = pd_tr_uniq[msi_tr_ok]
    y_msi_train = y_msi_tr[msi_tr_ok]
    demo_msi_train = demographics_for(clin, msi_train_pos)

    pd_te_all = np.where(is_pd_test.values)[0]
    pd_te_uniq = select_one_row_per_participant(
        clin, z_prot, M_prot, pd_te_all,
        LOCKED["subtype_prefer_visit"], True, PROT_COMPLETENESS_THRESHOLD)
    y_msi_te = y_all[pd_te_uniq]
    msi_te_ok = np.isfinite(y_msi_te)
    msi_test_pos = pd_te_uniq[msi_te_ok]
    y_msi_test = y_msi_te[msi_te_ok]
    demo_msi_test = demographics_for(clin, msi_test_pos)

    pids_msi_train = clin.index[msi_train_pos]
    pids_msi_test  = clin.index[msi_test_pos]
    print(f"  MSI_U  TRAIN n={len(pids_msi_train)},  TEST n={len(pids_msi_test)}")

    MSI_U_NC = LOCKED["msi_u_n_components"]
    X_msi_tr = X_prot[msi_train_pos]
    X_msi_te = X_prot[msi_test_pos]

    scaler_msi = StandardScaler(with_mean=True, with_std=True)
    X_msi_tr_s = scaler_msi.fit_transform(X_msi_tr)
    X_msi_te_s = scaler_msi.transform(X_msi_te)

    nc_msi = svd_n_components(X_msi_tr_s, MSI_U_NC)
    svd_msi = TruncatedSVD(n_components=nc_msi, random_state=SEED)
    Z_msi_tr = svd_msi.fit_transform(X_msi_tr_s)
    Z_msi_te = svd_msi.transform(X_msi_te_s)

    msi_u_train = Z_msi_tr[:, 0].copy()
    msi_u_test  = Z_msi_te[:, 0].copy()

    flip = False
    rho_raw = spearman_np(msi_u_train, y_msi_train)
    if rho_raw < 0:
        msi_u_train *= -1
        msi_u_test  *= -1
        flip = True
        rho_raw = -rho_raw

    rho_msi_u_train = rho_raw
    rho_msi_u_test  = spearman_np(msi_u_test, y_msi_test)
    var_pc1 = float(svd_msi.explained_variance_ratio_[0])

    print(f"  PC1 variance: {var_pc1*100:.1f}%")
    print(f"  MSI_U~UPDRS TRAIN rho={rho_msi_u_train:.3f}")
    print(f"  MSI_U~UPDRS TEST  rho={rho_msi_u_test:.3f}")

    for tag, pos, msi, y_, demo, fname in [
        ("TRAIN", msi_train_pos, msi_u_train, y_msi_train,
         demo_msi_train, "msi_U_TRAIN.csv"),
        ("TEST", msi_test_pos, msi_u_test, y_msi_test,
         demo_msi_test, "msi_U_TEST.csv"),
    ]:
        out = pd.DataFrame({
            "participant_id": demo["participant_id"].values,
            "MSI_U": msi, "UPDRS": y_,
            "UPSIT": upsit_all[pos],
            "sex": demo["sex"].values, "age": demo["age"].values,
            "site": demo["site"].values, "cohort": demo["cohort"].values,
        })
        out.to_csv(TAB / fname, index=False)

    json.dump({
        "preprocessing": f"StandardScaler -> TruncatedSVD(n={nc_msi}), fit on TRAIN only",
        "n_components": nc_msi, "var_explained_pc1": var_pc1,
        "total_var_explained": float(svd_msi.explained_variance_ratio_.sum()),
        "n_train": len(pids_msi_train), "n_test": len(pids_msi_test),
        "sign_flip": flip,
        "rho_train": float(rho_msi_u_train), "rho_test": float(rho_msi_u_test),
    }, open(TAB / "msi_U_model.json", "w"), indent=2)

    summary_update({
        "msi_u_rho_train": float(rho_msi_u_train),
        "msi_u_rho_test": float(rho_msi_u_test),
        "msi_u_var_pc1": var_pc1,
        "msi_u_n_train": len(pids_msi_train),
        "msi_u_n_test": len(pids_msi_test),
    })
    flow_record("07_MSI_U", int(len(pids_msi_train)), int(len(pids_msi_test)))

    # ── SUBTYPING ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("SUBTYPING (unsupervised GMM on SVD latent)")
    print(f"{'=' * 60}")

    Z_sub = Z_msi_tr.copy()
    y_sub = y_msi_train.copy()
    upsit_sub = upsit_all[msi_train_pos]
    sub_demo = demo_msi_train.copy()

    ss = StandardScaler(with_mean=True, with_std=True)
    Zs = ss.fit_transform(Z_sub)
    print(f"  n={len(pids_msi_train)} unique TRAIN-PD, dim={Zs.shape[1]}")
    flow_record("08_subtyping", int(len(pids_msi_train)), 0)

    # K selection
    print(f"\n  K SELECTION (method={K_SELECTION_METHOD})")
    rng = np.random.default_rng(SEED)
    rows_k = []
    for K_try in LOCKED["k_range"]:
        if Zs.shape[0] < max(60, 8 * K_try):
            rows_k.append({"K": K_try, "silhouette": np.nan,
                            "AMI_boot_mean": np.nan, "BIC": np.nan,
                            "eta2_UPDRS": np.nan, "eta2_UPSIT": np.nan})
            continue
        gm, _ = _safe_gmm(Zs, K_try, SEED)
        labs = gm.predict(Zs)
        sil = silhouette_score(Zs, labs) if len(np.unique(labs)) > 1 else np.nan
        bic = gm.bic(Zs) if hasattr(gm, "bic") else np.nan

        amis = []
        for _ in range(LOCKED["boot_B"]):
            bs = max(2, int(LOCKED["boot_frac"] * Zs.shape[0]))
            idx_b = rng.integers(0, Zs.shape[0], size=bs)
            gm_b, _ = _safe_gmm(Zs[idx_b], K_try, int(rng.integers(1e9)))
            amis.append(adjusted_mutual_info_score(labs[idx_b], gm_b.predict(Zs[idx_b])))

        rows_k.append({
            "K": K_try, "silhouette": float(sil),
            "AMI_boot_mean": float(np.nanmean(amis)),
            "AMI_boot_std": float(np.nanstd(amis)),
            "BIC": float(bic) if np.isfinite(bic) else np.nan,
            "eta2_UPDRS": float(eta2(labs, y_sub)),
            "eta2_UPSIT": float(eta2(labs, upsit_sub)),
            "counts": pd.Series(labs).value_counts().sort_index().to_dict(),
        })

    sub_df = pd.DataFrame(rows_k)
    sub_df.to_csv(TAB / "subtype_grid_TRAINPD.csv", index=False)

    valid_k = ((sub_df["silhouette"] > 0)
               & (sub_df["AMI_boot_mean"] >= LOCKED["ami_threshold"]))
    print(f"  Valid K: {sub_df.loc[valid_k, 'K'].tolist()}")

    if valid_k.any():
        sel = sub_df.loc[valid_k]
        if K_SELECTION_METHOD == "bic" and sel["BIC"].notna().any():
            best_idx = sel["BIC"].idxmin()
            sel_method = "min BIC among valid"
        else:
            best_idx = sel["silhouette"].idxmax()
            sel_method = "max silhouette"
        best_row = sub_df.loc[best_idx]
    else:
        relaxed = (sub_df["silhouette"] > 0) & (sub_df["AMI_boot_mean"] > 0.5)
        if relaxed.any():
            best_idx = (sub_df.loc[relaxed, "BIC"].idxmin()
                        if sub_df.loc[relaxed, "BIC"].notna().any()
                        else sub_df.loc[relaxed, "silhouette"].idxmax())
            best_row = sub_df.loc[best_idx]
            sel_method = "relaxed (AMI>0.5)"
        else:
            best_row = sub_df.iloc[0]
            sel_method = "fallback K=2"

    K = int(best_row["K"])
    print(f"  >>> K={K} via {sel_method}")

    model_k, _ = _safe_gmm(Zs, K, SEED)
    labs_raw = model_k.predict(Zs)

    def _order_clusters(Z, labels, K_):
        pca = PCA(n_components=min(5, Z.shape[1]), random_state=SEED)
        Z_pca = pca.fit_transform(Z)
        cpc1 = {k: float(Z_pca[labels == k, 0].mean()) for k in range(K_)}
        order = sorted(cpc1, key=cpc1.get)
        return order

    order = _order_clusters(Zs, labs_raw, K)
    remap = {old: new for new, old in enumerate(order)}
    labs_trpd = np.array([remap[k] for k in labs_raw], dtype=int)

    assign_df = pd.DataFrame(index=clin.index)
    assign_df[f"subtype_K{K}"] = np.nan
    assign_df.iloc[msi_train_pos,
                   assign_df.columns.get_loc(f"subtype_K{K}")] = labs_trpd
    assign_df.to_csv(TAB / f"subtypes_TRAINPD_K{K}_assignments.csv")

    eta_updrs = eta2(labs_trpd, y_sub)
    eta_upsit = eta2(labs_trpd, upsit_sub)
    print(f"  eta2(UPDRS)={eta_updrs:.3f}, eta2(UPSIT)={eta_upsit:.3f}")

    rows_s = []
    for k_ in range(K):
        mk = labs_trpd == k_
        rows_s.append({
            "cluster": k_, "n": int(mk.sum()),
            "UPDRS_mean": float(np.nanmean(y_sub[mk])),
            "UPDRS_std": float(np.nanstd(y_sub[mk])),
            "UPSIT_mean": float(np.nanmean(upsit_sub[mk]))
                          if np.isfinite(upsit_sub[mk]).any() else np.nan,
        })
    cluster_summary = pd.DataFrame(rows_s)
    cluster_summary.to_csv(TAB / f"subtypes_TRAINPD_K{K}_summary.csv", index=False)
    print(cluster_summary.to_string(index=False))

    summary_update({"subtype_K": K, "subtype_method": sel_method,
                    "eta2_updrs": eta_updrs, "eta2_upsit": eta_upsit})

    # Expose cleaned IDs and labels for figures / downstream
    pd_ids_clean = pd.Index(
        clin.index[msi_train_pos].astype(str))

    return {
        "K": K, "sel_method": sel_method, "best_row": best_row,
        "labs_trpd": labs_trpd,
        "msi_train_pos": msi_train_pos, "msi_test_pos": msi_test_pos,
        "msi_u_train": msi_u_train, "msi_u_test": msi_u_test,
        "y_msi_train": y_msi_train, "y_msi_test": y_msi_test,
        "demo_msi_train": demo_msi_train, "demo_msi_test": demo_msi_test,
        "rho_msi_u_train": rho_msi_u_train, "rho_msi_u_test": rho_msi_u_test,
        "var_pc1": var_pc1,
        "eta_updrs": eta_updrs, "eta_upsit": eta_upsit,
        "Zs": Zs,  # standardized latent (for confounding/robustness)
        "pd_ids_clean": pd_ids_clean,
        "y_pd_clean": y_msi_train,
    }

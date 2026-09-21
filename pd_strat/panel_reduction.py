"""
§11 — Reduced-panel analyses without test-set peeking.

  * Bootstrap stability selection (B resamples of 80 % of TRAIN participants):
    back-projected RidgeSVD protein weights, inclusion frequency in the top
    k = 20 / 50 / 100, sign consistency.  Stable set = frequency >= 0.8 at k = 50.
  * Cumulative-importance curve, *nested*: inside every CV fold the proteins
    are ranked from the fold's own training partition, so the OOF ρ(k) curve
    is leakage-free.  The panel size k* is chosen from the OOF curve by a
    pre-specified rule (smallest k reaching 95 % of the maximal OOF ρ); the
    TEST curve is reported descriptively and TEST at k* is the confirmatory
    number.
  * TEST performance of the stability-selected set.

All reduced models use the monolithic StandardScaler -> TruncatedSVD ->
RidgeCV representation so that weights back-project to individual proteins.
"""

from __future__ import annotations

from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

from .config import (
    TAB, ROB, RIDGE_ALPHAS, Y_LO, Y_HI, N_SVD, SEED, STABILITY_B,
    CUMULATIVE_K_GRID, LOCKED,
)
from .utils import map_participant_id, spearman_np, full_metrics, summary_update
from .features import svd_n_components


def _fit_weights(X, pos, y, n_svd=N_SVD, seed=SEED):
    """Fit scaler+SVD+ridge on rows `pos`; return (protein weights, model tuple)."""
    sc = StandardScaler(with_mean=False).fit(X[pos])
    Xs = sc.transform(X[pos])
    nc = svd_n_components(Xs, n_svd)
    svd = TruncatedSVD(n_components=nc, random_state=seed).fit(Xs)
    r = RidgeCV(alphas=RIDGE_ALPHAS).fit(svd.transform(Xs), y[pos])
    scale = np.where(sc.scale_ > 0, sc.scale_, 1.0)
    w = (svd.components_.T @ r.coef_) / scale
    return w, (sc, svd, r)


def _fit_predict_cols(X, cols, tr, te, y, seed=SEED):
    Xc = X[:, cols]
    sc = StandardScaler(with_mean=False).fit(Xc[tr])
    nc = svd_n_components(sc.transform(Xc[tr]), min(N_SVD, len(cols)))
    svd = TruncatedSVD(n_components=nc, random_state=seed).fit(sc.transform(Xc[tr]))
    r = RidgeCV(alphas=RIDGE_ALPHAS).fit(svd.transform(sc.transform(Xc[tr])), y[tr])
    return np.clip(r.predict(svd.transform(sc.transform(Xc[te]))), Y_LO, Y_HI)


def run_panel_reduction(clin, X_prot, y_all, train_idx_y, groups_train, gkf,
                        test_idx_omics, y_te_full, prot_cols, PANEL_COL_INDICES,
                        B: int = STABILITY_B) -> Dict[str, Any]:
    print(f"\n{'=' * 60}")
    print(f"PANEL REDUCTION: stability selection (B={B}) + nested cumulative curve")
    print(f"{'=' * 60}")
    out: Dict[str, Any] = {}
    d = X_prot.shape[1]
    ytr = y_all[train_idx_y]
    prot_to_panel = {}
    for pn, ci in PANEL_COL_INDICES.items():
        for c in ci:
            prot_to_panel[prot_cols[c]] = pn

    # ── stability selection ────────────────────────────────────────────
    w_full, _ = _fit_weights(X_prot, train_idx_y, y_all)
    pids = np.array([map_participant_id(str(x)) for x in clin.index[train_idx_y]])
    upid = np.unique(pids)
    rng = np.random.default_rng(SEED + 555)
    W = np.zeros((B, d))
    for b in range(B):
        keep = rng.choice(upid, size=max(5, int(LOCKED["boot_frac"] * len(upid))), replace=False)
        pos = train_idx_y[np.isin(pids, keep)]
        W[b], _ = _fit_weights(X_prot, pos, y_all, seed=SEED + b)
    ranks = np.argsort(np.argsort(-np.abs(W), axis=1), axis=1)      # 0 = most important
    freq = {k: (ranks < k).mean(axis=0) for k in (20, 50, 100)}
    w_med = np.median(W, axis=0)
    sign_cons = (np.sign(W) == np.sign(w_med)[None, :]).mean(axis=0)
    stab = pd.DataFrame({
        "protein": prot_cols, "panel": [prot_to_panel.get(p, "unknown") for p in prot_cols],
        "w_full": w_full, "w_boot_median": w_med, "w_boot_sd": W.std(axis=0),
        "sign_consistency": sign_cons,
        "incl_freq_k20": freq[20], "incl_freq_k50": freq[50], "incl_freq_k100": freq[100],
        "rank_full": np.argsort(np.argsort(-np.abs(w_full))) + 1,
        "median_boot_rank": np.median(ranks, axis=0) + 1,
    })
    stab["stable_importance"] = stab["sign_consistency"] * stab["w_boot_median"].abs()
    stab = stab.sort_values("stable_importance", ascending=False)
    stab.to_csv(ROB / "stability_selection.csv", index=False)
    stable_set = stab[stab["incl_freq_k50"] >= 0.8]["protein"].tolist()
    out["n_stable_k50_freq80"] = len(stable_set)
    out["stable_set"] = stable_set
    out["mean_sign_consistency_all"] = float(sign_cons.mean())
    out["mean_sign_consistency_top20"] = float(stab.head(20)["sign_consistency"].mean())
    out["mean_sign_consistency_top40"] = float(stab.head(40)["sign_consistency"].mean())
    out["top10_stable"] = stab.head(10)["protein"].tolist()
    print(f"  Stable proteins (freq>=0.8 at k=50): {len(stable_set)}; "
          f"sign consistency all/top20/top40 = {out['mean_sign_consistency_all']:.3f}/"
          f"{out['mean_sign_consistency_top20']:.3f}/{out['mean_sign_consistency_top40']:.3f}")

    # ── nested cumulative-importance curve (OOF) ────────────────────────
    grid = [k for k in CUMULATIVE_K_GRID if k < d] + [d]
    oof_k = {k: np.full(len(train_idx_y), np.nan) for k in grid}
    for tr, va in gkf.split(train_idx_y, ytr, groups_train):
        fp, vp = train_idx_y[tr], train_idx_y[va]
        w_f, _ = _fit_weights(X_prot, fp, y_all)
        order = np.argsort(-np.abs(w_f))
        for k in grid:
            oof_k[k][va] = _fit_predict_cols(X_prot, np.sort(order[:k]), fp, vp, y_all)
    rows = []
    order_full = np.argsort(-np.abs(w_full))
    have_test = len(test_idx_omics) >= 10 and np.isfinite(y_te_full).sum() >= 10
    for k in grid:
        r = {"k": k, "oof_rho": spearman_np(oof_k[k], ytr),
             "oof_mae": full_metrics(oof_k[k], ytr)["mae"]}
        if have_test:
            pte = _fit_predict_cols(X_prot, np.sort(order_full[:k]), train_idx_y,
                                    test_idx_omics, y_all)
            r["test_rho"] = spearman_np(pte, y_te_full)
            r["test_mae"] = full_metrics(pte, y_te_full)["mae"]
        rows.append(r)
    curve = pd.DataFrame(rows)
    curve.to_csv(ROB / "cumulative_importance.csv", index=False)
    max_oof = curve["oof_rho"].max()
    k_star = int(curve.loc[curve["oof_rho"] >= 0.95 * max_oof, "k"].min())
    out["k_star_rule"] = "smallest k with OOF rho >= 0.95 * max OOF rho (chosen on TRAIN only)"
    out["k_star"] = k_star
    out["oof_rho_at_k_star"] = float(curve.loc[curve["k"] == k_star, "oof_rho"].iloc[0])
    out["oof_rho_full"] = float(curve.loc[curve["k"] == d, "oof_rho"].iloc[0])
    if have_test:
        out["test_rho_at_k_star"] = float(curve.loc[curve["k"] == k_star, "test_rho"].iloc[0])
        out["test_rho_full"] = float(curve.loc[curve["k"] == d, "test_rho"].iloc[0])
    print("  Nested cumulative curve (k: OOF rho / TEST rho):")
    for _, r in curve.iterrows():
        print(f"    k={int(r['k']):>5}: {r['oof_rho']:.3f} / {r.get('test_rho', np.nan):.3f}")
    print(f"  k* = {k_star} (OOF rule); TEST rho at k* = {out.get('test_rho_at_k_star', np.nan):.3f}"
          f" vs full panel {out.get('test_rho_full', np.nan):.3f}")
    out["k_star_proteins"] = [prot_cols[i] for i in order_full[:k_star]]

    # ── stability-selected set on TEST ──────────────────────────────────
    if have_test and len(stable_set) >= 3:
        cols = np.array([prot_cols.index(p) for p in stable_set])
        pte = _fit_predict_cols(X_prot, np.sort(cols), train_idx_y, test_idx_omics, y_all)
        out["stable_set_test"] = full_metrics(pte, y_te_full)
        print(f"  Stable set (n={len(stable_set)}) TEST rho={out['stable_set_test']['spearman']:.3f}")
    pd.DataFrame({"protein": out["k_star_proteins"],
                  "panel": [prot_to_panel.get(p, "unknown") for p in out["k_star_proteins"]]}
                 ).to_csv(ROB / "reduced_panel_k_star.csv", index=False)
    summary_update({"panel_reduction": {k: v for k, v in out.items()
                                        if k not in ("stable_set", "k_star_proteins")}})
    return out

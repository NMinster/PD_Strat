"""
§6e — Validity package for the severity model.

Answers the questions a referee will ask of a "severity" model trained in a
cohort that also contains healthy controls:

  (a) HC-vs-PD discrimination of the severity score, and rank correlation
      *within* PD and *within* HC (is it severity, or diagnosis?).
  (b) PD-only refit: the whole model re-trained and evaluated on PD cases.
  (c) Extended covariate adjustment: age, sex, site, disease duration and
      Part-III medication state (covariates-only / combined / partial ρ),
      plus strata analyses.
  (d) Cross-endpoint validation: UPDRS I-IV, UPSIT, Hoehn & Yahr, DaTSCAN.
  (e) Error stratification, heteroscedasticity, range-restricted ρ.
  (f) Recalibration *in the target cohort*, cross-fitted (participant-grouped)
      so it is not optimistically evaluated on the same rows it was fit on.
  (g) Paired participant-level bootstrap comparing candidate models.
"""

from __future__ import annotations

import json
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from .config import (
    TAB, LOCKED, RIDGE_ALPHAS, Y_LO, Y_HI, SEED,
    PAIRED_BOOT_B, PLR_BOOT_B,
)
from .utils import (
    map_participant_id, full_metrics, spearman_np, summary_update, flow_record,
)
from .features import primary_feature_builder
from .calibration import participant_level_metrics, participant_level_bootstrap_ci


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  helpers                                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _pids(clin, positions) -> np.ndarray:
    return np.array([map_participant_id(str(x)) for x in clin.index[positions]])


def _agg_participant(clin, positions, pred, y, extra: Optional[Dict[str, np.ndarray]] = None
                     ) -> pd.DataFrame:
    df = pd.DataFrame({"pid": _pids(clin, positions), "pred": pred, "y": y})
    for k, v in (extra or {}).items():
        df[k] = v
    df = df.dropna(subset=["pred", "y"])
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    agg = df.groupby("pid")[num].mean()
    non = [c for c in df.columns if c not in num and c != "pid"]
    if non:
        agg = agg.join(df.groupby("pid")[non].agg(lambda s: s.mode().iloc[0]
                                                  if len(s.mode()) else np.nan))
    return agg.reset_index()


def _fisher_ci(rho: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if not np.isfinite(rho) or n < 4 or abs(rho) >= 1:
        return np.nan, np.nan
    f = np.arctanh(rho)
    se = 1.0 / np.sqrt(n - 3)
    return float(np.tanh(f - z * se)), float(np.tanh(f + z * se))


def _oof_predict(bff, X, M, y_all, pos, groups, n_splits=None):
    """Participant-grouped OOF ridge predictions for rows `pos`."""
    ns = max(2, min(LOCKED["n_cv_folds"], len(pd.unique(groups))))
    gkf = GroupKFold(n_splits=n_splits or ns)
    oof = np.full(len(pos), np.nan, dtype=np.float32)
    for tr, va in gkf.split(pos, y_all[pos], groups):
        fp, vp = pos[tr], pos[va]
        Z = bff(X, M, fp)
        r = RidgeCV(alphas=RIDGE_ALPHAS).fit(Z[fp], y_all[fp])
        oof[va] = np.clip(r.predict(Z[vp]), Y_LO, Y_HI)
    return oof


def _fit_predict(bff, X, M, y_all, train_pos, pred_pos):
    Z = bff(X, M, train_pos)
    r = RidgeCV(alphas=RIDGE_ALPHAS).fit(Z[train_pos], y_all[train_pos])
    return np.clip(r.predict(Z[pred_pos]), Y_LO, Y_HI).astype(np.float32)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  (a) severity vs diagnosis                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _severity_vs_diagnosis(clin, y_all, splits, is_pd, is_hc) -> Dict[str, Any]:
    print("\n  --- (a) Is the score severity or diagnosis? ---")
    out: Dict[str, Any] = {}
    for tag, pos, pred in splits:
        y = y_all[pos]
        m = np.isfinite(pred) & np.isfinite(y)
        pdm, hcm = is_pd[pos] & m, is_hc[pos] & m
        row: Dict[str, Any] = {
            "n_rows": int(m.sum()), "n_pd_rows": int(pdm.sum()),
            "n_hc_rows": int(hcm.sum()),
            "rho_all": spearman_np(pred, y),
            "rho_within_pd": spearman_np(pred[pdm], y[pdm]) if pdm.sum() >= 10 else np.nan,
            "rho_within_hc": spearman_np(pred[hcm], y[hcm]) if hcm.sum() >= 10 else np.nan,
        }
        agg = _agg_participant(clin, pos, pred, y,
                               {"is_pd": is_pd[pos].astype(float),
                                "is_hc": is_hc[pos].astype(float)})
        agg_pd = agg[agg["is_pd"] >= 0.5]
        row["n_pd_participants"] = int(len(agg_pd))
        row["rho_within_pd_participant"] = (
            spearman_np(agg_pd["pred"].values, agg_pd["y"].values)
            if len(agg_pd) >= 10 else np.nan)
        if pdm.sum() >= 10 and hcm.sum() >= 10:
            lab = is_pd[pos][m].astype(int)
            row["auroc_pd_vs_hc_rows"] = float(roc_auc_score(lab, pred[m]))
            both = agg[(agg["is_pd"] >= 0.5) | (agg["is_hc"] >= 0.5)]
            row["auroc_pd_vs_hc_participant"] = float(
                roc_auc_score((both["is_pd"] >= 0.5).astype(int), both["pred"]))
            row["mean_pred_hc"] = float(pred[hcm].mean())
            row["mean_pred_pd"] = float(pred[pdm].mean())
        else:
            row["auroc_pd_vs_hc_rows"] = np.nan
            row["auroc_pd_vs_hc_participant"] = np.nan
        out[tag] = row
        print(f"    {tag}: rho_all={row['rho_all']:.3f}  "
              f"rho_within_PD={row['rho_within_pd']:.3f} "
              f"(participant {row['rho_within_pd_participant']:.3f}, n={row['n_pd_participants']})  "
              f"rho_within_HC={row['rho_within_hc']:.3f}  "
              f"AUROC(PD vs HC)={row['auroc_pd_vs_hc_participant']:.3f}")
    pd.DataFrame(out).T.to_csv(TAB / "validity_severity_vs_diagnosis.csv")
    return out


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  (b) refit on the complementary population                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _population_refit(clin, X, M, y_all, bff, groups_all,
                      train_idx_y, test_idx_omics, keep_mask, label) -> Dict[str, Any]:
    print(f"\n  --- (b) Model refit on population = {label} ---")
    out: Dict[str, Any] = {"population": label}
    tr = train_idx_y[keep_mask[train_idx_y]]
    te = test_idx_omics[keep_mask[test_idx_omics]]
    n_pid = len(pd.unique(groups_all[tr]))
    out["n_train_rows"], out["n_train_participants"] = int(len(tr)), int(n_pid)
    out["n_test_rows"] = int(len(te))
    if len(tr) < 30 or n_pid < 10:
        print(f"    [SKIP] too few rows ({len(tr)}) / participants ({n_pid})")
        return out

    oof = _oof_predict(bff, X, M, y_all, tr, groups_all[tr])
    out["OOF_row"] = full_metrics(oof, y_all[tr])
    plr = participant_level_metrics(clin, tr, oof, y_all[tr], f"{label}_OOF")
    out["OOF_participant"] = plr.get(f"{label}_OOF_participant_mean", {})
    out["OOF_participant_ci"] = participant_level_bootstrap_ci(
        clin, tr, oof, y_all[tr], n_boot=PLR_BOOT_B)
    print(f"    OOF  rows={len(tr)} rho={out['OOF_row']['spearman']:.3f}  "
          f"participant rho={out['OOF_participant'].get('spearman', np.nan):.3f} "
          f"[{out['OOF_participant_ci']['spearman']['lo']:.3f}, "
          f"{out['OOF_participant_ci']['spearman']['hi']:.3f}]")

    pred_rows = pd.DataFrame({"participant_id": clin.index[tr], "split": "TRAIN_OOF",
                              "pred": oof, "y": y_all[tr]})
    if len(te) >= 10 and np.isfinite(y_all[te]).sum() >= 10:
        pte = _fit_predict(bff, X, M, y_all, tr, te)
        out["TEST_row"] = full_metrics(pte, y_all[te])
        plr_t = participant_level_metrics(clin, te, pte, y_all[te], f"{label}_TEST")
        out["TEST_participant"] = plr_t.get(f"{label}_TEST_participant_mean", {})
        out["TEST_participant_ci"] = participant_level_bootstrap_ci(
            clin, te, pte, y_all[te], n_boot=PLR_BOOT_B)
        print(f"    TEST rows={len(te)} rho={out['TEST_row']['spearman']:.3f}  "
              f"participant rho={out['TEST_participant'].get('spearman', np.nan):.3f} "
              f"[{out['TEST_participant_ci']['spearman']['lo']:.3f}, "
              f"{out['TEST_participant_ci']['spearman']['hi']:.3f}]")
        pred_rows = pd.concat([pred_rows, pd.DataFrame({
            "participant_id": clin.index[te], "split": "TEST", "pred": pte,
            "y": y_all[te]})])
    pred_rows.to_csv(TAB / f"predictions_{label}.csv", index=False)
    return out


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  (c) extended covariates                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

_NUMERIC_COVS = [("age", "_age"), ("disease_duration", "disease_duration_years"),
                 ("ledd", "ledd")]
_CATEG_COVS = [("sex", "_sex"), ("site", "_site"), ("med_state", "updrs3_state"),
               ("levodopa", "on_levodopa")]


def _design(clin, positions, ref_cols=None, fill=None):
    parts, names = [], []
    fill = dict(fill or {})
    for nm, col in _NUMERIC_COVS:
        if col not in clin.columns:
            continue
        v = pd.to_numeric(clin[col].iloc[positions], errors="coerce").values.astype(float)
        if np.isfinite(v).sum() < 5:
            continue
        mu = fill.get(nm, float(np.nanmean(v)))
        fill[nm] = mu
        v = np.where(np.isfinite(v), v, mu)
        parts.append(v.reshape(-1, 1)); names.append(nm)
    for nm, col in _CATEG_COVS:
        if col not in clin.columns:
            continue
        s = clin[col].iloc[positions].astype(str).replace({"nan": "UNKNOWN", "": "UNKNOWN"})
        if s.nunique() < 2:
            continue
        d = pd.get_dummies(s, prefix=nm, drop_first=True).astype(float)
        parts.append(d.values); names.extend(d.columns.tolist())
    if not parts:
        return None, [], fill
    Xc = np.hstack(parts)
    if ref_cols is not None:
        Xc = pd.DataFrame(Xc, columns=names).reindex(columns=ref_cols, fill_value=0.0).values
        names = list(ref_cols)
    return Xc.astype(np.float64), names, fill


def _partial_rho(pred, y, Xc):
    m = np.isfinite(pred) & np.isfinite(y)
    if Xc is None or m.sum() < 20:
        return np.nan
    rp = pred[m] - LinearRegression().fit(Xc[m], pred[m]).predict(Xc[m])
    ry = y[m] - LinearRegression().fit(Xc[m], y[m]).predict(Xc[m])
    return spearman_np(rp, ry)


def _extended_covariates(clin, X, M, y_all, bff, train_idx_y, groups_train, gkf,
                         test_idx_omics, oof_pred, test_pred_omics) -> Dict[str, Any]:
    print("\n  --- (c) Extended covariate adjustment ---")
    out: Dict[str, Any] = {}
    Xc_tr, names, fill = _design(clin, train_idx_y)
    if Xc_tr is None:
        print("    [SKIP] no usable covariates")
        return out
    out["covariates"] = names
    print(f"    covariates: {names}")

    ytr = y_all[train_idx_y]
    oof_cov = np.full(len(train_idx_y), np.nan, dtype=np.float32)
    oof_comb = np.full(len(train_idx_y), np.nan, dtype=np.float32)
    for tr, va in gkf.split(train_idx_y, ytr, groups_train):
        fp, vp = train_idx_y[tr], train_idx_y[va]
        r = RidgeCV(alphas=RIDGE_ALPHAS).fit(Xc_tr[tr], ytr[tr])
        oof_cov[va] = np.clip(r.predict(Xc_tr[va]), Y_LO, Y_HI)
        Z = bff(X, M, fp)
        r2 = RidgeCV(alphas=RIDGE_ALPHAS).fit(
            np.hstack([Z[fp], Xc_tr[tr]]), ytr[tr])
        oof_comb[va] = np.clip(r2.predict(np.hstack([Z[vp], Xc_tr[va]])), Y_LO, Y_HI)
    out["OOF"] = {
        "covariates_only": full_metrics(oof_cov, ytr),
        "proteomics_only": full_metrics(oof_pred, ytr),
        "combined": full_metrics(oof_comb, ytr),
        "partial_rho_prot_given_cov": _partial_rho(oof_pred, ytr, Xc_tr),
    }
    print(f"    OOF  cov-only rho={out['OOF']['covariates_only']['spearman']:.3f}  "
          f"prot rho={out['OOF']['proteomics_only']['spearman']:.3f}  "
          f"combined rho={out['OOF']['combined']['spearman']:.3f}  "
          f"partial rho(prot|cov)={out['OOF']['partial_rho_prot_given_cov']:.3f}")

    if len(test_idx_omics) >= 20:
        Xc_te, _, _ = _design(clin, test_idx_omics, ref_cols=names, fill=fill)
        yte = y_all[test_idx_omics]
        r_cov = RidgeCV(alphas=RIDGE_ALPHAS).fit(Xc_tr, ytr)
        pred_cov_te = np.clip(r_cov.predict(Xc_te), Y_LO, Y_HI)
        Zf = bff(X, M, train_idx_y)
        r_comb = RidgeCV(alphas=RIDGE_ALPHAS).fit(
            np.hstack([Zf[train_idx_y], Xc_tr]), ytr)
        pred_comb_te = np.clip(r_comb.predict(
            np.hstack([Zf[test_idx_omics], Xc_te])), Y_LO, Y_HI)
        out["TEST"] = {
            "covariates_only": full_metrics(pred_cov_te, yte),
            "proteomics_only": full_metrics(test_pred_omics, yte),
            "combined": full_metrics(pred_comb_te, yte),
            "partial_rho_prot_given_cov": _partial_rho(test_pred_omics, yte, Xc_te),
        }
        print(f"    TEST cov-only rho={out['TEST']['covariates_only']['spearman']:.3f}  "
              f"prot rho={out['TEST']['proteomics_only']['spearman']:.3f}  "
              f"combined rho={out['TEST']['combined']['spearman']:.3f}  "
              f"partial rho(prot|cov)={out['TEST']['partial_rho_prot_given_cov']:.3f}")

    # strata: medication state and disease-duration tertiles
    strata: Dict[str, Any] = {}
    for tag, pos, pred in (("OOF", train_idx_y, oof_pred),
                           ("TEST", test_idx_omics, test_pred_omics)):
        y = y_all[pos]
        if "updrs3_state" in clin.columns:
            st = clin["updrs3_state"].iloc[pos].astype(str).values
            for s in ("OFF", "ON"):
                mm = (st == s) & np.isfinite(pred) & np.isfinite(y)
                if mm.sum() >= 20:
                    strata[f"{tag}_medstate_{s}"] = {
                        "n": int(mm.sum()), "rho": spearman_np(pred[mm], y[mm])}
        if "on_levodopa" in clin.columns:
            lv = pd.to_numeric(clin["on_levodopa"].iloc[pos], errors="coerce").values
            for lab, val in (("levodopa_yes", 1.0), ("levodopa_no", 0.0)):
                mm = (lv == val) & np.isfinite(pred) & np.isfinite(y)
                if mm.sum() >= 20:
                    strata[f"{tag}_{lab}"] = {
                        "n": int(mm.sum()), "rho": spearman_np(pred[mm], y[mm])}
        if "disease_duration_years" in clin.columns:
            dd = pd.to_numeric(clin["disease_duration_years"].iloc[pos],
                               errors="coerce").values
            ok = np.isfinite(dd) & np.isfinite(pred) & np.isfinite(y)
            if ok.sum() >= 60:
                q = np.nanpercentile(dd[ok], [33.3, 66.7])
                for lab, mm in (("short", ok & (dd <= q[0])),
                                ("mid", ok & (dd > q[0]) & (dd <= q[1])),
                                ("long", ok & (dd > q[1]))):
                    if mm.sum() >= 20:
                        strata[f"{tag}_duration_{lab}"] = {
                            "n": int(mm.sum()), "rho": spearman_np(pred[mm], y[mm]),
                            "duration_range": [float(np.nanmin(dd[mm])),
                                               float(np.nanmax(dd[mm]))]}
    if strata:
        out["strata"] = strata
        for k, v in strata.items():
            print(f"    {k}: n={v['n']} rho={v['rho']:.3f}")
    rows = []
    for split in ("OOF", "TEST"):
        for mdl, mets in out.get(split, {}).items():
            if isinstance(mets, dict):
                rows.append({"split": split, "model": mdl, **mets})
            else:
                rows.append({"split": split, "model": mdl, "spearman": mets})
    pd.DataFrame(rows).to_csv(TAB / "validity_extended_covariates.csv", index=False)
    return out


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  (d) cross-endpoint validation                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

_ENDPOINTS = [
    ("UPDRS_I", "mds_updrs_part_i_total"), ("UPDRS_II", "mds_updrs_part_ii_total"),
    ("UPDRS_III", "mds_updrs_part_iii_total"), ("UPDRS_IV", "mds_updrs_part_iv_total"),
    ("UPSIT", "upsit_total"), ("Hoehn_Yahr", "hoehn_yahr"),
    ("DaTSCAN_putamen", "datscan_putamen"), ("DaTSCAN_caudate", "datscan_caudate"),
    ("DaTSCAN_striatum", "datscan_striatum"),
]
from .config import EXTRA_BIOMARKERS as _EXTRA
_ENDPOINTS += [(str(b.get("name")), str(b.get("name"))) for b in _EXTRA if b.get("name")]


def _cross_endpoint(clin, splits, is_pd) -> pd.DataFrame:
    print("\n  --- (d) Cross-endpoint validation ---")
    rows = []
    for tag, pos, pred in splits:
        for ep, col in _ENDPOINTS:
            if col not in clin.columns:
                continue
            v = pd.to_numeric(clin[col].iloc[pos], errors="coerce").values.astype(float)
            for pop, mask in (("all", np.ones(len(pos), bool)), ("PD", is_pd[pos])):
                m = mask & np.isfinite(v) & np.isfinite(pred)
                if m.sum() < 20:
                    continue
                rho = spearman_np(pred[m], v[m])
                lo, hi = _fisher_ci(rho, int(m.sum()))
                agg = _agg_participant(clin, pos[m], pred[m], v[m])
                rho_p = spearman_np(agg["pred"].values, agg["y"].values)
                lo_p, hi_p = _fisher_ci(rho_p, len(agg))
                rows.append({"split": tag, "endpoint": ep, "population": pop,
                             "n_rows": int(m.sum()), "rho_rows": rho,
                             "ci_lo_rows": lo, "ci_hi_rows": hi,
                             "n_participants": int(len(agg)),
                             "rho_participant": rho_p,
                             "ci_lo_participant": lo_p, "ci_hi_participant": hi_p})
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(TAB / "validity_cross_endpoint.csv", index=False)
        for _, r in df[df["population"] == "PD"].iterrows():
            print(f"    {r['split']:<4} {r['endpoint']:<16} PD  n={r['n_participants']:>4}  "
                  f"rho(participant)={r['rho_participant']:+.3f} "
                  f"[{r['ci_lo_participant']:+.3f}, {r['ci_hi_participant']:+.3f}]")
    else:
        print("    [SKIP] no endpoint columns found")
    return df


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  (e) error structure                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _error_structure(clin, y_all, splits, y_train_max) -> Dict[str, Any]:
    print("\n  --- (e) Error stratification / heteroscedasticity / range ---")
    out: Dict[str, Any] = {}
    rows = []
    sev_bins = [(-1, 20, "0-20"), (20, 40, "20-40"), (40, 60, "40-60"), (60, 1e9, ">60")]
    mon_bins = [(-1, 6, "M0-5"), (6, 12, "M6-11"), (12, 24, "M12-23"),
                (24, 36, "M24-35"), (36, 1e9, "M36+")]
    months = pd.to_numeric(clin.get("visit_month"), errors="coerce").values
    for tag, pos, pred in splits:
        y = y_all[pos]
        m = np.isfinite(pred) & np.isfinite(y)
        res = pred - y
        for lo, hi, lab in sev_bins:
            mm = m & (y > lo) & (y <= hi)
            if mm.sum() >= 5:
                rows.append({"split": tag, "stratum": "severity", "bin": lab,
                             "n": int(mm.sum()), "mae": float(np.mean(np.abs(res[mm]))),
                             "bias": float(np.mean(res[mm]))})
        mo = months[pos]
        for lo, hi, lab in mon_bins:
            mm = m & np.isfinite(mo) & (mo > lo) & (mo <= hi) if lo >= 0 else \
                 m & np.isfinite(mo) & (mo <= hi)
            if mm.sum() >= 5:
                rows.append({"split": tag, "stratum": "visit_month", "bin": lab,
                             "n": int(mm.sum()), "mae": float(np.mean(np.abs(res[mm]))),
                             "bias": float(np.mean(res[mm]))})
        d: Dict[str, Any] = {}
        try:
            import statsmodels.api as sm
            from statsmodels.stats.diagnostic import het_breuschpagan
            bp = het_breuschpagan(res[m], sm.add_constant(pred[m]))
            d["breusch_pagan_LM_p"] = float(bp[1]); d["breusch_pagan_F_p"] = float(bp[3])
        except Exception:
            pass
        # range-restricted rho: TEST rows within the TRAIN observed range
        mr = m & (y <= y_train_max)
        d["rho_all"] = spearman_np(pred[m], y[m])
        d["rho_within_train_range"] = spearman_np(pred[mr], y[mr]) if mr.sum() >= 20 else np.nan
        d["n_within_train_range"] = int(mr.sum())
        d["n_above_train_range"] = int((m & (y > y_train_max)).sum())
        d["calibration_slope"] = float(np.polyfit(pred[m], y[m], 1)[0]) if m.sum() >= 5 else np.nan
        out[tag] = d
        print(f"    {tag}: rho={d['rho_all']:.3f}, within TRAIN range "
              f"(y<={y_train_max:.0f}, n={d['n_within_train_range']}) rho="
              f"{d['rho_within_train_range']:.3f}; above-range n={d['n_above_train_range']}; "
              f"BP p={d.get('breusch_pagan_F_p', np.nan):.2e}")
    pd.DataFrame(rows).to_csv(TAB / "validity_error_stratification.csv", index=False)
    out["table"] = rows
    return out


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  (f) cross-fitted recalibration in the target cohort                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _target_recalibration(clin, y_all, test_idx_omics, test_pred_omics, groups_all
                          ) -> Dict[str, Any]:
    print("\n  --- (f) Cross-fitted recalibration in TEST (target cohort) ---")
    out: Dict[str, Any] = {}
    y = y_all[test_idx_omics]
    m = np.isfinite(test_pred_omics) & np.isfinite(y)
    if m.sum() < 40:
        print("    [SKIP] too few TEST rows")
        return out
    pos = test_idx_omics[m]; p = test_pred_omics[m]; yy = y[m]
    g = groups_all[pos]
    ns = max(2, min(5, len(pd.unique(g))))
    gkf = GroupKFold(n_splits=ns)
    lin = np.full(len(p), np.nan); iso = np.full(len(p), np.nan)
    for tr, va in gkf.split(p, yy, g):
        a, b = np.polyfit(p[tr], yy[tr], 1)
        lin[va] = np.clip(a * p[va] + b, Y_LO, Y_HI)
        ir = IsotonicRegression(out_of_bounds="clip").fit(p[tr], yy[tr])
        iso[va] = ir.predict(p[va])
    for lab, q in (("raw_transport", p), ("linear_crossfit", lin), ("isotonic_crossfit", iso)):
        d = full_metrics(q, yy)
        plr = participant_level_metrics(clin, pos, q, yy, f"recal_{lab}")
        d["participant"] = plr.get(f"recal_{lab}_participant_mean", {})
        out[lab] = d
        print(f"    {lab:<18} rho={d['spearman']:.3f} MAE={d['mae']:.2f} R2={d['r2']:.3f}  "
              f"participant MAE={d['participant'].get('mae', np.nan):.2f} "
              f"R2={d['participant'].get('r2', np.nan):.3f}")
    out["note"] = ("Recalibration maps were fit within TEST using participant-grouped "
                   "5-fold cross-fitting; MAE/R2 are out-of-fold. Rank metrics are "
                   "unchanged by linear recalibration by construction.")
    pd.DataFrame({k: v for k, v in out.items() if isinstance(v, dict)}).T.drop(
        columns=["participant"], errors="ignore").to_csv(TAB / "validity_target_recalibration.csv")
    return out


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  (g) paired model comparison                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _paired_comparison(clin, y_all, primary, preds_by_model: Dict[str, np.ndarray],
                       positions, tag, B=PAIRED_BOOT_B) -> List[Dict[str, Any]]:
    rows = []
    if primary not in preds_by_model:
        return rows
    y = y_all[positions]
    base = _agg_participant(clin, positions, preds_by_model[primary], y).set_index("pid")
    rng = np.random.default_rng(SEED + 31)
    for name, pred in preds_by_model.items():
        if name == primary or pred is None or len(pred) != len(positions):
            continue
        other = _agg_participant(clin, positions, pred, y).set_index("pid")
        j = base.join(other, lsuffix="_p", rsuffix="_o", how="inner").dropna()
        if len(j) < 20:
            continue
        yp, pp, po = j["y_p"].values, j["pred_p"].values, j["pred_o"].values
        d0 = spearman_np(po, yp) - spearman_np(pp, yp)
        deltas = []
        for _ in range(B):
            ix = rng.integers(0, len(j), len(j))
            deltas.append(spearman_np(po[ix], yp[ix]) - spearman_np(pp[ix], yp[ix]))
        deltas = np.array(deltas)
        rows.append({"split": tag, "primary": primary, "comparator": name,
                     "n_participants": int(len(j)),
                     "rho_primary": spearman_np(pp, yp), "rho_comparator": spearman_np(po, yp),
                     "delta_rho": d0,
                     "delta_ci_lo": float(np.nanpercentile(deltas, 2.5)),
                     "delta_ci_hi": float(np.nanpercentile(deltas, 97.5)),
                     "p_two_sided": float(min(1.0, 2 * min(np.mean(deltas >= 0),
                                                           np.mean(deltas <= 0))))})
    return rows


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  driver                                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def run_validity(clin, z_prot, X_prot, M_prot, y_all, tv, cohort,
                 PANEL_COL_INDICES, USE_PANEL_AWARE, PRIMARY_LABEL,
                 oof_pred, test_pred, _oof_preds, _test_preds,
                 severity_population: str = "all") -> Dict[str, Any]:
    print(f"\n{'=' * 60}")
    print("VALIDITY PACKAGE (severity vs diagnosis, covariates, endpoints, errors)")
    print(f"{'=' * 60}")
    train_idx_y, test_idx_omics = tv["train_idx_y"], tv["test_idx_omics"]
    prot_ok_test, groups_all = tv["prot_ok_test"], tv["groups_all"]
    groups_train, gkf = tv["groups_train"], tv["gkf"]
    test_pred_omics = test_pred[prot_ok_test] if len(test_idx_omics) else np.array([])
    is_pd = cohort["is_pd_flag"].values.astype(bool)
    is_hc = cohort["is_control"].values.astype(bool)
    bff = primary_feature_builder(PANEL_COL_INDICES, USE_PANEL_AWARE, PRIMARY_LABEL)
    splits = [("OOF", train_idx_y, oof_pred)]
    if len(test_idx_omics) >= 10:
        splits.append(("TEST", test_idx_omics, test_pred_omics))

    res: Dict[str, Any] = {"primary_population": severity_population}
    res["severity_vs_diagnosis"] = _severity_vs_diagnosis(clin, y_all, splits, is_pd, is_hc)

    if severity_population == "all":
        res["refit_pd_only"] = _population_refit(
            clin, X_prot, M_prot, y_all, bff, groups_all,
            train_idx_y, test_idx_omics, is_pd, "pd_only")
    else:
        res["refit_all"] = _population_refit(
            clin, X_prot, M_prot, y_all, bff, groups_all,
            tv["train_idx_y_all"], tv["test_idx_omics_all"],
            np.ones(len(clin), bool), "all_population")

    res["extended_covariates"] = _extended_covariates(
        clin, X_prot, M_prot, y_all, bff, train_idx_y, groups_train, gkf,
        test_idx_omics, oof_pred, test_pred_omics)
    ce = _cross_endpoint(clin, splits, is_pd)
    res["cross_endpoint"] = ce.to_dict("records") if not ce.empty else []
    y_train_max = float(np.nanmax(y_all[train_idx_y]))
    res["error_structure"] = _error_structure(clin, y_all, splits, y_train_max)
    if len(test_idx_omics) >= 10:
        res["target_recalibration"] = _target_recalibration(
            clin, y_all, test_idx_omics, test_pred_omics, groups_all)

    print("\n  --- (g) Paired participant-level model comparison ---")
    rows = _paired_comparison(clin, y_all, PRIMARY_LABEL, _oof_preds, train_idx_y, "OOF")
    if len(test_idx_omics) >= 10:
        te_models = {k: (v[prot_ok_test] if v is not None and len(v) == len(prot_ok_test) else None)
                     for k, v in _test_preds.items()}
        rows += _paired_comparison(clin, y_all, PRIMARY_LABEL, te_models, test_idx_omics, "TEST")
    if rows:
        cmp_df = pd.DataFrame(rows)
        cmp_df.to_csv(TAB / "validity_model_comparison.csv", index=False)
        for _, r in cmp_df.iterrows():
            print(f"    {r['split']:<4} {r['comparator']:<20} delta_rho={r['delta_rho']:+.3f} "
                  f"[{r['delta_ci_lo']:+.3f}, {r['delta_ci_hi']:+.3f}] p={r['p_two_sided']:.3f}")
        res["model_comparison"] = rows

    summary_update({"validity": res})
    return res

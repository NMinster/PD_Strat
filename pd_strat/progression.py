"""
§6f — Longitudinal progression and prognostic value of the proteomic
severity estimate (PD participants only; controls do not progress).

  1. Per-participant UPDRS slopes (OLS, >=2 visits spanning >=3 months) from
     *all* clinical visits, not only those with proteomics.
  2. Baseline proteomic severity (earliest visit with a prediction) vs slope:
     Spearman ρ with bootstrap CI, residual (after baseline UPDRS) association,
     and a combined OLS with HC3 robust SEs.
  3. Mixed model:  UPDRS ~ pred0_z × months + baseline_UPDRS + (1 | participant).
  4. Cox proportional hazards (statsmodels PHReg) for every endpoint found in
     results/tables/time_to_event.csv, HR per SD of baseline prediction with
     and without adjustment for baseline UPDRS, age and sex; Harrell's C.
  5. Progression by molecular subtype (TRAIN) when subtypes are available.

Everything is computed separately for TRAIN (OOF predictions) and TEST.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from .config import TAB, SEED, PROGRESSION_MIN_VISITS, PROGRESSION_MIN_SPAN_MONTHS
from .utils import map_participant_id, spearman_np, summary_update


def _months(clin) -> np.ndarray:
    m = pd.to_numeric(clin.get("visit_month"), errors="coerce").values.astype(float)
    if np.isfinite(m).sum() == 0 and "visit_code" in clin.columns:
        m = pd.to_numeric(clin["visit_code"].astype(str).str.extract(r"M(\d+)", expand=False),
                          errors="coerce").values.astype(float)
    return m


def participant_slopes(clin, y_all, keep_mask) -> pd.DataFrame:
    """OLS slope of UPDRS on months per participant (rows in keep_mask)."""
    months = _months(clin)
    pids = np.array([map_participant_id(str(x)) for x in clin.index])
    df = pd.DataFrame({"pid": pids, "months": months, "y": y_all})[keep_mask]
    df = df.dropna()
    rows = []
    for pid, g in df.groupby("pid"):
        g = g.sort_values("months")
        span = g["months"].max() - g["months"].min()
        if len(g) < PROGRESSION_MIN_VISITS or span < PROGRESSION_MIN_SPAN_MONTHS:
            continue
        b = np.polyfit(g["months"].values, g["y"].values, 1)
        rows.append({"pid": pid, "slope_per_month": float(b[0]),
                     "slope_per_year": float(12 * b[0]), "n_visits": int(len(g)),
                     "span_months": float(span), "baseline_y": float(g["y"].iloc[0]),
                     "baseline_month": float(g["months"].iloc[0])})
    return pd.DataFrame(rows)


def _baseline_pred(clin, positions, pred) -> pd.DataFrame:
    months = _months(clin)[positions]
    df = pd.DataFrame({"pid": [map_participant_id(str(x)) for x in clin.index[positions]],
                       "months": months, "pred": pred}).dropna()
    df = df.sort_values(["pid", "months"]).drop_duplicates("pid", keep="first")
    return df.rename(columns={"months": "pred_month", "pred": "pred0"})


def _boot_rho(a, b, B=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(a)
    if n < 10:
        return np.nan, np.nan
    vals = [spearman_np(a[ix], b[ix]) for ix in (rng.integers(0, n, n) for _ in range(B))]
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def _harrell_c(time, event, risk) -> float:
    n = len(time); num = den = 0.0
    for i in range(n):
        if not event[i]:
            continue
        for j in range(n):
            if time[j] > time[i]:
                den += 1
                if risk[i] > risk[j]:
                    num += 1
                elif risk[i] == risk[j]:
                    num += 0.5
    return float(num / den) if den > 0 else np.nan


def _slope_analysis(tag, merged: pd.DataFrame) -> Dict[str, Any]:
    import statsmodels.api as sm
    d: Dict[str, Any] = {"n_participants": int(len(merged))}
    if len(merged) < 20:
        return d
    s = merged["slope_per_month"].values
    p0 = merged["pred0"].values
    y0 = merged["baseline_y"].values
    d["rho_baselineUPDRS_slope"] = spearman_np(y0, s)
    d["rho_pred0_slope"] = spearman_np(p0, s)
    d["rho_pred0_slope_ci"] = list(_boot_rho(p0, s))
    resid = p0 - np.polyval(np.polyfit(y0, p0, 1), y0)
    d["rho_pred0_resid_slope"] = spearman_np(resid, s)
    d["rho_pred0_resid_slope_ci"] = list(_boot_rho(resid, s))
    Xd = sm.add_constant(np.column_stack([
        (p0 - p0.mean()) / p0.std(ddof=1), (y0 - y0.mean()) / y0.std(ddof=1)]))
    fit = sm.OLS(s, Xd).fit(cov_type="HC3")
    d["ols_slope_on_pred0_and_baseline"] = {
        "beta_pred0_per_SD": float(fit.params[1]), "p_pred0": float(fit.pvalues[1]),
        "beta_baselineUPDRS_per_SD": float(fit.params[2]), "p_baselineUPDRS": float(fit.pvalues[2]),
        "r2": float(fit.rsquared), "n": int(fit.nobs)}
    print(f"    {tag}: n={len(merged)}  rho(baseline UPDRS, slope)={d['rho_baselineUPDRS_slope']:+.3f}  "
          f"rho(pred0, slope)={d['rho_pred0_slope']:+.3f} "
          f"[{d['rho_pred0_slope_ci'][0]:+.3f}, {d['rho_pred0_slope_ci'][1]:+.3f}]  "
          f"rho(resid, slope)={d['rho_pred0_resid_slope']:+.3f}  "
          f"combined p(pred0)={d['ols_slope_on_pred0_and_baseline']['p_pred0']:.2e}")
    return d


def _mixed_model(tag, clin, y_all, keep_mask, bl: pd.DataFrame) -> Dict[str, Any]:
    import statsmodels.formula.api as smf
    months = _months(clin)
    pids = np.array([map_participant_id(str(x)) for x in clin.index])
    df = pd.DataFrame({"pid": pids, "months": months, "y": y_all})[keep_mask].dropna()
    df = df.merge(bl[["pid", "pred0", "baseline_y"]], on="pid", how="inner")
    cnt = df.groupby("pid")["months"].agg(["count", lambda s: s.max() - s.min()])
    ok = cnt[(cnt["count"] >= PROGRESSION_MIN_VISITS) &
             (cnt.iloc[:, 1] >= PROGRESSION_MIN_SPAN_MONTHS)].index
    df = df[df["pid"].isin(ok)]
    d: Dict[str, Any] = {"n_rows": int(len(df)), "n_participants": int(df["pid"].nunique())}
    if df["pid"].nunique() < 20:
        return d
    df["pred0_z"] = (df["pred0"] - df["pred0"].mean()) / df["pred0"].std(ddof=1)
    df["bl_z"] = (df["baseline_y"] - df["baseline_y"].mean()) / df["baseline_y"].std(ddof=1)
    df["months_y"] = df["months"] / 12.0
    import warnings
    for lab, f in (("adjusted", "y ~ pred0_z * months_y + bl_z"),
                   ("unadjusted", "y ~ pred0_z * months_y")):
        res = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for kw in ({"re_formula": "~months_y"}, {}):
                try:
                    res = smf.mixedlm(f, df, groups=df["pid"], **kw).fit(
                        reml=True, method="lbfgs", maxiter=500)
                    if np.isfinite(res.bse_fe["pred0_z:months_y"]):
                        break
                except Exception:
                    res = None
        if res is None:
            d[lab] = {"error": "mixed model did not converge"}
            continue
        k = "pred0_z:months_y"
        ci = res.conf_int().loc[k]
        d[lab] = {"beta_pred0_x_year": float(res.fe_params[k]),
                  "ci_lo": float(ci[0]), "ci_hi": float(ci[1]),
                  "p": float(res.pvalues[k]), "formula": f}
        print(f"    {tag} mixed ({lab}): beta(pred0_z x year)={d[lab]['beta_pred0_x_year']:+.3f} "
              f"[{ci[0]:+.3f}, {ci[1]:+.3f}] p={d[lab]['p']:.2e}")
    return d


def _cox(tag, tte: pd.DataFrame, bl: pd.DataFrame, clin) -> List[Dict[str, Any]]:
    from statsmodels.duration.hazard_regression import PHReg
    from .config import EXTRA_BIOMARKERS
    rows = []
    demo = pd.DataFrame({
        "pid": [map_participant_id(str(x)) for x in clin.index],
        "age": pd.to_numeric(clin.get("_age"), errors="coerce").values,
        "male": clin.get("_sex", pd.Series(index=clin.index, dtype=object))
                    .astype(str).str.upper().str.startswith("M").astype(float).values,
    })
    extra = []
    for spec in EXTRA_BIOMARKERS:
        n = str(spec.get("name", ""))
        col = f"{n}_baseline" if f"{n}_baseline" in clin.columns else n
        if col in clin.columns:
            demo[n] = pd.to_numeric(clin[col], errors="coerce").values
            extra.append(n)
    demo = demo.groupby("pid").first().reset_index()
    for ep, g in tte.groupby("endpoint"):
        df = g.merge(bl, on="pid").merge(demo, on="pid", how="left").dropna(
            subset=["time", "event", "pred0", "baseline_y"])
        df = df[df["time"] > 0]
        n_ev = int(df["event"].sum())
        if len(df) < 30 or n_ev < 10:
            rows.append({"split": tag, "endpoint": ep, "n": int(len(df)), "n_events": n_ev,
                         "note": "insufficient"})
            continue
        z = lambda v: (v - v.mean()) / v.std(ddof=1)
        specs = [("unadjusted", ["pred0"]),
                 ("adj_baselineUPDRS", ["pred0", "baseline_y"]),
                 ("adj_full", ["pred0", "baseline_y", "age", "male"])]
        avail = [c for c in extra if c in df.columns and df[c].notna().sum() >= 30]
        for c in avail:
            specs.append((f"comparator_{c}_only", [c]))                      # HR is for the biomarker
            specs.append((f"adj_full_plus_{c}", ["pred0", "baseline_y", "age", "male", c]))
        for lab, cols in specs:
            sub = df.dropna(subset=[c for c in cols if c in df.columns])
            if len(sub) < 30 or sub["event"].sum() < 10:
                continue
            X = pd.DataFrame({c: (z(sub[c]) if c != "male" else sub[c]) for c in cols}).fillna(0.0)
            try:
                res = PHReg(sub["time"].values, X.values, status=sub["event"].values,
                            ties="efron").fit()
                hr = float(np.exp(res.params[0])); ci = np.exp(res.conf_int()[0])
                risk = X.values @ res.params
                rows.append({"split": tag, "endpoint": ep, "model": lab, "n": int(len(sub)),
                             "n_events": int(sub["event"].sum()),
                             "first_term": cols[0], "HR_per_SD_pred0": hr,
                             "HR_ci_lo": float(ci[0]), "HR_ci_hi": float(ci[1]),
                             "p": float(res.pvalues[0]),
                             "harrell_C": _harrell_c(sub["time"].values, sub["event"].values, risk)})
                print(f"    {tag} Cox {ep} ({lab}): HR/SD[{cols[0]}]={hr:.2f} "
                      f"[{ci[0]:.2f}, {ci[1]:.2f}] p={res.pvalues[0]:.2e} "
                      f"C={rows[-1]['harrell_C']:.3f} (n={len(sub)}, events={int(sub['event'].sum())})")
            except Exception as e:
                rows.append({"split": tag, "endpoint": ep, "model": lab, "note": str(e)})
    return rows


def run_progression(clin, y_all, train_idx_y, oof_pred, test_idx_omics, test_pred_omics,
                    cohort, sub_results: Optional[dict] = None) -> Dict[str, Any]:
    print(f"\n{'=' * 60}")
    print("PROGRESSION & PROGNOSTIC VALUE (PD participants)")
    print(f"{'=' * 60}")
    is_pd = cohort["is_pd_flag"].values.astype(bool)
    is_train = cohort["is_train"].values.astype(bool)
    is_test = cohort["is_test"].values.astype(bool)
    out: Dict[str, Any] = {}

    slopes_tr = participant_slopes(clin, y_all, is_pd & is_train)
    slopes_te = participant_slopes(clin, y_all, is_pd & is_test)
    slopes = pd.concat([slopes_tr.assign(split="TRAIN"), slopes_te.assign(split="TEST")])
    slopes.to_csv(TAB / "progression_slopes.csv", index=False)
    print(f"  Participants with usable slopes: TRAIN={len(slopes_tr)}, TEST={len(slopes_te)}")
    for tag, s in (("TRAIN", slopes_tr), ("TEST", slopes_te)):
        if len(s):
            out[f"{tag}_slope_summary"] = {
                "n": int(len(s)), "median_slope_per_year": float(s["slope_per_year"].median()),
                "iqr_slope_per_year": [float(s["slope_per_year"].quantile(.25)),
                                       float(s["slope_per_year"].quantile(.75))],
                "median_span_months": float(s["span_months"].median()),
                "median_n_visits": float(s["n_visits"].median())}

    tte_path = TAB / "time_to_event.csv"
    tte = pd.read_csv(tte_path) if tte_path.exists() else None
    if tte is not None:
        tte["pid"] = tte["participant_id"].astype(str).map(map_participant_id)

    cox_rows: List[Dict[str, Any]] = []
    baselines = {}
    for tag, pos, pred, slopes_s, mask in (
            ("TRAIN", train_idx_y, oof_pred, slopes_tr, is_pd & is_train),
            ("TEST", test_idx_omics, test_pred_omics, slopes_te, is_pd & is_test)):
        if len(pos) < 20:
            continue
        pdm = is_pd[pos]
        bl = _baseline_pred(clin, pos[pdm], pred[pdm])
        baselines[tag] = bl
        merged = slopes_s.merge(bl, on="pid", how="inner")
        print(f"\n  --- {tag}: baseline proteomic severity vs progression slope ---")
        out[f"{tag}_slope"] = _slope_analysis(tag, merged)
        merged.assign(split=tag).to_csv(TAB / f"progression_baseline_vs_slope_{tag}.csv",
                                        index=False)
        bl2 = bl.merge(slopes_s[["pid", "baseline_y"]], on="pid", how="inner")
        # trajectory table for figures: every clinical visit of PD participants
        # with a baseline prediction, labelled by predicted-severity tertile
        try:
            months = _months(clin)
            pid_all = np.array([map_participant_id(str(x)) for x in clin.index])
            traj = pd.DataFrame({"pid": pid_all, "months": months, "y": y_all})[mask].dropna()
            q = np.nanpercentile(bl["pred0"], [33.3, 66.7])
            tert = pd.Series(np.digitize(bl["pred0"], q) + 1, index=bl["pid"].values)
            traj = traj[traj["pid"].isin(tert.index)]
            traj["pred0_tertile"] = traj["pid"].map(tert).values
            traj["pred0"] = traj["pid"].map(bl.set_index("pid")["pred0"]).values
            traj.assign(split=tag).to_csv(TAB / f"progression_trajectories_{tag}.csv", index=False)
            if tte is not None:
                km = tte.merge(bl[["pid", "pred0"]], on="pid").assign(split=tag)
                km["pred0_tertile"] = km["pid"].map(tert).values
                km.to_csv(TAB / f"progression_km_{tag}.csv", index=False)
        except Exception as e:
            print(f"    [trajectories] skipped: {e}")
        if len(bl2) >= 20:
            out[f"{tag}_mixed"] = _mixed_model(tag, clin, y_all, mask, bl2)
        if tte is not None and len(bl2) >= 20:
            print(f"  --- {tag}: time-to-event ---")
            cox_rows += _cox(tag, tte, bl2, clin)
    if cox_rows:
        pd.DataFrame(cox_rows).to_csv(TAB / "progression_cox.csv", index=False)
        out["cox"] = cox_rows
    elif tte is None:
        print("  [Cox] no results/tables/time_to_event.csv -- time-to-event skipped")

    if sub_results and "labs_trpd" in sub_results and len(slopes_tr):
        lab = pd.DataFrame({"pid": [map_participant_id(str(x)) for x in sub_results["pd_ids_clean"]],
                            "subtype": sub_results["labs_trpd"]})
        j = slopes_tr.merge(lab, on="pid")
        if j["subtype"].nunique() >= 2 and len(j) >= 20:
            from scipy.stats import kruskal
            grp = [g["slope_per_year"].values for _, g in j.groupby("subtype")]
            H, p = kruskal(*grp)
            out["subtype_progression"] = {
                "n": int(len(j)), "kruskal_p": float(p),
                "median_slope_per_year_by_subtype":
                    j.groupby("subtype")["slope_per_year"].median().round(3).to_dict(),
                "n_by_subtype": j.groupby("subtype").size().to_dict()}
            print(f"\n  Subtype x progression (TRAIN): Kruskal p={p:.3f}; median slope/yr by "
                  f"subtype = {out['subtype_progression']['median_slope_per_year_by_subtype']}")
    summary_update({"progression": out})
    return out

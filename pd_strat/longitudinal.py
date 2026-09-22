"""
§15 — Within-person coupling: does *change* in the proteomic severity score
track *change* in MDS-UPDRS inside the same participant?

Baseline prediction of future progression failed (§5b/§5c).  The monitoring
question is different: a state biomarker only has to move with the clinical
state.  With visit-matched samples (~4 per participant) we can test it.

For PD participants with >= 2 visit-matched samples:

  1. Mixed model with person-mean centring
        UPDRS ~ score_within + score_between + years + (1 | participant)
     score_within  = score - participant mean   (per SD of within deviations)
     score_between = participant mean            (per SD across participants)
     The within-person coefficient is the monitoring claim; the between-person
     coefficient is the cross-sectional severity claim.
  2. Consecutive-visit differences: Spearman rho(delta score, delta UPDRS)
     with participant-bootstrap CI.
  3. Per-participant slopes (>= 3 samples spanning >= 12 months):
     rho(slope of score, slope of UPDRS) across participants.

Run for the primary model and the PD-only refit, in TRAIN (OOF) and TEST;
against DaTSCAN putamen SBR where scans exist (PPMI); and per protein for the
locked confirmatory list (same-sign within-person effects in both cohorts).
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from .config import TAB, ROB, SEED
from .utils import map_participant_id, spearman_np, summary_update, load_protein_annotation
from .progression import _months
from .confirmatory import _fit_mixed


def _frame(clin, y_all, pos, pred, is_pd) -> pd.DataFrame:
    months = _months(clin)
    m = is_pd[pos] & np.isfinite(pred) & np.isfinite(y_all[pos]) & np.isfinite(months[pos])
    p = pos[m]
    df = pd.DataFrame({"pid": [map_participant_id(str(x)) for x in clin.index[p]],
                       "months": months[p], "y": y_all[p], "score": pred[m], "pos": p})
    if "datscan_putamen" in clin.columns:
        df["dat"] = pd.to_numeric(clin["datscan_putamen"].iloc[p], errors="coerce").values
    return df.sort_values(["pid", "months"]).reset_index(drop=True)


def _boot_rho(a, b, groups, B=1000, seed=SEED):
    """Participant-block bootstrap CI for Spearman rho."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    if len(ug) < 10:
        return np.nan, np.nan
    idx_by_g = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(B):
        ix = np.concatenate([idx_by_g[g] for g in rng.choice(ug, len(ug), replace=True)])
        vals.append(spearman_np(a[ix], b[ix]))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def coupling(df: pd.DataFrame, target: str = "y", label: str = "") -> Dict[str, Any]:
    """Within/between decomposition for one (split, model, target)."""
    d = df.dropna(subset=[target, "score"]).copy()
    cnt = d.groupby("pid")["months"].agg(["count", lambda s: s.max() - s.min()])
    ok = cnt[(cnt["count"] >= 2) & (cnt.iloc[:, 1] >= 3)].index
    d = d[d["pid"].isin(ok)]
    out: Dict[str, Any] = {"label": label, "target": target,
                           "n_participants": int(d["pid"].nunique()), "n_samples": int(len(d))}
    if d["pid"].nunique() < 15:
        return out
    pm = d.groupby("pid")["score"].transform("mean")
    d["score_b"] = pm
    d["score_w"] = d["score"] - pm
    sd_w = d["score_w"].std(ddof=1) or 1.0
    sd_b = d.groupby("pid")["score_b"].first().std(ddof=1) or 1.0
    d["score_w"] /= sd_w; d["score_b"] = (d["score_b"] - d["score_b"].mean()) / sd_b
    d["years"] = d["months"] / 12.0
    d = d.rename(columns={target: "tgt"})
    fw = _fit_mixed("tgt ~ score_w + score_b + years", d, d["pid"], "score_w")
    fb = _fit_mixed("tgt ~ score_w + score_b + years", d, d["pid"], "score_b")
    if fw:
        out.update({"within_beta_per_SD": fw["beta"], "within_ci_lo": fw["ci_lo"],
                    "within_ci_hi": fw["ci_hi"], "within_p": fw["p"]})
    if fb:
        out.update({"between_beta_per_SD": fb["beta"], "between_ci_lo": fb["ci_lo"],
                    "between_ci_hi": fb["ci_hi"], "between_p": fb["p"]})
    # consecutive-visit differences
    dd = d.groupby("pid")[["months", "score", "tgt"]].diff().dropna()
    dd["pid"] = d.loc[dd.index, "pid"].values
    dd = dd[dd["months"] > 0]
    if len(dd) >= 20:
        out["n_consecutive_pairs"] = int(len(dd))
        out["rho_delta_score_delta_target"] = spearman_np(dd["score"].values, dd["tgt"].values)
        lo, hi = _boot_rho(dd["score"].values, dd["tgt"].values, dd["pid"].values)
        out["rho_delta_ci_lo"], out["rho_delta_ci_hi"] = lo, hi
    # slope vs slope
    rows = []
    for pid, g in d.groupby("pid"):
        if len(g) >= 3 and g["months"].max() - g["months"].min() >= 12:
            bs = np.polyfit(g["months"], g["score"], 1)[0] * 12
            bt = np.polyfit(g["months"], g["tgt"], 1)[0] * 12
            rows.append((pid, bs, bt))
    if len(rows) >= 15:
        s = pd.DataFrame(rows, columns=["pid", "slope_score", "slope_target"])
        out["n_slope_pairs"] = int(len(s))
        out["rho_slope_score_slope_target"] = spearman_np(s["slope_score"].values, s["slope_target"].values)
        rng = np.random.default_rng(SEED); n = len(s); vals = []
        for _ in range(1000):
            ix = rng.integers(0, n, n)
            vals.append(spearman_np(s["slope_score"].values[ix], s["slope_target"].values[ix]))
        out["rho_slope_ci_lo"], out["rho_slope_ci_hi"] = float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))
    return out, dd.assign(label=label, target=target) if len(dd) else None


def run_longitudinal(clin, z_prot, y_all, cohort, train_idx_y, oof_pred,
                     test_idx_omics, test_pred_omics, prot_cols) -> Dict[str, Any]:
    print(f"\n{'=' * 60}")
    print("WITHIN-PERSON COUPLING (monitoring biomarker; PD, visit-matched samples)")
    print(f"{'=' * 60}")
    is_pd = cohort["is_pd_flag"].values.astype(bool)
    frames: Dict[str, pd.DataFrame] = {}
    if len(train_idx_y):
        frames[("TRAIN", "primary")] = _frame(clin, y_all, train_idx_y, oof_pred, is_pd)
    if len(test_idx_omics):
        frames[("TEST", "primary")] = _frame(clin, y_all, test_idx_omics, test_pred_omics, is_pd)
    p = TAB / "predictions_pd_only.csv"
    if p.exists():
        po = pd.read_csv(p)
        if "pos" in po.columns:
            for split, tag in (("TRAIN_OOF", "TRAIN"), ("TEST", "TEST")):
                s = po[po["split"] == split]
                if len(s):
                    frames[(tag, "pd_only")] = _frame(clin, y_all, s["pos"].values.astype(int),
                                                      s["pred"].values.astype(float), is_pd)
    results: List[Dict[str, Any]] = []
    pairs: List[pd.DataFrame] = []
    for (split, model), df in frames.items():
        for target in (["y", "dat"] if "dat" in df.columns and df["dat"].notna().sum() >= 40 else ["y"]):
            res = coupling(df, target, f"{split}/{model}")
            r, dd = res if isinstance(res, tuple) else (res, None)
            r.update({"split": split, "model": model})
            results.append(r)
            if dd is not None:
                pairs.append(dd.assign(split=split, model=model))
            tname = "UPDRS" if target == "y" else "DaTSCAN putamen"
            if "within_beta_per_SD" in r:
                print(f"  {split:<5} {model:<8} -> {tname:<16} n={r['n_participants']:>3} participants / "
                      f"{r['n_samples']:>4} samples | within b={r['within_beta_per_SD']:+.2f} "
                      f"[{r['within_ci_lo']:+.2f}, {r['within_ci_hi']:+.2f}] p={r['within_p']:.3f} | "
                      f"between b={r.get('between_beta_per_SD', np.nan):+.2f} p={r.get('between_p', np.nan):.3f} | "
                      f"rho(dScore,dTarget)={r.get('rho_delta_score_delta_target', np.nan):+.3f} "
                      f"[{r.get('rho_delta_ci_lo', np.nan):+.2f}, {r.get('rho_delta_ci_hi', np.nan):+.2f}] | "
                      f"rho(slopes)={r.get('rho_slope_score_slope_target', np.nan):+.3f} (n={r.get('n_slope_pairs', 0)})")
            else:
                print(f"  {split:<5} {model:<8} -> {tname}: insufficient longitudinal samples "
                      f"(n={r['n_participants']} participants)")
    res_df = pd.DataFrame(results)
    res_df.to_csv(TAB / "longitudinal_coupling.csv", index=False)
    if pairs:
        pd.concat(pairs).to_csv(TAB / "longitudinal_pairs.csv", index=False)

    # ── protein-level within-person coupling (locked list) ──────────────
    prot_rows = []
    lst_path = ROB / "confirmatory_protein_list.csv"
    if lst_path.exists():
        prots = pd.read_csv(lst_path)["protein"].astype(str).tolist()
        src = "locked confirmatory list"
    else:
        pi = ROB / "protein_importance.csv"
        if pi.exists():
            prots = pd.read_csv(pi).sort_values("abs_importance", ascending=False)["protein"].astype(str).head(40).tolist()
            src = "top-40 |importance|"
        else:
            # robustness skipped: rank by |Spearman| with the target on TRAIN PD samples only
            # (TRAIN-only selection; TEST replication below remains untouched).
            tr_pd = np.asarray(train_idx_y)[is_pd[np.asarray(train_idx_y)]] if len(train_idx_y) else np.array([], int)
            prots, src = [], "top-40 |rho| with target on TRAIN PD"
            if len(tr_pd) >= 20:
                ytr = y_all[tr_pd]
                Z = z_prot.iloc[tr_pd].values.astype(float)
                rk_y = pd.Series(ytr).rank().values
                rk_y = (rk_y - rk_y.mean()) / (rk_y.std() + 1e-12)
                rhos = np.full(Z.shape[1], np.nan)
                for j in range(Z.shape[1]):
                    m = np.isfinite(Z[:, j]) & np.isfinite(ytr)
                    if m.sum() >= 20:
                        rz = pd.Series(Z[m, j]).rank().values
                        rz = (rz - rz.mean()) / (rz.std() + 1e-12)
                        rhos[j] = float(np.mean(rz * rk_y[m]))
                order = np.argsort(-np.abs(np.nan_to_num(rhos)))[:40]
                prots = [prot_cols[j] for j in order if np.isfinite(rhos[j])]
    prots = [x for x in prots if x in set(prot_cols)]
    if prots:
        print(f"  Protein-level coupling on {len(prots)} proteins ({src})")
        annot = load_protein_annotation()
        col_idx = {c: i for i, c in enumerate(prot_cols)}
        for prot in prots:
            row: Dict[str, Any] = {"protein": prot, "gene": annot.get(prot, "")}
            for split, pos in (("TRAIN", train_idx_y), ("TEST", test_idx_omics)):
                if len(pos) == 0:
                    continue
                zc = z_prot.iloc[pos, col_idx[prot]].values.astype(float)
                df = _frame(clin, y_all, pos, zc, is_pd)
                r = coupling(df, "y", f"{split}/{prot}")
                r = r[0] if isinstance(r, tuple) else r
                for k in ("within_beta_per_SD", "within_p", "between_beta_per_SD", "between_p", "n_participants"):
                    if k in r:
                        row[f"{split.lower()}_{k}"] = r[k]
            if "train_within_beta_per_SD" in row and "test_within_beta_per_SD" in row:
                row["within_sign_replicated"] = bool(np.sign(row["train_within_beta_per_SD"]) ==
                                                     np.sign(row["test_within_beta_per_SD"]))
                row["within_replicated_p05"] = bool(row["within_sign_replicated"] and
                                                    row["train_within_p"] < 0.05 and row["test_within_p"] < 0.05)
            prot_rows.append(row)
        pdf = pd.DataFrame(prot_rows)
        if not pdf.empty:
            pdf = pdf.sort_values("test_within_p") if "test_within_p" in pdf else pdf
            pdf.to_csv(TAB / "longitudinal_protein_coupling.csv", index=False)
            n_rep = int(pdf.get("within_replicated_p05", pd.Series(dtype=bool)).sum())
            n_sign = int(pdf.get("within_sign_replicated", pd.Series(dtype=bool)).sum())
            print(f"  Protein-level within-person coupling: {n_rep}/{len(pdf)} replicate (same sign, "
                  f"p<0.05 both cohorts); {n_sign} same sign")
            for _, r in pdf.head(5).iterrows():
                print(f"    {r['protein']:<10} TRAIN within b={r.get('train_within_beta_per_SD', np.nan):+.2f} "
                      f"p={r.get('train_within_p', np.nan):.3f} | TEST b={r.get('test_within_beta_per_SD', np.nan):+.2f} "
                      f"p={r.get('test_within_p', np.nan):.3f}")
    out = {"coupling": results,
           "n_proteins_within_replicated_p05": int(pd.DataFrame(prot_rows).get("within_replicated_p05", pd.Series(dtype=bool)).sum()) if prot_rows else None,
           "n_proteins_tested": len(prot_rows)}
    summary_update({"longitudinal": out})
    return out

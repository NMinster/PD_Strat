"""
§16  Phenoconversion in prodromal enrollees — does the baseline plasma proteome
predict who converts to PD, beyond what a neurologist can assess at that point?

Population: PPMI participants enrolled as *Prodromal* (hyposmia, RBD, genetic
carriers) with a baseline Olink sample and no PD diagnosis at that sample
(table ``ppmi_prodromal_conversion.csv`` written by the PPMI-native clinical
reader).  Conversion = first research or clinical diagnosis in
``ppmi_conversion_codes`` after the sample; censoring at the last dated visit.

Three analyses, each on the TRAIN prodromal participants with the held-out
(holdout_fraction) prodromal participants evaluated once:

1. The PD-severity score (primary model, trained on PD + HC, never on prodromal
   participants) as a conversion predictor: Cox HR per SD, unadjusted and
   adjusted for age, sex, UPSIT, DaTSCAN putamen SBR and baseline Part III;
   Harrell C against a clinical-only Cox model.
2. Time-horizon AUROC (36 / 60 months): score vs a clinical logistic model,
   paired bootstrap Δ.
3. A conversion-specific nested-CV classifier (proteomics / clinical / both,
   model zoo from the discovery benchmark) with the clinical comparator, TEST
   once.

A Δ over the clinical model whose CI excludes 0 on the held-out participants is
the result that supports "predicts conversion beyond clinical assessment".
"""
from __future__ import annotations

import warnings
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .config import TAB, SEED, DISCOVERY_REPEATS, PRODROMAL_HORIZONS
from .utils import map_participant_id, summary_update
from .features import primary_feature_builder
from .validity import _fit_predict
from .progression import _months, _harrell_c
from .discovery import _zoo, _nested_oof, _fit_full, _predict, _metric, _boot_ci

warnings.filterwarnings("ignore")

_CLIN = [("age", "_age"), ("male", "_sex"), ("upsit", "upsit_total"),
         ("datscan_putamen", "datscan_putamen_baseline"), ("updrs_iii", "mds_updrs_part_iii_total"),
         ("rbd", "enrlrbd"), ("hyposmia", "enrlhpsm"), ("lrrk2", "enrllrrk2"), ("gba", "enrlgba"),
         ("datscan_visual_positive", "datscan_visual_baseline")]


def _cox(df: pd.DataFrame, cols: List[str], label: str, split: str) -> Optional[Dict[str, Any]]:
    from statsmodels.duration.hazard_regression import PHReg
    sub = df.dropna(subset=[c for c in cols if c in df.columns] + ["months", "event"])
    sub = sub[sub["months"] > 0]
    if len(sub) < 25 or sub["event"].sum() < 8:
        return {"split": split, "model": label, "n": int(len(sub)), "n_events": int(sub["event"].sum()),
                "note": "insufficient"}
    X = pd.DataFrame({c: (sub[c] if sub[c].nunique() <= 2 else (sub[c] - sub[c].mean()) / (sub[c].std(ddof=1) or 1.0))
                      for c in cols}).fillna(0.0)
    try:
        res = PHReg(sub["months"].values, X.values, status=sub["event"].values, ties="efron").fit()
    except Exception as e:
        return {"split": split, "model": label, "n": int(len(sub)), "note": str(e)}
    risk = X.values @ res.params
    out = {"split": split, "model": label, "n": int(len(sub)), "n_events": int(sub["event"].sum()),
           "first_term": cols[0], "HR_per_SD": float(np.exp(res.params[0])),
           "HR_ci_lo": float(np.exp(res.conf_int()[0][0])), "HR_ci_hi": float(np.exp(res.conf_int()[0][1])),
           "p": float(res.pvalues[0]), "harrell_C": _harrell_c(sub["months"].values, sub["event"].values, risk)}
    print(f"    {split:<6} Cox {label:<22} HR/SD[{cols[0]}]={out['HR_per_SD']:.2f} "
          f"[{out['HR_ci_lo']:.2f}, {out['HR_ci_hi']:.2f}] p={out['p']:.2e} C={out['harrell_C']:.3f} "
          f"(n={out['n']}, events={out['n_events']})")
    return out


def _horizon_label(df: pd.DataFrame, h: float) -> np.ndarray:
    """1 = converted within h months; 0 = followed >= h months without conversion; NaN otherwise."""
    y = np.full(len(df), np.nan)
    ev, t = df["event"].values.astype(float), df["months"].values.astype(float)
    y[(ev == 1) & (t <= h)] = 1.0
    y[(t >= h) & ~((ev == 1) & (t <= h))] = 0.0
    return y


def _km(time: np.ndarray, event: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(time); t, e = time[order], event[order]
    ts, surv, s = [0.0], [1.0], 1.0
    for ut in np.unique(t[e == 1]):
        at_risk = (t >= ut).sum(); d = ((t == ut) & (e == 1)).sum()
        if at_risk > 0:
            s *= 1 - d / at_risk
        ts.append(ut); surv.append(s)
    return np.array(ts), np.array(surv)


def _fig_km(df: pd.DataFrame, made: List[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .figures_advanced import _style, ORD3, _save
    _style()
    d = df.dropna(subset=["score", "months", "event"])
    d = d[d["months"] > 0]
    if len(d) < 30 or d["event"].sum() < 8:
        return
    d = d.assign(tertile=pd.qcut(d["score"], 3, labels=[1, 2, 3]))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for t, col in zip((1, 2, 3), ORD3):
        g = d[d["tertile"] == t]
        ts, sv = _km(g["months"].values, g["event"].values)
        ax.step(ts / 12.0, 1 - sv, where="post", color=col, lw=2,
                label=f"score tertile {t} (n={len(g)}, {int(g['event'].sum())} converted)")
    ax.set_xlabel("years from baseline sample"); ax.set_ylabel("cumulative conversion to PD")
    ax.set_title("Prodromal enrollees: conversion by baseline PD-severity score tertile", loc="left")
    ax.legend(frameon=False, fontsize=9); ax.set_ylim(0, 1)
    _save(fig, "fig_prodromal_conversion_km.png", made)


def run_prodromal(clin, z_prot, X, M, y_all, tv, cohort,
                  PANEL_COL_INDICES, USE_PANEL_AWARE, PRIMARY_LABEL) -> Dict[str, Any]:
    print(f"\n{'=' * 60}")
    print("PRODROMAL PHENOCONVERSION (baseline proteome -> conversion to PD)")
    print(f"{'=' * 60}")
    out: Dict[str, Any] = {}
    path = TAB / "ppmi_prodromal_conversion.csv"
    if not path.exists():
        print("  [SKIP] no ppmi_prodromal_conversion.csv (set ppmi_clinical_dir and rebuild with --force_assembly)")
        return out
    conv = pd.read_csv(path)
    conv["participant_id"] = [map_participant_id(str(x)) for x in conv["participant_id"]]
    conv = conv[~conv["prevalent_at_baseline"].astype(bool)].dropna(subset=["months"])
    if conv.empty:
        print("  [SKIP] no prodromal participants with follow-up")
        return out

    pids = np.array([map_participant_id(str(x)) for x in clin.index])
    months = _months(clin)
    has = (M.sum(1) > 0) & np.isfinite(months)
    is_train = cohort["is_train"].values.astype(bool)
    is_test = cohort["is_test"].values.astype(bool)
    cand = np.where(np.isin(pids, conv["participant_id"].values) & has)[0]
    base = (pd.DataFrame({"pos": cand, "participant_id": pids[cand], "sample_month": months[cand]})
              .sort_values(["participant_id", "sample_month"]).drop_duplicates("participant_id"))
    if len(base) < 30:
        print(f"  [SKIP] only {len(base)} prodromal participants have a proteomic sample")
        return out
    df = base.merge(conv, on="participant_id", how="left")
    # time starts at the sample; drop conversions that precede the sample
    df["months"] = df["months"] - df["sample_month"]
    pre = (df["event"] == 1) & (df["months"] <= 0)
    df = df[~pre & (df["months"] > 0)].copy()
    df["split"] = np.where(is_test[df["pos"].values], "TEST", np.where(is_train[df["pos"].values], "TRAIN", "OTHER"))
    df = df[df["split"] != "OTHER"]

    # severity score transported from the PD + HC model (never trained on prodromal rows)
    bff = primary_feature_builder(PANEL_COL_INDICES, USE_PANEL_AWARE, PRIMARY_LABEL)
    df["score"] = _fit_predict(bff, X, M, y_all, tv["train_idx_y"], df["pos"].values)
    for name, col in _CLIN:
        if col in clin.columns:
            v = clin[col].iloc[df["pos"].values]
            df[name] = (v.astype(str).str.upper().str.startswith("M").astype(float).values if name == "male"
                        else pd.to_numeric(v, errors="coerce").values)
        elif col in df.columns:
            df[name] = pd.to_numeric(df[col], errors="coerce")
    clin_cols = [n for n, _ in _CLIN if n in df.columns and df[n].notna().mean() >= 0.5 and df[n].nunique() > 1]
    df.to_csv(TAB / "prodromal_scores.csv", index=False)
    n_ev = int(df["event"].sum())
    print(f"  {len(df)} prodromal participants with a baseline sample "
          f"(TRAIN {int((df['split'] == 'TRAIN').sum())}, TEST {int((df['split'] == 'TEST').sum())}); "
          f"{n_ev} converted after the sample ({int(pre.sum())} excluded as already diagnosed at sampling); "
          f"median follow-up {df.loc[df['event'] == 0, 'months'].median():.0f} months; "
          f"clinical covariates: {clin_cols}")
    out.update({"n": int(len(df)), "n_events": n_ev, "clinical_covariates": clin_cols})

    # comparator tiers: what a neurologist has at each stage of a prodromal work-up
    _demo = [c for c in ("age", "male") if c in clin_cols]
    _office = _demo + [c for c in ("upsit", "updrs_iii", "rbd", "hyposmia", "lrrk2", "gba") if c in clin_cols]
    _imaging = _office + [c for c in ("datscan_putamen", "datscan_visual_positive") if c in clin_cols]
    tiers: Dict[str, List[str]] = {}
    for name, cols in (("demographics", _demo), ("office", _office), ("imaging", _imaging)):
        if cols and (not tiers or cols != list(tiers.values())[-1]):
            tiers[name] = cols
    out["comparator_tiers"] = tiers
    print("  comparator tiers: " + "; ".join(f"{k} = {v}" for k, v in tiers.items()))

    # 1. Cox ---------------------------------------------------------------
    cox_rows = []
    for split, sub in (("ALL", df), ("TRAIN", df[df["split"] == "TRAIN"]), ("TEST", df[df["split"] == "TEST"])):
        specs = [("unadjusted", ["score"])]
        for tname, cols in tiers.items():
            specs.append((f"adj_{tname}", ["score"] + cols))
            specs.append((f"{tname}_only", cols))
        for lab, cols in specs:
            r = _cox(sub, cols, lab, split)
            if r:
                cox_rows.append(r)
    pd.DataFrame(cox_rows).to_csv(TAB / "prodromal_conversion_cox.csv", index=False)
    out["cox"] = cox_rows

    # 2. horizon AUROC: score vs each comparator tier, and tier + score vs tier ---
    auc_rows = []
    tr, te = df[df["split"] == "TRAIN"], df[df["split"] == "TEST"]
    clf_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                         ("lr", LogisticRegression(max_iter=2000, C=1.0))])

    def _logit_preds(cols: List[str], split: str, sub: pd.DataFrame, y: np.ndarray, ok: np.ndarray,
                     ytr: np.ndarray, oktr: np.ndarray) -> Optional[np.ndarray]:
        """Logistic predictions on `sub[ok]`: out-of-fold within the split for TRAIN / ALL,
        fitted on TRAIN for TEST."""
        X = sub[cols].values.astype(float)
        if split == "TEST":
            if oktr.sum() < 20 or min((ytr[oktr] == 1).sum(), (ytr[oktr] == 0).sum()) < 5:
                return None
            return clone(clf_pipe).fit(tr[cols].values.astype(float)[oktr], ytr[oktr]).predict_proba(X[ok])[:, 1]
        Xk, yk = X[ok], y[ok]
        if min((yk == 1).sum(), (yk == 0).sum()) < 5:
            return None
        q = np.full(ok.sum(), np.nan)
        for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xk, yk):
            q[b] = clone(clf_pipe).fit(Xk[a], yk[a]).predict_proba(Xk[b])[:, 1]
        return q if np.isfinite(q).all() else None

    for h in PRODROMAL_HORIZONS:
        ytr = _horizon_label(tr, h); oktr = np.isfinite(ytr)
        for split, sub in (("TRAIN", tr), ("TEST", te), ("ALL", df)):
            y = _horizon_label(sub, h); ok = np.isfinite(y)
            if ok.sum() < 30 or min((y[ok] == 1).sum(), (y[ok] == 0).sum()) < 8:
                auc_rows.append({"horizon_months": h, "split": split, "n": int(ok.sum()),
                                 "n_converted": int(np.nansum(y)), "note": "insufficient"})
                continue
            s = sub["score"].values[ok]
            base_row = {"horizon_months": h, "split": split, "n": int(ok.sum()), "n_converted": int(y[ok].sum()),
                        "auroc_score": _metric("clf", y[ok], s)}
            base_row.update({f"score_{k}": v for k, v in _boot_ci("clf", y[ok], s).items()})
            line = (f"    {split:<6} {h:>3}-month conversion (n={base_row['n']}, converted={base_row['n_converted']}): "
                    f"score AUROC={base_row['auroc_score']:.3f} [{base_row['score_ci_lo']:.2f}, {base_row['score_ci_hi']:.2f}]")
            if not tiers:
                auc_rows.append(base_row); print(line); continue
            for tname, cols in tiers.items():
                row = dict(base_row); row["comparator"] = tname
                q = _logit_preds(cols, split, sub, y, ok, ytr, oktr)
                qs = _logit_preds(cols + ["score"], split, sub, y, ok, ytr, oktr)
                if q is None:
                    auc_rows.append(row); continue
                row["auroc_comparator"] = _metric("clf", y[ok], q)
                row["delta_score_vs_comparator"] = row["auroc_score"] - row["auroc_comparator"]
                row.update({f"score_vs_comparator_{k}": v for k, v in _boot_ci("clf", y[ok], s, q).items()
                            if k.startswith("delta")})
                if qs is not None:
                    row["auroc_comparator_plus_score"] = _metric("clf", y[ok], qs)
                    row["delta_plus_score_vs_comparator"] = row["auroc_comparator_plus_score"] - row["auroc_comparator"]
                    row.update({f"plus_score_{k}": v for k, v in _boot_ci("clf", y[ok], qs, q).items()
                                if k.startswith("delta")})
                auc_rows.append(row)
                line += (f"\n           vs {tname:<12} comparator={row['auroc_comparator']:.3f}  "
                         f"score delta={row['delta_score_vs_comparator']:+.3f} "
                         f"[{row.get('score_vs_comparator_delta_ci_lo', np.nan):+.3f}, {row.get('score_vs_comparator_delta_ci_hi', np.nan):+.3f}]"
                         + (f"  | comparator+score={row['auroc_comparator_plus_score']:.3f} "
                            f"delta={row['delta_plus_score_vs_comparator']:+.3f} "
                            f"[{row.get('plus_score_delta_ci_lo', np.nan):+.3f}, {row.get('plus_score_delta_ci_hi', np.nan):+.3f}] "
                            f"p={row.get('plus_score_delta_p', np.nan):.3f}" if "auroc_comparator_plus_score" in row else ""))
            print(line)
    pd.DataFrame(auc_rows).to_csv(TAB / "prodromal_conversion_auroc.csv", index=False)
    out["auroc"] = auc_rows

    # 3. conversion-specific nested-CV classifier, TEST once ---------------
    bench_rows = []
    for h in PRODROMAL_HORIZONS:
        ytr = _horizon_label(tr, h); oktr = np.isfinite(ytr)
        if oktr.sum() < 40 or min((ytr[oktr] == 1).sum(), (ytr[oktr] == 0).sum()) < 12:
            print(f"  [benchmark {h} m] insufficient TRAIN outcomes (n={int(oktr.sum())}, "
                  f"converted={int(np.nansum(ytr))}) -> skipped")
            continue
        Xp = z_prot.iloc[tr["pos"].values].values.astype(float)
        Xc = tr[clin_cols].values.astype(float) if clin_cols else np.zeros((len(tr), 0))
        Xc = np.column_stack([Xc, tr["score"].values.astype(float)])     # last clinical column = severity score
        n_prot = Xp.shape[1]
        Xall = np.hstack([Xp, Xc])[oktr]; yv = ytr[oktr]
        prot_idx = np.arange(n_prot); clin_idx = np.arange(n_prot, n_prot + Xc.shape[1] - 1)
        score_idx = np.array([n_prot + Xc.shape[1] - 1])
        print(f"\n  --- benchmark: conversion within {h} months (TRAIN n={len(yv)}, converted={int(yv.sum())}) ---")
        oof: Dict[Tuple[str, str], np.ndarray] = {}
        office_idx = np.array([n_prot + clin_cols.index(c) for c in tiers.get("office", []) if c in clin_cols], int)
        fsets = {"clinical": (False, clin_idx), "score": (False, score_idx),
                 "score+clinical": (False, np.concatenate([clin_idx, score_idx])),
                 "proteomics": (True, np.array([], int)), "proteomics+clinical": (True, clin_idx)}
        if len(office_idx) and len(office_idx) < len(clin_idx):
            fsets["office"] = (False, office_idx)
            fsets["score+office"] = (False, np.concatenate([office_idx, score_idx]))
        for fs, (use_p, ci) in fsets.items():
            pi = prot_idx if use_p else np.array([], int)
            if not use_p and len(ci) == 0:
                continue
            for mname, (pipe, grid) in _zoo("clf", pi, ci, n_prot).items():
                if not use_p and mname == "LogisticL1" and len(ci) < 3:
                    continue
                if fs == "score" and mname != "LogisticSVD":
                    continue
                pooled, per_rep = _nested_oof(pipe, grid, Xall, yv, "clf", DISCOVERY_REPEATS)
                m = _metric("clf", yv, pooled); oof[(fs, mname)] = pooled
                bench_rows.append({"horizon_months": h, "feature_set": fs, "model": mname, "n_train": int(len(yv)),
                                   "oof_auroc": m, "oof_auroc_sd_over_repeats": float(np.nanstd(per_rep))})
                print(f"    {fs:<21} {mname:<13} OOF AUROC={m:.3f} (+-{np.nanstd(per_rep):.3f})")
        clin_keys = [k for k in oof if k[0] == "clinical"]
        office_keys = [k for k in oof if k[0] == "office"]
        prot_keys = [k for k in oof if k[0] not in ("clinical", "office")]
        if not prot_keys:
            continue
        ref = max(clin_keys, key=lambda k: np.nan_to_num(_metric("clf", yv, oof[k]), nan=-9)) if clin_keys else None
        ref_off = max(office_keys, key=lambda k: np.nan_to_num(_metric("clf", yv, oof[k]), nan=-9)) if office_keys else None
        best = max(prot_keys, key=lambda k: np.nan_to_num(_metric("clf", yv, oof[k]), nan=-9))
        # best configuration *without imaging* (score+office / proteomics / proteomics+office)
        no_img = [k for k in prot_keys if k[0] in ("score", "score+office", "proteomics")]
        best_no_img = max(no_img, key=lambda k: np.nan_to_num(_metric("clf", yv, oof[k]), nan=-9)) if no_img else None
        summ: Dict[str, Any] = {"horizon_months": h, "best_feature_set": best[0], "best_model": best[1],
                                "oof_best": _metric("clf", yv, oof[best]),
                                "oof_clinical": _metric("clf", yv, oof[ref]) if ref else np.nan}
        summ.update({f"oof_{k}": v for k, v in _boot_ci("clf", yv, oof[best], oof[ref] if ref else None).items()})
        if ref_off and best_no_img:
            summ["office_ref_model"] = ref_off[1]; summ["oof_office"] = _metric("clf", yv, oof[ref_off])
            summ["best_no_imaging"] = "/".join(best_no_img); summ["oof_best_no_imaging"] = _metric("clf", yv, oof[best_no_img])
            summ.update({f"oof_no_imaging_{k}": v for k, v in _boot_ci("clf", yv, oof[best_no_img], oof[ref_off]).items()
                         if k.startswith("delta")})
        yte = _horizon_label(te, h); okte = np.isfinite(yte)
        if okte.sum() >= 20 and min((yte[okte] == 1).sum(), (yte[okte] == 0).sum()) >= 6:
            fs, mname = best
            use_p, ci = fsets[fs]
            pi = prot_idx if use_p else np.array([], int)
            pipe, grid = _zoo("clf", pi, ci, n_prot)[mname]
            est, params = _fit_full(pipe, grid, Xall, yv, "clf")
            Xt = np.hstack([z_prot.iloc[te["pos"].values].values.astype(float),
                            te[clin_cols].values.astype(float) if clin_cols else np.zeros((len(te), 0)),
                            te["score"].values.astype(float)[:, None]])[okte]
            pt = _predict(est, Xt, "clf")
            summ["n_test"] = int(okte.sum()); summ["test_converted"] = int(yte[okte].sum())
            summ["test_best"] = _metric("clf", yte[okte], pt)
            qt = None
            if ref:
                rpipe, rgrid = _zoo("clf", np.array([], int), clin_idx, n_prot)[ref[1]]
                rest, _ = _fit_full(rpipe, rgrid, Xall, yv, "clf")
                qt = _predict(rest, Xt, "clf"); summ["test_clinical"] = _metric("clf", yte[okte], qt)
                summ["test_delta_vs_clinical"] = summ["test_best"] - summ["test_clinical"]
            summ.update({f"test_{k}": v for k, v in _boot_ci("clf", yte[okte], pt, qt).items()})
            if ref_off and best_no_img:
                opipe, ogrid = _zoo("clf", np.array([], int), office_idx, n_prot)[ref_off[1]]
                oest, _ = _fit_full(opipe, ogrid, Xall, yv, "clf"); qo = _predict(oest, Xt, "clf")
                use_p2, ci2 = fsets[best_no_img[0]]
                npipe, ngrid = _zoo("clf", prot_idx if use_p2 else np.array([], int), ci2, n_prot)[best_no_img[1]]
                nest, _ = _fit_full(npipe, ngrid, Xall, yv, "clf"); pn = _predict(nest, Xt, "clf")
                summ["test_office"] = _metric("clf", yte[okte], qo)
                summ["test_best_no_imaging"] = _metric("clf", yte[okte], pn)
                summ["test_delta_no_imaging_vs_office"] = summ["test_best_no_imaging"] - summ["test_office"]
                summ.update({f"test_no_imaging_{k}": v for k, v in _boot_ci("clf", yte[okte], pn, qo).items()
                             if k.startswith("delta")})
                print(f"       without imaging: {summ['best_no_imaging']} AUROC={summ['test_best_no_imaging']:.3f} vs office "
                      f"{summ['test_office']:.3f}  delta={summ['test_delta_no_imaging_vs_office']:+.3f} "
                      f"[{summ['test_no_imaging_delta_ci_lo']:+.3f}, {summ['test_no_imaging_delta_ci_hi']:+.3f}]")
            print(f"    >> TEST ({h} m): {fs}/{mname} AUROC={summ['test_best']:.3f} vs clinical "
                  f"{summ.get('test_clinical', np.nan):.3f}  delta={summ.get('test_delta_vs_clinical', np.nan):+.3f} "
                  f"[{summ.get('test_delta_ci_lo', np.nan):+.3f}, {summ.get('test_delta_ci_hi', np.nan):+.3f}] "
                  f"(n={summ['n_test']}, converted={summ['test_converted']})")
        else:
            print(f"    TEST ({h} m): insufficient held-out outcomes (n={int(okte.sum())}, "
                  f"converted={int(np.nansum(yte))}) -> OOF only")
        out.setdefault("benchmark", []).append(summ)
    if bench_rows:
        pd.DataFrame(bench_rows).to_csv(TAB / "prodromal_conversion_benchmark_grid.csv", index=False)
    if out.get("benchmark"):
        pd.DataFrame(out["benchmark"]).to_csv(TAB / "prodromal_conversion_benchmark.csv", index=False)

    made: List[str] = []
    try:
        _fig_km(df, made)
    except Exception as e:  # pragma: no cover
        print(f"  [figure] KM skipped: {e}")
    summary_update({"prodromal": out})
    return out

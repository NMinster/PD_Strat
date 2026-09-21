"""
§10n — Confirmatory protein-level analysis with proper replication logic.

The locked list (top-N by stability-weighted importance) is derived from
TRAIN.  Mixed-model p-values in TRAIN are therefore *descriptive* (the
proteins were selected on the same data); the confirmatory test is
replication in TEST: same-sign effect, TEST p-value, Bonferroni across the
locked list in TEST, and confidence-interval overlap between cohorts.

Models
  severity     :  UPDRS ~ protein_z + (1 | participant)
  progression  :  UPDRS ~ protein_z0 × years + baseline_UPDRS + (1 | participant)
                  (protein_z0 = baseline protein level; also fitted without
                  baseline UPDRS to detect baseline-severity confounding)
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from .config import TAB, ROB, LOCKED, PROGRESSION_MIN_VISITS, PROGRESSION_MIN_SPAN_MONTHS
from .utils import map_participant_id, summary_update
from .progression import _months


def _fit_mixed(formula, df, groups, term):
    import warnings
    import statsmodels.formula.api as smf
    for method in ("lbfgs", "powell"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = smf.mixedlm(formula, df, groups=groups).fit(
                    reml=True, method=method, maxiter=500)
            beta = float(res.fe_params[term]); se = float(res.bse_fe[term])
            if np.isfinite(se) and se > 0:
                return {"beta": beta, "se": se, "p": float(res.pvalues[term]),
                        "ci_lo": beta - 1.96 * se, "ci_hi": beta + 1.96 * se,
                        "n_rows": int(res.nobs), "method": method}
        except Exception:
            continue
    return None


def _locked_list(prot_cols: List[str], n: int) -> List[str]:
    p = ROB / "stability_selection.csv"
    if p.exists():
        df = pd.read_csv(p)
        src = "stability_selection (sign_consistency x |median bootstrap weight|)"
        lst = df.sort_values("stable_importance", ascending=False)["protein"].tolist()
    else:
        p = ROB / "protein_importance.csv"
        if not p.exists():
            return []
        df = pd.read_csv(p)
        src = "protein_importance (|back-projected weight|)"
        lst = df.sort_values("abs_importance", ascending=False)["protein"].tolist()
    lst = [x for x in lst if x in set(prot_cols)][:n]
    pd.DataFrame({"protein": lst, "rank": np.arange(1, len(lst) + 1), "source": src}
                 ).to_csv(ROB / "confirmatory_protein_list.csv", index=False)
    print(f"  Locked list: {len(lst)} proteins from {src}")
    return lst


def run_confirmatory(clin, z_prot, y_all, train_idx_y, test_idx_omics, prot_cols,
                     cohort, n_proteins: Optional[int] = None) -> Dict[str, Any]:
    n_proteins = n_proteins or LOCKED["confirmatory_n_proteins"]
    print(f"\n{'=' * 60}")
    print(f"CONFIRMATORY PROTEIN ANALYSIS (locked top {n_proteins}; TEST replication)")
    print(f"{'=' * 60}")
    out: Dict[str, Any] = {}
    prots = _locked_list(prot_cols, n_proteins)
    if not prots:
        print("  [SKIP] no importance table available")
        return out
    col_idx = {c: i for i, c in enumerate(prot_cols)}
    months = _months(clin)
    pid_all = np.array([map_participant_id(str(x)) for x in clin.index])
    is_pd = cohort["is_pd_flag"].values.astype(bool)
    n_bonf = len(prots)

    # ── severity ─────────────────────────────────────────────────────────
    sev_rows = []
    for prot in prots:
        j = col_idx[prot]
        row: Dict[str, Any] = {"protein": prot}
        for tag, pos in (("train", train_idx_y), ("test", test_idx_omics)):
            if len(pos) < 20:
                continue
            df = pd.DataFrame({"z": z_prot.iloc[pos, j].values, "y": y_all[pos],
                               "pid": pid_all[pos]}).dropna()
            if len(df) < 20 or df["pid"].nunique() < 10:
                continue
            fit = _fit_mixed("y ~ z", df, df["pid"], "z")
            if fit:
                for k, v in fit.items():
                    row[f"{tag}_{k}"] = v
                row[f"{tag}_n_participants"] = int(df["pid"].nunique())
        if "train_beta" in row and "test_beta" in row:
            row["sign_replicated"] = bool(np.sign(row["train_beta"]) == np.sign(row["test_beta"]))
            row["ci_overlap"] = bool(max(row["train_ci_lo"], row["test_ci_lo"]) <=
                                     min(row["train_ci_hi"], row["test_ci_hi"]))
            row["test_bonferroni_sig"] = bool(row["test_p"] < 0.05 / n_bonf)
            row["replicated_p05_same_sign"] = bool(row["sign_replicated"] and row["test_p"] < 0.05)
        if "train_p" in row:
            row["train_bonferroni_sig_descriptive"] = bool(row["train_p"] < 0.05 / n_bonf)
        sev_rows.append(row)
    sev = pd.DataFrame(sev_rows)
    if not sev.empty:
        sev = sev.sort_values("test_p" if "test_p" in sev else "train_p")
        sev.to_csv(ROB / "confirmatory_severity.csv", index=False)
        s = {
            "n_tested": int(len(sev)),
            "train_nominal_p05_descriptive": int((sev.get("train_p", pd.Series(dtype=float)) < 0.05).sum()),
            "train_bonferroni_descriptive": int(sev.get("train_bonferroni_sig_descriptive", pd.Series(dtype=bool)).sum()),
            "test_nominal_p05": int((sev.get("test_p", pd.Series(dtype=float)) < 0.05).sum()),
            "test_bonferroni": int(sev.get("test_bonferroni_sig", pd.Series(dtype=bool)).sum()),
            "sign_replicated": int(sev.get("sign_replicated", pd.Series(dtype=bool)).sum()),
            "replicated_same_sign_p05": int(sev.get("replicated_p05_same_sign", pd.Series(dtype=bool)).sum()),
            "ci_overlap": int(sev.get("ci_overlap", pd.Series(dtype=bool)).sum()),
            "bonferroni_alpha": 0.05 / n_bonf,
        }
        out["severity"] = s
        print(f"  Severity: TRAIN nominal {s['train_nominal_p05_descriptive']}/{s['n_tested']} "
              f"(descriptive) | TEST nominal {s['test_nominal_p05']}, Bonferroni {s['test_bonferroni']}, "
              f"same-sign & p<.05 {s['replicated_same_sign_p05']}, sign replicated "
              f"{s['sign_replicated']}, CI overlap {s['ci_overlap']}")
        top = sev.head(5)
        for _, r in top.iterrows():
            print(f"    {r['protein']:<10} TRAIN b={r.get('train_beta', np.nan):+.2f} "
                  f"p={r.get('train_p', np.nan):.1e} | TEST b={r.get('test_beta', np.nan):+.2f} "
                  f"p={r.get('test_p', np.nan):.1e}")

    # ── progression (baseline protein × time), PD only ───────────────────
    prog_rows = []
    for tag, mask in (("train", cohort["is_train"].values & is_pd),
                      ("test", cohort["is_test"].values & is_pd)):
        rows_pos = np.where(mask & np.isfinite(y_all) & np.isfinite(months))[0]
        if len(rows_pos) < 40:
            continue
        base = pd.DataFrame({"pid": pid_all[rows_pos], "months": months[rows_pos],
                             "y": y_all[rows_pos], "pos": rows_pos})
        g = base.groupby("pid")["months"].agg(["count", lambda s: s.max() - s.min()])
        ok = g[(g["count"] >= PROGRESSION_MIN_VISITS) & (g.iloc[:, 1] >= PROGRESSION_MIN_SPAN_MONTHS)].index
        base = base[base["pid"].isin(ok)].sort_values(["pid", "months"])
        bl_y = base.groupby("pid")["y"].first().rename("bl_y")
        base = base.merge(bl_y, on="pid")
        base["years"] = base["months"] / 12.0
        if base["pid"].nunique() < 20:
            continue
        for prot in prots:
            j = col_idx[prot]
            zc = z_prot.iloc[:, j].values
            zdf = pd.DataFrame({"pid": pid_all, "months": months, "z": zc}).dropna()
            z0 = zdf.sort_values(["pid", "months"]).drop_duplicates("pid").set_index("pid")["z"]
            df = base.merge(z0.rename("z0"), left_on="pid", right_index=True, how="inner")
            if df["pid"].nunique() < 20:
                continue
            row: Dict[str, Any] = {"protein": prot, "split": tag,
                                   "n_participants": int(df["pid"].nunique())}
            fa = _fit_mixed("y ~ z0 * years + bl_y", df, df["pid"], "z0:years")
            fu = _fit_mixed("y ~ z0 * years", df, df["pid"], "z0:years")
            if fa:
                row.update({f"adj_{k}": v for k, v in fa.items()})
            if fu:
                row.update({f"unadj_{k}": v for k, v in fu.items()})
            if fa and fu:
                row["sign_flip_with_baseline_adjustment"] = bool(np.sign(fa["beta"]) != np.sign(fu["beta"]))
            prog_rows.append(row)
    prog = pd.DataFrame(prog_rows)
    if not prog.empty:
        wide = prog.pivot(index="protein", columns="split")
        wide.columns = [f"{b}_{a}" for a, b in wide.columns]
        wide = wide.reset_index()
        if "train_adj_beta" in wide and "test_adj_beta" in wide:
            wide["interaction_sign_replicated"] = np.sign(wide["train_adj_beta"]) == np.sign(wide["test_adj_beta"])
        wide.to_csv(ROB / "confirmatory_progression.csv", index=False)
        pr = {"n_tested": int(len(wide))}
        for tag in ("train", "test"):
            if f"{tag}_adj_p" in wide:
                pr[f"{tag}_nominal_p05"] = int((wide[f"{tag}_adj_p"] < 0.05).sum())
                pr[f"{tag}_bonferroni"] = int((wide[f"{tag}_adj_p"] < 0.05 / n_bonf).sum())
        if "interaction_sign_replicated" in wide:
            pr["interaction_sign_replicated"] = int(wide["interaction_sign_replicated"].sum())
        if "train_sign_flip_with_baseline_adjustment" in wide:
            pr["train_sign_flips_baseline_adjustment"] = int(
                wide["train_sign_flip_with_baseline_adjustment"].fillna(False).sum())
        out["progression"] = pr
        print(f"  Progression (protein x time): {pr}")
    summary_update({"confirmatory": out})
    return out

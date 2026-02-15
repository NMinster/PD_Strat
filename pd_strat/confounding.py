"""
§9 — Confounding audit: sex, site, and age effects on MSI_U.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge as RidgeModel

from .config import TAB
from .utils import spearman_np, summary_update


def build_confounder_matrix(demo_df: pd.DataFrame,
                            reference_cols: Optional[List[str]] = None
                            ) -> Tuple[Optional[np.ndarray],
                                       List[str], Dict[str, List[int]]]:
    """Build confounder design matrix from demographics."""
    parts: list = []
    names: list = []
    groups: Dict[str, List[int]] = {}
    diag: list = []

    age = pd.to_numeric(demo_df["age"], errors="coerce").values
    n_fin = int(np.isfinite(age).sum())
    if n_fin >= 5:
        a = age.copy()
        a[~np.isfinite(a)] = np.nanmean(a)
        groups["age"] = [len(names)]
        names.append("age")
        parts.append(a.reshape(-1, 1))
        diag.append(f"age({n_fin})")
    else:
        diag.append(f"age SKIP({n_fin})")

    sex_s = demo_df["sex"].astype(str).fillna("UNKNOWN")
    dum_sex = pd.get_dummies(sex_s, prefix="sex", drop_first=True)
    if dum_sex.shape[1] >= 1:
        groups["sex"] = list(range(len(names), len(names) + dum_sex.shape[1]))
        names.extend(dum_sex.columns.tolist())
        parts.append(dum_sex.values.astype(np.float64))
        diag.append(f"sex({sex_s.nunique()}->{dum_sex.shape[1]})")

    site_s = demo_df["site"].astype(str).fillna("UNKNOWN")
    dum_site = pd.get_dummies(site_s, prefix="site", drop_first=True)
    if dum_site.shape[1] >= 1:
        groups["site"] = list(range(len(names), len(names) + dum_site.shape[1]))
        names.extend(dum_site.columns.tolist())
        parts.append(dum_site.values.astype(np.float64))
        diag.append(f"site({site_s.nunique()}->{dum_site.shape[1]})")

    if not parts:
        print(f"    [Conf] No usable confounders: {', '.join(diag)}")
        return None, [], {}

    X = np.hstack(parts)
    print(f"    [Conf] {X.shape[1]} cols: {', '.join(diag)}")

    if reference_cols is not None:
        df_x = pd.DataFrame(X, columns=names)
        df_x = df_x.reindex(columns=reference_cols, fill_value=0.0)
        X = df_x.values.astype(np.float64)
        names = list(df_x.columns)
        groups_new = {}
        for gname, idxs in groups.items():
            orig = [names[i] for i in idxs if i < len(names)]
            groups_new[gname] = [names.index(n) for n in orig if n in names]
        groups = groups_new

    return X, names, groups


def _residualise(X_conf, y, alpha=1.0):
    m = np.isfinite(y)
    if X_conf is None or m.sum() < 10:
        return y.copy(), None
    mdl = RidgeModel(alpha=alpha)
    mdl.fit(X_conf[m], y[m])
    r = y.copy()
    r[m] = y[m] - mdl.predict(X_conf[m])
    return r, mdl


def run_confounding_audit(sub_results: dict):
    """Run confounding audit on MSI_U and subtype labels."""
    print(f"\n{'=' * 60}")
    print("CONFOUNDING AUDIT -- sex / site / age")
    print(f"{'=' * 60}")

    demo_train = sub_results["demo_msi_train"]
    demo_test  = sub_results["demo_msi_test"]
    msi_u_train = sub_results["msi_u_train"]
    msi_u_test  = sub_results["msi_u_test"]
    y_msi_train = sub_results["y_msi_train"]
    y_msi_test  = sub_results["y_msi_test"]

    X_conf_train, conf_cols, conf_groups = build_confounder_matrix(demo_train)
    X_conf_test, _, _ = build_confounder_matrix(demo_test, reference_cols=conf_cols)
    conf_groups_test = conf_groups.copy()

    confounding_report: Dict[str, Dict] = {}
    mdl_msi_conf = mdl_updrs_conf = None

    for tag, msi_vals, updrs_vals, X_conf, grp in [
        ("TRAIN", msi_u_train, y_msi_train, X_conf_train, conf_groups),
        ("TEST", msi_u_test, y_msi_test, X_conf_test, conf_groups_test),
    ]:
        n = len(msi_vals)
        print(f"\n  --- {tag} (n={n}) ---")
        row: Dict[str, Any] = {"cohort": tag, "n": n}

        rho_u = spearman_np(msi_vals, updrs_vals)
        row["rho_unadjusted"] = float(rho_u)
        print(f"    MSI_U~UPDRS unadjusted rho = {rho_u:.3f}")

        if X_conf is not None and X_conf.shape[1] >= 1:
            if tag == "TRAIN":
                resid_msi, mdl_msi_conf = _residualise(X_conf, msi_vals)
                resid_updrs, mdl_updrs_conf = _residualise(X_conf, updrs_vals)
            else:
                resid_msi = (msi_vals - mdl_msi_conf.predict(X_conf)
                             if mdl_msi_conf else msi_vals.copy())
                resid_updrs = (updrs_vals - mdl_updrs_conf.predict(X_conf)
                               if mdl_updrs_conf else updrs_vals.copy())

            row["rho_adj_residMSI"] = float(spearman_np(resid_msi, updrs_vals))
            row["rho_adj_residUPDRS"] = float(spearman_np(msi_vals, resid_updrs))
            print(f"    rho(resid(MSI|conf), UPDRS) = {row['rho_adj_residMSI']:.3f}")
            print(f"    rho(MSI, resid(UPDRS|conf)) = {row['rho_adj_residUPDRS']:.3f}")
        else:
            row["rho_adj_residMSI"] = row["rho_adj_residUPDRS"] = np.nan

        for gname in ("age", "sex", "site"):
            col_key = f"r2_MSI_from_{gname}"
            if X_conf is not None and gname in grp and len(grp[gname]) > 0:
                cidx = grp[gname]
                Xg = X_conf[:, cidx]
                mv = np.isfinite(msi_vals)
                if mv.sum() >= 10:
                    mdl_g = RidgeModel(alpha=1.0)
                    mdl_g.fit(Xg[mv], msi_vals[mv])
                    ss_res = np.sum((msi_vals[mv] - mdl_g.predict(Xg[mv]))**2)
                    ss_tot = np.sum((msi_vals[mv] - msi_vals[mv].mean())**2)
                    r2 = max(0.0, 1 - ss_res / max(ss_tot, 1e-12))
                    row[col_key] = float(r2)
                    print(f"    R2(MSI_U ~ {gname}) = {r2:.4f}")
                else:
                    row[col_key] = np.nan
            else:
                row[col_key] = np.nan

        confounding_report[tag] = row

    pd.DataFrame(list(confounding_report.values())).to_csv(
        TAB / "msi_U_confounding_report.csv", index=False)
    summary_update({"msi_u_confounding": confounding_report})
    return confounding_report

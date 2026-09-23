"""
§14 — Pre-specified discovery benchmark for progression and stratification.

Question: does the baseline plasma proteome predict *future* change better
than what a neurologist already knows at baseline?

Design (locked before any TEST look):
  * Unit of analysis: PD participant; baseline = earliest visit-matched
    proteomic sample (M0 preferred).  Targets are built from *all* later
    clinical visits.
  * Targets      : slope (UPDRS points / year), 24-month change, fast
                   progressor (top tertile of slope; classification),
                   time-to-milestone (Cox; when time_to_event.csv exists),
                   and cross-sectional severity (reference).
  * Feature sets : clinical-only (the neurologist comparator: age, sex,
                   disease duration, baseline UPDRS total & Part III, H&Y,
                   UPSIT, levodopa), proteomics-only, proteomics + clinical.
  * Models       : ridge-on-SVD, elastic net, PLS, kernel ridge (RBF), SVR,
                   random forest, extra trees, gradient boosting, MLP
                   (regression); logistic (SVD / L1), RF, GBT, SVC, MLP
                   (classification); penalised Cox on SVD (survival).
  * Evaluation   : repeated nested CV on TRAIN (outer RepeatedKFold over
                   participants, inner GridSearchCV); pooled OOF metric with
                   participant bootstrap CI; paired Δ vs the clinical model
                   on identical folds; permutation p for the selected config.
                   The OOF-selected configuration per target is then fitted
                   on all TRAIN and evaluated ONCE on TEST.
  * Multiplicity : the number of configurations is reported next to every
                   result; only the single pre-declared selection rule
                   (max pooled OOF metric among proteomic feature sets) is
                   carried to TEST.
  * Enrichment   : for the fast-progressor model, lift curve and implied
                   trial sample-size ratio when enrolling the top X % by
                   predicted risk.

Everything is written to results/tables/discovery_*.csv and summarised in
SUMMARY_REPORT.md §5c.
"""

from __future__ import annotations

import warnings
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR, SVC
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              HistGradientBoostingRegressor, RandomForestClassifier,
                              HistGradientBoostingClassifier)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import RepeatedKFold, GridSearchCV, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from .config import (
    TAB, SEED, DISCOVERY_REPEATS, DISCOVERY_MIN_SPAN_MONTHS, DISCOVERY_PERMUTATIONS,
    DISCOVERY_MODELS, DISCOVERY_INNER_CV,
)
from .utils import map_participant_id, spearman_np, summary_update
from .progression import _months, participant_slopes, _harrell_c

warnings.filterwarnings("ignore")

_CLIN_FEATURES = [("age", "_age"), ("male", "_sex"), ("disease_duration", "disease_duration_years"),
                  ("updrs_total_bl", None), ("updrs_iii_bl", "mds_updrs_part_iii_total"),
                  ("hoehn_yahr_bl", "hoehn_yahr"), ("upsit_bl", "upsit_total"),
                  ("levodopa", "on_levodopa")]


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  data assembly                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _baseline_frame(clin, z_prot, M_prot, y_all, positions) -> pd.DataFrame:
    """One row per PD participant: earliest visit-matched proteomic sample."""
    months = _months(clin)
    has = M_prot[positions].sum(1) > 0
    pos = positions[has & np.isfinite(y_all[positions]) & np.isfinite(months[positions])]
    df = pd.DataFrame({"pos": pos, "pid": [map_participant_id(str(x)) for x in clin.index[pos]],
                       "months": months[pos]})
    df = df.sort_values(["pid", "months"]).drop_duplicates("pid", keep="first")
    return df.reset_index(drop=True)


def _clinical_matrix(clin, base: pd.DataFrame, y_all) -> Tuple[np.ndarray, List[str]]:
    cols, names = [], []
    for nm, col in _CLIN_FEATURES:
        if nm == "updrs_total_bl":
            v = y_all[base["pos"].values].astype(float)
        elif col is None or col not in clin.columns:
            continue
        elif nm == "male":
            v = clin[col].iloc[base["pos"].values].astype(str).str.upper().str.startswith("M").astype(float).values
        else:
            v = pd.to_numeric(clin[col].iloc[base["pos"].values], errors="coerce").values.astype(float)
        if np.isfinite(v).sum() >= 0.5 * len(v):
            cols.append(v); names.append(nm)
    return (np.column_stack(cols) if cols else np.zeros((len(base), 0))), names


def _targets(clin, y_all, base: pd.DataFrame, mask_rows: np.ndarray) -> pd.DataFrame:
    """Progression targets from all clinical visits at/after the baseline."""
    months = _months(clin)
    pids = np.array([map_participant_id(str(x)) for x in clin.index])
    rows = pd.DataFrame({"pid": pids, "months": months, "y": y_all})[mask_rows].dropna()
    out = base[["pid", "months"]].rename(columns={"months": "m0"}).copy()
    out["y0"] = y_all[base["pos"].values]
    slope, delta24, n_vis, span = [], [], [], []
    for _, r in out.iterrows():
        g = rows[(rows["pid"] == r["pid"]) & (rows["months"] >= r["m0"] - 0.5)].sort_values("months")
        sp = g["months"].max() - g["months"].min() if len(g) else 0.0
        n_vis.append(len(g)); span.append(sp)
        if len(g) >= 2 and sp >= DISCOVERY_MIN_SPAN_MONTHS:
            b = np.polyfit(g["months"].values - r["m0"], g["y"].values, 1)
            slope.append(12 * b[0])
        else:
            slope.append(np.nan)
        tgt = g[(g["months"] - r["m0"]).between(18, 30)]
        delta24.append(float(tgt["y"].iloc[0] - r["y0"]) if len(tgt) else np.nan)
    out["slope_per_year"] = slope; out["delta_24m"] = delta24
    out["n_visits"] = n_vis; out["span_months"] = span
    # DaTSCAN targets (objective; PPMI only): baseline putamen SBR and its
    # annualised change over >= 12 months of follow-up scans.  The baseline
    # scan is the one closest to the baseline proteomic sample within
    # [-12, +6] months (PPMI scans at screening, before the baseline visit;
    # some participants' first usable sample is a later visit).
    if "datscan_putamen" in clin.columns:
        dat = pd.to_numeric(clin["datscan_putamen"], errors="coerce").values
        drows = pd.DataFrame({"pid": pids, "months": months, "dat": dat}).dropna()
        d0, dsl, dlag = [], [], []
        for _, r in out.iterrows():
            g = drows[(drows["pid"] == r["pid"]) & (drows["months"] >= r["m0"] - 12)].sort_values("months")
            lag = g["months"] - r["m0"]
            near = g[(lag >= -12) & (lag <= 6)]
            if len(near):
                k = (near["months"] - r["m0"]).abs().idxmin()
                d0.append(float(near.loc[k, "dat"])); dlag.append(float(near.loc[k, "months"] - r["m0"]))
            else:
                d0.append(np.nan); dlag.append(np.nan)
            if len(g) >= 2 and g["months"].max() - g["months"].min() >= 12:
                dsl.append(12 * np.polyfit(g["months"].values, g["dat"].values, 1)[0])
            else:
                dsl.append(np.nan)
        out["datscan_putamen_bl"] = d0; out["datscan_putamen_bl_lag_months"] = dlag
        out["datscan_putamen_change_per_year"] = dsl
    return out


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  model zoo                                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class _PLS(BaseEstimator, RegressorMixin):
    def __init__(self, n_components=4):
        self.n_components = n_components

    def fit(self, X, y):
        k = int(min(self.n_components, X.shape[1], max(1, X.shape[0] - 2)))
        self.m_ = PLSRegression(n_components=k).fit(X, y)
        return self

    def predict(self, X):
        return np.asarray(self.m_.predict(X)).ravel()


class _SafeSVD(TruncatedSVD):
    """TruncatedSVD whose n_components is capped by the fold's sample size.
    (TruncatedSVD.fit delegates to fit_transform, so only that is overridden.)"""

    def fit_transform(self, X, y=None):
        self.n_components = int(max(1, min(self.n_components, X.shape[1] - 1, X.shape[0] - 2)))
        return super().fit_transform(X, y)


def _pre(prot_idx, clin_idx, reduce: Optional[int]):
    """ColumnTransformer: proteins (impute→scale[→SVD]) + clinical (impute→scale)."""
    prot_steps = [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
    if reduce:
        prot_steps.append(("svd", _SafeSVD(n_components=reduce, random_state=SEED)))
    parts = []
    if len(prot_idx):
        parts.append(("prot", Pipeline(prot_steps), list(prot_idx)))
    if len(clin_idx):
        parts.append(("clin", Pipeline([("imp", SimpleImputer(strategy="median")),
                                        ("sc", StandardScaler())]), list(clin_idx)))
    return ColumnTransformer(parts, remainder="drop")


def _zoo(task: str, prot_idx, clin_idx, n_prot: int) -> Dict[str, Tuple[Pipeline, dict]]:
    """name -> (pipeline, param_grid).  Grids are deliberately small."""
    has_prot = len(prot_idx) > 0
    red = 16 if has_prot else None
    Z: Dict[str, Tuple[Pipeline, dict]] = {}
    if task == "reg":
        Z["RidgeSVD"] = (Pipeline([("pre", _pre(prot_idx, clin_idx, red)), ("m", Ridge())]),
                         {"m__alpha": [0.1, 1, 10, 100], **({"pre__prot__svd__n_components": [4, 8, 16, 32]} if has_prot else {})})
        Z["ElasticNet"] = (Pipeline([("pre", _pre(prot_idx, clin_idx, None)), ("m", ElasticNet(max_iter=5000))]),
                           {"m__alpha": [0.05, 0.2, 1.0], "m__l1_ratio": [0.2, 0.5, 0.9]})
        Z["PLS"] = (Pipeline([("pre", _pre(prot_idx, clin_idx, None)), ("m", _PLS())]),
                    {"m__n_components": [2, 4, 8]})
        Z["KernelRidgeRBF"] = (Pipeline([("pre", _pre(prot_idx, clin_idx, red)), ("m", KernelRidge(kernel="rbf"))]),
                               {"m__alpha": [0.1, 1, 10], "m__gamma": [0.01, 0.05, 0.2]})
        Z["SVR"] = (Pipeline([("pre", _pre(prot_idx, clin_idx, red)), ("m", SVR())]),
                    {"m__C": [1, 10, 100], "m__gamma": ["scale"], "m__epsilon": [0.5, 2.0]})
        Z["RandomForest"] = (Pipeline([("pre", _pre(prot_idx, clin_idx, None)),
                                       ("m", RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1))]),
                             {"m__min_samples_leaf": [3, 6], "m__max_features": ["sqrt", 0.3]})
        Z["ExtraTrees"] = (Pipeline([("pre", _pre(prot_idx, clin_idx, None)),
                                     ("m", ExtraTreesRegressor(n_estimators=300, random_state=SEED, n_jobs=1))]),
                           {"m__min_samples_leaf": [3, 6]})
        Z["HistGBT"] = (Pipeline([("pre", _pre(prot_idx, clin_idx, None)),
                                  ("m", HistGradientBoostingRegressor(random_state=SEED, max_iter=300,
                                                                      early_stopping=True))]),
                        {"m__learning_rate": [0.03, 0.1], "m__max_depth": [2, 4], "m__min_samples_leaf": [8]})
        Z["MLP"] = (Pipeline([("pre", _pre(prot_idx, clin_idx, red)),
                              ("m", MLPRegressor(hidden_layer_sizes=(32,), max_iter=800, random_state=SEED,
                                                 early_stopping=True, n_iter_no_change=25))]),
                    {"m__alpha": [1e-3, 1e-1, 1.0]})
    else:
        Z["LogisticSVD"] = (Pipeline([("pre", _pre(prot_idx, clin_idx, red)),
                                      ("m", LogisticRegression(max_iter=2000))]),
                            {"m__C": [0.05, 0.3, 1, 5], **({"pre__prot__svd__n_components": [4, 8, 16]} if has_prot else {})})
        Z["LogisticL1"] = (Pipeline([("pre", _pre(prot_idx, clin_idx, None)),
                                     ("m", LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000))]),
                           {"m__C": [0.05, 0.2, 1.0]})
        Z["RandomForest"] = (Pipeline([("pre", _pre(prot_idx, clin_idx, None)),
                                       ("m", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1))]),
                             {"m__min_samples_leaf": [3, 6], "m__max_features": ["sqrt", 0.3]})
        Z["HistGBT"] = (Pipeline([("pre", _pre(prot_idx, clin_idx, None)),
                                  ("m", HistGradientBoostingClassifier(random_state=SEED, max_iter=300,
                                                                       early_stopping=True))]),
                        {"m__learning_rate": [0.03, 0.1], "m__max_depth": [2, 4], "m__min_samples_leaf": [8]})
        Z["SVC"] = (Pipeline([("pre", _pre(prot_idx, clin_idx, red)), ("m", SVC(probability=True, random_state=SEED))]),
                    {"m__C": [0.5, 2, 10], "m__gamma": ["scale"]})
        Z["MLP"] = (Pipeline([("pre", _pre(prot_idx, clin_idx, red)),
                              ("m", MLPClassifier(hidden_layer_sizes=(32,), max_iter=800, random_state=SEED,
                                                  early_stopping=True, n_iter_no_change=25))]),
                    {"m__alpha": [1e-3, 1e-1, 1.0]})
    if DISCOVERY_MODELS:
        Z = {k: v for k, v in Z.items() if k in set(DISCOVERY_MODELS)}
    return Z


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  nested CV engine                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _metric(task, y, p):
    if task == "reg":
        return spearman_np(p, y)
    m = np.isfinite(p)
    return float(roc_auc_score(y[m], p[m])) if len(np.unique(y[m])) == 2 else np.nan


def _predict(model, X, task):
    return model.predict_proba(X)[:, 1] if task == "clf" else model.predict(X)


def _nested_oof(pipe, grid, X, y, task, repeats, seed=SEED):
    """Repeated nested CV -> (per-participant mean OOF prediction, per-repeat metrics)."""
    n = len(y)
    preds = np.full((repeats, n), np.nan)
    rk = RepeatedKFold(n_splits=5, n_repeats=repeats, random_state=seed)
    inner = (StratifiedKFold(DISCOVERY_INNER_CV, shuffle=True, random_state=seed) if task == "clf"
             else KFold(DISCOVERY_INNER_CV, shuffle=True, random_state=seed))
    scoring = "roc_auc" if task == "clf" else "r2"
    for i, (tr, va) in enumerate(rk.split(X, y)):
        r = i // 5
        gs = GridSearchCV(clone(pipe), grid, cv=inner, scoring=scoring, n_jobs=-1, error_score=np.nan)
        try:
            gs.fit(X[tr], y[tr])
            preds[r, va] = _predict(gs.best_estimator_, X[va], task)
        except Exception:
            continue
    per_rep = [_metric(task, y, preds[r]) for r in range(repeats)]
    pooled = np.nanmean(preds, axis=0)
    return pooled, np.array(per_rep, dtype=float)


def _boot_ci(task, y, p, q=None, B=1000, seed=SEED):
    """Bootstrap CI of metric (and of paired Δ vs q when given)."""
    rng = np.random.default_rng(seed)
    n = len(y); vals, deltas = [], []
    for _ in range(B):
        ix = rng.integers(0, n, n)
        m = _metric(task, y[ix], p[ix]); vals.append(m)
        if q is not None:
            deltas.append(m - _metric(task, y[ix], q[ix]))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    out = {"ci_lo": float(lo), "ci_hi": float(hi)}
    if q is not None:
        d = np.array(deltas, dtype=float)
        out.update({"delta_ci_lo": float(np.nanpercentile(d, 2.5)),
                    "delta_ci_hi": float(np.nanpercentile(d, 97.5)),
                    "delta_p": float(min(1.0, 2 * min(np.nanmean(d >= 0), np.nanmean(d <= 0))))})
    return out


def _fit_full(pipe, grid, X, y, task):
    inner = (StratifiedKFold(DISCOVERY_INNER_CV, shuffle=True, random_state=SEED) if task == "clf"
             else KFold(DISCOVERY_INNER_CV, shuffle=True, random_state=SEED))
    gs = GridSearchCV(clone(pipe), grid, cv=inner, scoring="roc_auc" if task == "clf" else "r2",
                      n_jobs=-1, error_score=np.nan).fit(X, y)
    return gs.best_estimator_, gs.best_params_


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  driver                                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def run_discovery(clin, z_prot, M_prot, y_all, cohort, tv) -> Dict[str, Any]:
    print(f"\n{'=' * 60}")
    print(f"DISCOVERY BENCHMARK (pre-specified; nested CV x{DISCOVERY_REPEATS}; TEST once)")
    print(f"{'=' * 60}")
    out: Dict[str, Any] = {}
    is_pd = cohort["is_pd_flag"].values.astype(bool)
    is_train = cohort["is_train"].values.astype(bool)
    is_test = cohort["is_test"].values.astype(bool)

    frames = {}
    for split, mask in (("TRAIN", is_pd & is_train), ("TEST", is_pd & is_test)):
        base = _baseline_frame(clin, z_prot, M_prot, y_all, np.where(mask)[0])
        if len(base) < 20:
            print(f"  [{split}] only {len(base)} PD participants with a baseline sample -> skipped")
            continue
        tg = _targets(clin, y_all, base, mask)
        Xp = z_prot.iloc[base["pos"].values].values.astype(float)
        Xc, cnames = _clinical_matrix(clin, base, y_all)
        frames[split] = dict(base=base, tg=tg, Xp=Xp, Xc=Xc, cnames=cnames)
        print(f"  [{split}] {len(base)} PD participants; clinical features: {cnames}; "
              f"slopes available: {int(tg['slope_per_year'].notna().sum())}, "
              f"24-month change: {int(tg['delta_24m'].notna().sum())}"
              + (f", DaTSCAN baseline: {int(tg['datscan_putamen_bl'].notna().sum())}, "
                 f"DaTSCAN change: {int(tg['datscan_putamen_change_per_year'].notna().sum())}"
                 if "datscan_putamen_bl" in tg.columns else ""))
    if "TRAIN" not in frames:
        return out
    tr = frames["TRAIN"]; te = frames.get("TEST")
    n_prot = tr["Xp"].shape[1]
    out["clinical_features"] = tr["cnames"]
    tr["tg"].assign(split="TRAIN").to_csv(TAB / "discovery_targets_TRAIN.csv", index=False)
    if te:
        te["tg"].assign(split="TEST").to_csv(TAB / "discovery_targets_TEST.csv", index=False)

    # targets --------------------------------------------------------------
    fast_thr = float(np.nanpercentile(tr["tg"]["slope_per_year"], 66.7))
    out["fast_progressor_threshold_slope_per_year"] = fast_thr
    # For the cross-sectional reference target the clinical comparator must not
    # contain the outcome itself (baseline UPDRS / Part III / H&Y).
    _drop_for_target = {"severity_baseline": {"updrs_total_bl", "updrs_iii_bl", "hoehn_yahr_bl"}}
    targets = {
        "severity_baseline": ("reg", lambda f: f["tg"]["y0"].values.astype(float)),
        "slope_per_year": ("reg", lambda f: f["tg"]["slope_per_year"].values.astype(float)),
        "delta_24m": ("reg", lambda f: f["tg"]["delta_24m"].values.astype(float)),
        "fast_progressor": ("clf", lambda f: np.where(np.isfinite(f["tg"]["slope_per_year"].values),
                                                     (f["tg"]["slope_per_year"].values >= fast_thr).astype(float), np.nan)),
    }
    if "datscan_putamen_bl" in tr["tg"].columns:
        targets["datscan_putamen_baseline"] = ("reg", lambda f: (f["tg"]["datscan_putamen_bl"].values.astype(float)
                                                                if "datscan_putamen_bl" in f["tg"] else np.full(len(f["tg"]), np.nan)))
        targets["datscan_putamen_change_per_year"] = ("reg", lambda f: (f["tg"]["datscan_putamen_change_per_year"].values.astype(float)
                                                                       if "datscan_putamen_change_per_year" in f["tg"] else np.full(len(f["tg"]), np.nan)))
    fsets = {"clinical": (False, True), "proteomics": (True, False), "proteomics+clinical": (True, True)}

    grid_rows, best_rows, n_configs = [], [], 0
    best_preds_test: Dict[str, pd.DataFrame] = {}
    for tname, (task, getter) in targets.items():
        y = getter(tr); ok = np.isfinite(y)
        if ok.sum() < 30 or (task == "clf" and min((y[ok] == 1).sum(), (y[ok] == 0).sum()) < 10):
            print(f"\n  [{tname}] insufficient outcomes (n={int(ok.sum())}) -> skipped")
            continue
        print(f"\n  --- target: {tname} ({task}, n={int(ok.sum())}) ---")
        keep_c = [i for i, nm in enumerate(tr["cnames"]) if nm not in _drop_for_target.get(tname, set())]
        Xall = np.hstack([tr["Xp"], tr["Xc"][:, keep_c]])[ok]; yv = y[ok]
        prot_idx = np.arange(n_prot); clin_idx = np.arange(n_prot, n_prot + len(keep_c))
        if keep_c != list(range(tr["Xc"].shape[1])):
            print(f"    clinical comparator for this target: {[tr['cnames'][i] for i in keep_c]}")
        oof: Dict[Tuple[str, str], np.ndarray] = {}
        for fs, (use_p, use_c) in fsets.items():
            pi = prot_idx if use_p else np.array([], int); ci = clin_idx if use_c else np.array([], int)
            if (use_c and len(ci) == 0):
                continue
            for mname, (pipe, grid) in _zoo(task, pi, ci, n_prot).items():
                if not use_p and mname in ("PLS", "ElasticNet", "LogisticL1") and len(ci) < 3:
                    continue
                pooled, per_rep = _nested_oof(pipe, grid, Xall, yv, task, DISCOVERY_REPEATS)
                m = _metric(task, yv, pooled); n_configs += 1
                oof[(fs, mname)] = pooled
                grid_rows.append({"target": tname, "task": task, "feature_set": fs, "model": mname,
                                  "n": int(len(yv)), "oof_metric_pooled": m,
                                  "oof_metric_mean_over_repeats": float(np.nanmean(per_rep)),
                                  "oof_metric_sd_over_repeats": float(np.nanstd(per_rep))})
                print(f"    {fs:<21} {mname:<15} OOF {'AUROC' if task == 'clf' else 'rho'}="
                      f"{m:.3f} (repeats {np.nanmean(per_rep):.3f} +- {np.nanstd(per_rep):.3f})")
        # clinical reference = best clinical-only config by OOF
        clin_keys = [k for k in oof if k[0] == "clinical"]
        ref_key = max(clin_keys, key=lambda k: np.nan_to_num(_metric(task, yv, oof[k]), nan=-9)) if clin_keys else None
        prot_keys = [k for k in oof if k[0] != "clinical"]
        if not prot_keys:
            continue
        best_key = max(prot_keys, key=lambda k: np.nan_to_num(_metric(task, yv, oof[k]), nan=-9))
        row: Dict[str, Any] = {"target": tname, "task": task, "n_train": int(len(yv)),
                               "n_configs_tested": len(oof),
                               "best_feature_set": best_key[0], "best_model": best_key[1],
                               "oof_best": _metric(task, yv, oof[best_key])}
        row.update({f"oof_best_{k}": v for k, v in _boot_ci(task, yv, oof[best_key],
                                                              oof[ref_key] if ref_key else None).items()})
        if ref_key:
            row["clinical_ref_model"] = ref_key[1]; row["oof_clinical"] = _metric(task, yv, oof[ref_key])
            row["oof_delta_vs_clinical"] = row["oof_best"] - row["oof_clinical"]
        # permutation p for the selected config (fixed hyper-parameters)
        fs, mname = best_key
        pi = prot_idx if fsets[fs][0] else np.array([], int); ci = clin_idx if fsets[fs][1] else np.array([], int)
        pipe, grid = _zoo(task, pi, ci, n_prot)[mname]
        est, params = _fit_full(pipe, grid, Xall, yv, task)
        rng = np.random.default_rng(SEED + 99); null = []
        for _ in range(DISCOVERY_PERMUTATIONS):
            yp = rng.permutation(yv); pp = np.full(len(yv), np.nan)
            for trn, va in KFold(5, shuffle=True, random_state=int(rng.integers(1e9))).split(Xall):
                try:
                    e = clone(est).fit(Xall[trn], yp[trn]); pp[va] = _predict(e, Xall[va], task)
                except Exception:
                    pass
            null.append(_metric(task, yp, pp))
        null = np.array(null, dtype=float)
        row["perm_null_mean"] = float(np.nanmean(null)); row["perm_null_max"] = float(np.nanmax(null))
        row["perm_p"] = float((1 + np.nansum(null >= row["oof_best"])) / (len(null) + 1))
        row["best_params"] = str(params)
        # TEST, once ---------------------------------------------------------
        if te is not None:
            yt = getter(te); okt = np.isfinite(yt)
            if okt.sum() >= 20 and not (task == "clf" and min((yt[okt] == 1).sum(), (yt[okt] == 0).sum()) < 8):
                Xt = np.hstack([te["Xp"], te["Xc"][:, keep_c]])[okt]
                pt = _predict(est, Xt, task)
                row["n_test"] = int(okt.sum()); row["test_best"] = _metric(task, yt[okt], pt)
                qt = None
                if ref_key:
                    rpipe, rgrid = _zoo(task, np.array([], int), clin_idx, n_prot)[ref_key[1]]
                    rest, _ = _fit_full(rpipe, rgrid, Xall, yv, task)
                    qt = _predict(rest, Xt, task); row["test_clinical"] = _metric(task, yt[okt], qt)
                    row["test_delta_vs_clinical"] = row["test_best"] - row["test_clinical"]
                row.update({f"test_best_{k}": v for k, v in _boot_ci(task, yt[okt], pt, qt).items()})
                best_preds_test[tname] = pd.DataFrame({
                    "pid": te["tg"]["pid"].values[okt], "y": yt[okt], "pred_best": pt,
                    "pred_clinical": qt if qt is not None else np.nan, "target": tname})
                print(f"    >> TEST: {row['best_feature_set']}/{row['best_model']} = {row['test_best']:.3f} "
                      f"vs clinical {row.get('test_clinical', np.nan):.3f}  "
                      f"delta={row.get('test_delta_vs_clinical', np.nan):+.3f} "
                      f"[{row['test_best_delta_ci_lo']:+.3f}, {row['test_best_delta_ci_hi']:+.3f}]  "
                      f"(OOF perm p={row['perm_p']:.3f})")
        best_rows.append(row)

    grid_df = pd.DataFrame(grid_rows); grid_df.to_csv(TAB / "discovery_grid.csv", index=False)
    best_df = pd.DataFrame(best_rows); best_df.to_csv(TAB / "discovery_best.csv", index=False)
    if best_preds_test:
        pd.concat(best_preds_test.values()).to_csv(TAB / "discovery_predictions_TEST.csv", index=False)
    out["n_configs_total"] = int(n_configs)
    out["best"] = best_rows

    # enrichment for trial design (fast progressor / slope) ------------------
    enr_rows = []
    src = best_preds_test.get("fast_progressor")
    if src is None:
        src = best_preds_test.get("slope_per_year")
    if src is not None and te is not None:
        sl = te["tg"].set_index("pid")["slope_per_year"]
        d = src.set_index("pid").join(sl, how="inner").dropna(subset=["pred_best", "slope_per_year"])
        d = d.sort_values("pred_best", ascending=False)
        fast = (d["slope_per_year"] >= fast_thr).astype(float)
        base_rate = float(fast.mean()); all_mean, all_sd = d["slope_per_year"].mean(), d["slope_per_year"].std(ddof=1)
        for frac in (0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0):
            k = max(5, int(round(frac * len(d)))); top = d.iloc[:k]
            mean_k, sd_k = top["slope_per_year"].mean(), top["slope_per_year"].std(ddof=1)
            ss_ratio = ((sd_k / mean_k) ** 2) / ((all_sd / all_mean) ** 2) if mean_k > 0 and all_mean > 0 else np.nan
            enr_rows.append({"enrolled_fraction": frac, "n_enrolled": int(k),
                             "fast_progressor_rate": float(fast.iloc[:k].mean()), "base_rate": base_rate,
                             "enrichment_factor": float(fast.iloc[:k].mean() / base_rate) if base_rate > 0 else np.nan,
                             "mean_slope_enrolled": float(mean_k), "mean_slope_all": float(all_mean),
                             "relative_trial_sample_size": float(ss_ratio)})
        pd.DataFrame(enr_rows).to_csv(TAB / "discovery_enrichment.csv", index=False)
        out["enrichment"] = enr_rows
        print("\n  Trial-enrichment (TEST, ranked by predicted risk):")
        for r in enr_rows:
            print(f"    top {r['enrolled_fraction']:.0%}: fast-progressor rate {r['fast_progressor_rate']:.2f} "
                  f"(base {base_rate:.2f}, x{r['enrichment_factor']:.2f}); mean slope "
                  f"{r['mean_slope_enrolled']:.2f} vs {all_mean:.2f}; relative N = {r['relative_trial_sample_size']:.2f}")

    # protein x endpoint correlation map (PD, visit-matched rows) -------------
    ep_cols = [("UPDRS_I", "mds_updrs_part_i_total"), ("UPDRS_II", "mds_updrs_part_ii_total"),
               ("UPDRS_III", "mds_updrs_part_iii_total"), ("UPDRS_IV", "mds_updrs_part_iv_total"),
               ("H&Y", "hoehn_yahr"), ("UPSIT", "upsit_total"), ("DaT_putamen", "datscan_putamen")]
    top_path = TAB.parent / "robustness" / "stability_selection.csv"
    if top_path.exists():
        top = pd.read_csv(top_path).head(30)
        rows = []
        for split, mask in (("TRAIN", is_pd & is_train), ("TEST", is_pd & is_test)):
            pos = np.where(mask & (M_prot.sum(1) > 0))[0]
            for _, pr in top.iterrows():
                if pr["protein"] not in z_prot.columns:
                    continue
                zc = z_prot[pr["protein"]].values[pos]
                for ep, col in ep_cols + [("UPDRS_total", None)]:
                    v = y_all[pos] if col is None else (pd.to_numeric(clin[col], errors="coerce").values[pos]
                                                        if col in clin.columns else None)
                    if v is None:
                        continue
                    m = np.isfinite(zc) & np.isfinite(v)
                    if m.sum() >= 20:
                        rows.append({"split": split, "protein": pr["protein"], "gene": pr.get("gene", ""),
                                     "endpoint": ep, "rho": spearman_np(zc[m], v[m]), "n": int(m.sum())})
        if rows:
            pd.DataFrame(rows).to_csv(TAB / "protein_endpoint_correlations.csv", index=False)

    summary_update({"discovery": out})
    return out

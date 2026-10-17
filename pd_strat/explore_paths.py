"""
§17  Path exploration — which of the candidate extensions in
docs/literature_review_2026_gaps.md can the data on disk support?

For every candidate path the stage reports

  (a) coverage: participants and events available, by split and subgroup;
  (b) the minimum effect detectable at 80 % power (two-sided α = 0.05) with that
      coverage — Fisher z for correlations, Schoenfeld for a Cox HR per SD, the
      null standard error of the Mann-Whitney statistic for an AUROC;
  (c) an exploratory signal computed on TRAIN rows only: the out-of-fold severity
      score for PD / HC rows (GroupKFold refit of the primary model) and the
      transported score for prodromal rows (fitted on TRAIN PD + HC).  Holdout
      rows are counted, never scored.

Nothing here is confirmatory.  The exploratory effects rank the paths; a path
that is "feasible" still needs its own pre-specified analysis with the holdout
evaluated once.

Paths (numbering follows the gap analysis):
  P1  plasma proteome -> CSF α-synuclein SAA status / NSD stage
  P2  prodromal phenoconversion: arms, endpoint codes, DaT-negative subgroup, deaths
  P3  dynamic (serial-sample) prodromal proteomics
  P4  cognition: MoCA, PD-MCI / dementia categorisation, APOE
  P5  single-protein and published-panel comparators (NEFL, GFAP, TPK1, ...)
  P6  plasma vs CSF overlap (other tissue configured under proteomics_panels)
  P7  genetic and sex subgroups

Inputs: the aligned clinical table and proteomic matrices of a completed run plus
the PPMI-native exports (``ppmi_clinical_dir``): SAA results, the curated data
cut (NSD status / stage, education, death), MoCA, cognitive categorization,
genetics (``ppmi_clinical.load_ppmi_extras``).

Outputs (out_dir): tables/path_feasibility.csv (every number),
tables/path_exploration_summary.csv (one row per path), PATH_EXPLORATION.md,
figures/fig_path_exploration.png.

Standalone (re-uses the cached proteomic matrix of a finished run):
    python -m pd_strat.explore_paths --config config_ht.yaml --out_dir results_ht \
        --ppmi_clinical_dir S:/PPMI_clinical
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (TAB, OUT, SEED, PPMI_CLINICAL_DIR, PRODROMAL_HORIZONS,
                     EXTRA_BIOMARKERS, PROTEOMICS_TISSUES)
from .utils import map_participant_id, summary_update
from .features import primary_feature_builder
from .validity import _fit_predict
from .progression import _months, _harrell_c
from .prodromal import _horizon_label

warnings.filterwarnings("ignore")

_Z80 = 1.96 + 0.8416            # two-sided α = 0.05, power 0.80
_MIN_EVENTS = 30                # below this a time-to-event / binary path is "underpowered on disk"
_MIN_N = 60                     # below this a correlation path is "underpowered on disk"

# gene -> (UniProt, published direction for PD risk / severity, source).  Direction 0 =
# reported without a usable sign; such proteins are checked individually, never pooled.
_COMPARATORS: Dict[str, Tuple[str, int, str]] = {
    "NEFL":    ("P07196", +1, "UKB NEFL/TPK1 2025; incumbent blood progression marker"),
    "GFAP":    ("P14136", +1, "astroglial comparator"),
    "TPK1":    ("Q9H3S4", -1, "UKB 2025"),
    "MERTK":   ("Q12866", +1, "UKB 2025"),
    "ITGAV":   ("P06756", -1, "UKB 2025; AD/PD 2026"),
    "ITGAM":   ("P11215", -1, "UKB 2025; AD/PD 2026"),
    "BAG3":    ("O95817", -1, "UKB 2025"),
    "CLEC10A": ("Q8IUN9", -1, "UKB 2025"),
    "HNMT":    ("P50135", -1, "UKB 2025"),
    "DDC":     ("P20711", +1, "severity / levodopa-responsive (prior work)"),
    "GPNMB":   ("Q14956", +1, "cis-pQTL MR risk protein"),
    "TPPP2":   ("P59282", 0, "pre-diagnostic 17-protein set (2025)"),
    "HPGDS":   ("O60760", 0, "pre-diagnostic 17-protein set (2025)"),
    "ALPL":    ("P05186", 0, "pre-diagnostic 17-protein set (2025)"),
    "MFAP5":   ("Q13361", 0, "pre-diagnostic 17-protein set (2025)"),
    "OGFR":    ("Q9NZT2", 0, "pre-diagnostic 17-protein set (2025)"),
    "ACAD8":   ("Q9UKU7", 0, "pre-diagnostic 17-protein set (2025)"),
    "TCL1A":   ("P56279", 0, "pre-diagnostic 17-protein set (2025)"),
    "GPC4":    ("O75487", 0, "pre-diagnostic 17-protein set (2025)"),
    "GSTA3":   ("Q16772", 0, "pre-diagnostic 17-protein set (2025)"),
    "LCN2":    ("P80188", 0, "pre-diagnostic 17-protein set (2025)"),
    "KRAS":    ("P01116", 0, "pre-diagnostic 17-protein set (2025)"),
    "GJA1":    ("P17302", 0, "pre-diagnostic 17-protein set (2025)"),
}
_UKB_PANEL = ["NEFL", "TPK1", "MERTK", "ITGAV", "ITGAM", "BAG3", "CLEC10A", "HNMT"]


# ── power and effect helpers ────────────────────────────────────────────────

def mde_rho(n: int) -> float:
    """Smallest |ρ| detectable at 80 % power with n pairs (Fisher z)."""
    return float(np.tanh(_Z80 / np.sqrt(n - 3))) if n > 6 else np.nan


def mde_hr(events: int) -> float:
    """Smallest HR per SD detectable at 80 % power with `events` events (Schoenfeld)."""
    return float(np.exp(_Z80 / np.sqrt(events))) if events > 0 else np.nan


def mde_auroc(n1: int, n0: int) -> float:
    """Smallest AUROC (vs 0.5) detectable at 80 % power (null SE of the Mann-Whitney U)."""
    if n1 < 1 or n0 < 1:
        return np.nan
    return float(0.5 + _Z80 * np.sqrt((n1 + n0 + 1) / (12.0 * n1 * n0)))


def _auroc(y, s, B: int = 400) -> Tuple[float, float, float, int, int]:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, float); s = np.asarray(s, float)
    ok = np.isfinite(y) & np.isfinite(s)
    y, s = y[ok], s[ok]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 < 3 or n0 < 3:
        return np.nan, np.nan, np.nan, n1, n0
    a = float(roc_auc_score(y, s))
    rng = np.random.default_rng(SEED)
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    bs = []
    for _ in range(B):
        idx = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])
        bs.append(roc_auc_score(y[idx], s[idx]))
    return a, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), n1, n0


def _rho(x, y) -> Tuple[float, float, float, int]:
    from scipy.stats import spearmanr
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 6:
        return np.nan, np.nan, np.nan, n
    r = float(spearmanr(x[ok], y[ok]).correlation)
    if not np.isfinite(r):
        return np.nan, np.nan, np.nan, n
    z, se = np.arctanh(np.clip(r, -0.999, 0.999)), 1.0 / np.sqrt(n - 3)
    return r, float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se)), n


def _resid(v, covs: List[np.ndarray]) -> np.ndarray:
    """Residual of v on [1, covs] over rows where everything is finite (NaN elsewhere)."""
    v = np.asarray(v, float)
    C = np.column_stack([np.ones(len(v))] + [np.asarray(c, float) for c in covs])
    ok = np.isfinite(v) & np.all(np.isfinite(C), axis=1)
    out = np.full(len(v), np.nan)
    if ok.sum() > C.shape[1] + 2:
        beta, *_ = np.linalg.lstsq(C[ok], v[ok], rcond=None)
        out[ok] = v[ok] - C[ok] @ beta
    return out


def _cox(df: pd.DataFrame, cols: List[str]) -> Dict[str, Any]:
    """HR per SD of cols[0] (binary columns left as is), adjusted for the rest."""
    from statsmodels.duration.hazard_regression import PHReg
    sub = df.dropna(subset=cols + ["months", "event"])
    sub = sub[sub["months"] > 0]
    n, ev = int(len(sub)), int(sub["event"].sum())
    if n < 25 or ev < 8:
        return {"n": n, "events": ev, "note": "insufficient"}
    X = pd.DataFrame({c: (sub[c] if sub[c].nunique() <= 2 else (sub[c] - sub[c].mean()) / (sub[c].std(ddof=1) or 1.0))
                      for c in cols}).fillna(0.0)
    try:
        res = PHReg(sub["months"].values, X.values, status=sub["event"].values, ties="efron").fit()
    except Exception as e:                      # noqa: BLE001
        return {"n": n, "events": ev, "note": str(e)[:80]}
    ci = res.conf_int()
    return {"n": n, "events": ev, "hr": float(np.exp(res.params[0])), "lo": float(np.exp(ci[0][0])),
            "hi": float(np.exp(ci[0][1])), "p": float(res.pvalues[0]),
            "c": _harrell_c(sub["months"].values, sub["event"].values, X.values @ res.params)}


def _slope(months: np.ndarray, values: np.ndarray, min_span: float = 6.0) -> float:
    ok = np.isfinite(months) & np.isfinite(values)
    if ok.sum() < 2 or (months[ok].max() - months[ok].min()) < min_span:
        return np.nan
    return float(np.polyfit(months[ok] / 12.0, values[ok], 1)[0])


# ── bookkeeping ─────────────────────────────────────────────────────────────

class _Rows:
    def __init__(self):
        self.rows: List[Dict[str, Any]] = []

    def add(self, path: str, item: str, scope: str, metric: str, value=np.nan, lo=np.nan, hi=np.nan,
            n=np.nan, events=np.nan, note: str = ""):
        self.rows.append({"path": path, "item": item, "scope": scope, "metric": metric,
                          "value": value, "ci_lo": lo, "ci_hi": hi, "n": n, "events": events, "note": note})

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def _fmt(v, nd=2) -> str:
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def _ci(lo, hi, nd=2) -> str:
    return "" if not (np.isfinite(lo) and np.isfinite(hi)) else f" [{lo:.{nd}f}, {hi:.{nd}f}]"


def _verdict(data_ok: bool, powered: bool, lo: float, hi: float, null: float) -> str:
    if not data_ok:
        return "needs data"
    if not powered:
        return "underpowered on disk"
    if np.isfinite(lo) and np.isfinite(hi):
        return "feasible; exploratory signal" if (lo > null or hi < null) else "feasible; no exploratory signal"
    return "feasible; no signal computed"


_NULL = {"auroc": 0.5, "rho": 0.0, "hr": 1.0}


def _summary_row(path, question, data, n_all, ev_all, n_train, ev_train, mde, mde_unit, signal, lo, hi, kind,
                 powered, verdict=None, next_step="", signal_text="") -> Dict[str, Any]:
    """One summary row; `kind` names the headline statistic (auroc / rho / hr / text / none)."""
    null = _NULL.get(kind, np.nan)
    return {"path": path, "question": question, "data_on_disk": data, "n_all": n_all, "events_all": ev_all,
            "n_train": n_train, "events_train": ev_train, "mde_80pct": mde, "mde_unit": mde_unit,
            "signal_kind": kind, "exploratory_signal": signal, "signal_lo": lo, "signal_hi": hi, "null": null,
            "signal_text": signal_text,
            "verdict": verdict or _verdict(data != "no", powered, lo, hi, null), "next_step": next_step}


def _signal_str(r) -> str:
    if isinstance(r.get("signal_text"), str) and r["signal_text"]:
        return r["signal_text"]
    e = r["exploratory_signal"]
    if not (isinstance(e, (int, float)) and np.isfinite(e)):
        return "—"
    lab = {"auroc": "AUROC", "rho": "ρ", "hr": "HR/SD"}.get(r.get("signal_kind"), "")
    return f"{lab} {_fmt(e)}{_ci(r['signal_lo'], r['signal_hi'])}".strip()


def _mde_str(r) -> str:
    m = r["mde_80pct"]
    return "—" if not (isinstance(m, (int, float)) and np.isfinite(m)) else f"{r.get('mde_unit', '')} {_fmt(m)}".strip()


# ── context ─────────────────────────────────────────────────────────────────

def _context(clin, X, M, y_all, tv, cohort, PANEL_COL_INDICES, USE_PANEL_AWARE, PRIMARY_LABEL,
             prot_cols, extras) -> Dict[str, Any]:
    pids = np.array([map_participant_id(str(x)) for x in clin.index])
    months = _months(clin)
    is_train = cohort["is_train"].values.astype(bool)
    is_test = cohort["is_test"].values.astype(bool)
    is_pd = cohort["is_pd_flag"].values.astype(bool)
    is_hc = cohort["is_control"].values.astype(bool)
    has = (M.sum(1) > 0) & np.isfinite(months)

    conv = None
    p = TAB / "ppmi_prodromal_conversion.csv"
    if p.exists():
        conv = pd.read_csv(p)
        conv["participant_id"] = [map_participant_id(str(x)) for x in conv["participant_id"]]
    pro_ids = set(conv["participant_id"]) if conv is not None else set()
    is_pro = np.isin(pids, list(pro_ids)) & ~is_pd & ~is_hc

    group = np.where(is_pd, "PD", np.where(is_hc, "HC", np.where(is_pro, "PRODROMAL", "OTHER")))
    split = np.where(is_test, "TEST", np.where(is_train, "TRAIN", "OTHER"))
    rows = pd.DataFrame({"pos": np.arange(len(pids)), "participant_id": pids, "month": months,
                         "group": group, "split": split})[has]
    base = rows.sort_values(["participant_id", "month"]).drop_duplicates("participant_id").reset_index(drop=True)

    pro_pos = rows.loc[rows["group"] == "PRODROMAL", "pos"].values
    print(f"  scoring: {len(tv['train_idx_y'])} TRAIN PD/HC rows out-of-fold; {len(pro_pos)} prodromal rows "
          f"transported (fitted on TRAIN PD + HC); holdout rows are never scored")
    bff = primary_feature_builder(PANEL_COL_INDICES, USE_PANEL_AWARE, PRIMARY_LABEL)
    tr, groups, gkf = tv["train_idx_y"], tv["groups_train"], tv["gkf"]
    oof = np.full(X.shape[0], np.nan, dtype=np.float32)
    for a, b in gkf.split(tr, tv["y_tr_nonan"], groups):
        oof[tr[b]] = _fit_predict(bff, X, M, y_all, tr[a], tr[b])
    trans = np.full(X.shape[0], np.nan, dtype=np.float32)
    if len(pro_pos):
        trans[pro_pos] = _fit_predict(bff, X, M, y_all, tr, pro_pos)
    score = np.where(np.isfinite(oof), oof, trans)          # TRAIN PD/HC: OOF; prodromal: transported
    score[is_test] = np.nan                                  # holdout stays unscored
    rows["score"] = score[rows["pos"].values]
    base["score"] = score[base["pos"].values]
    base["y"] = y_all[base["pos"].values]

    def _clin(col, pos):
        return pd.to_numeric(clin[col].iloc[pos], errors="coerce").values if col in clin.columns else np.full(len(pos), np.nan)

    base["age"] = _clin("_age", base["pos"].values) if "_age" in clin.columns else _clin("age", base["pos"].values)
    sexcol = "_sex" if "_sex" in clin.columns else ("sex" if "sex" in clin.columns else None)
    base["male"] = (clin[sexcol].iloc[base["pos"].values].astype(str).str.upper().str.startswith("M").astype(float).values
                    if sexcol else np.nan)
    base["caudate"] = _clin("datscan_caudate_baseline", base["pos"].values)
    if not np.isfinite(base["caudate"]).any():
        base["caudate"] = _clin("datscan_caudate", base["pos"].values)

    return {"clin": clin, "X": X, "M": M, "y_all": y_all, "pids": pids, "months": months, "has": has,
            "is_train": is_train, "is_test": is_test, "is_pd": is_pd, "is_hc": is_hc, "is_pro": is_pro,
            "rows": rows, "base": base, "score": score, "conv": conv, "extras": extras, "prot_cols": list(prot_cols),
            "horizon": float(PRODROMAL_HORIZONS[0]) if PRODROMAL_HORIZONS else 36.0}


def _prodromal_frame(ctx) -> Optional[pd.DataFrame]:
    """One row per prodromal participant with a sample: time from the sample (as in §16)."""
    conv, base = ctx["conv"], ctx["base"]
    if conv is None:
        return None
    c = conv[~conv["prevalent_at_baseline"].astype(bool)].dropna(subset=["months"])
    df = base[base["group"] == "PRODROMAL"].merge(c, on="participant_id", how="inner")
    df["months"] = df["months"] - df["month"]
    pre = (df["event"] == 1) & (df["months"] <= 0)
    df = df[~pre & (df["months"] > 0)].copy()
    for c_ in ("enrlrbd", "enrlhpsm", "enrllrrk2", "enrlgba", "datscan_visual_baseline", "converted_code"):
        if c_ not in df.columns:
            df[c_] = np.nan
    return df


# ── P1  SAA / NSD ───────────────────────────────────────────────────────────

def _p1_saa(ctx, R: _Rows) -> Dict[str, Any]:
    P = "P1 plasma -> SAA / NSD status"
    q = "Does the plasma score separate α-synuclein SAA+ from SAA- (and track NSD-ISS stage)?"
    ex, base = ctx["extras"], ctx["base"].copy()
    saa, cur = ex.get("saa"), ex.get("curated")
    have_nsd = cur is not None and "nsd_status" in cur.columns and cur["nsd_status"].notna().any()
    if saa is None and not have_nsd:
        return _summary_row(P, q, "no", 0, 0, 0, 0, np.nan, "AUROC", np.nan, np.nan, np.nan, "none", False,
                            next_step="download 'SAA Biospecimen Analysis Results' (Biospecimen > Biospecimen Analysis; "
                                      "PD only on LONI, HC/prodromal by request to ppmi@michaeljfox.org) and the Curated Data Cut")
    if saa is not None:
        base = base.merge(saa[[c for c in ("participant_id", "saa_status", "saa_type", "saa_fmax", "saa_ttt", "saa_cohort")
                               if c in saa.columns]], on="participant_id", how="left")
    if cur is not None:
        base = base.merge(cur[[c for c in ("participant_id", "nsd_status", "nsd_stage", "nsd_stage_label") if c in cur.columns]],
                          on="participant_id", how="left")
    for c in ("saa_status", "saa_fmax", "saa_ttt", "nsd_status", "nsd_stage"):
        if c not in base.columns:
            base[c] = np.nan

    # coverage
    cov = {}
    for g in ("PD", "HC", "PRODROMAL"):
        for sp in ("TRAIN", "TEST"):
            b = base[(base["group"] == g) & (base["split"] == sp)]
            n_s, n_pos = int(b["saa_status"].notna().sum()), int((b["saa_status"] == 1).sum())
            n_n = int(b["nsd_status"].notna().sum())
            cov[(g, sp)] = (n_s, n_pos, n_n)
            R.add(P, f"{g}: SAA status available", sp, "n", n_s, n=len(b), events=n_pos,
                  note=f"{n_pos} positive, {n_s - n_pos} negative")
            if n_n:
                R.add(P, f"{g}: NSD status available", sp, "n", n_n, n=len(b), events=int((b['nsd_status'] == 1).sum()))
    n_pd = sum(cov[("PD", s)][0] for s in ("TRAIN", "TEST"))
    n_pro = sum(cov[("PRODROMAL", s)][0] for s in ("TRAIN", "TEST"))
    n_hc = sum(cov[("HC", s)][0] for s in ("TRAIN", "TEST"))
    seq = saa is not None and n_pd > 0 and n_pro == 0 and n_hc == 0
    if seq:
        R.add(P, "HC / prodromal SAA results", "ALL", "note", note="none on disk: LONI serves PD only; request the sequestered "
              "HC + prodromal results from ppmi@michaeljfox.org (this is where the discrimination lives)")
    if "nsd_stage_label" in base.columns and base["nsd_stage_label"].notna().any():
        for lab, n in base["nsd_stage_label"].value_counts().items():
            R.add(P, f"NSD-ISS {lab}", "ALL", "n", int(n))

    # exploratory signals (TRAIN only)
    t = base[base["split"] == "TRAIN"]
    best = (np.nan, np.nan, np.nan, "none", "none")
    for g, label in (("PRODROMAL", "prodromal: AUROC score for SAA+"), ("PD", "PD: AUROC score for SAA+"),
                     ("HC", "HC: AUROC score for SAA+")):
        b = t[t["group"] == g]
        a, lo, hi, n1, n0 = _auroc(b["saa_status"], b["score"])
        if n1 + n0:
            R.add(P, label, "TRAIN", "auroc", a, lo, hi, n=n1 + n0, events=n1,
                  note=f"MDE AUROC {_fmt(mde_auroc(n1, n0))} at this n")
        if np.isfinite(a) and (best[3] == "none" or g == "PRODROMAL"):
            best = (a, lo, hi, label, "auroc")
    b = t[t["group"].isin(["PD", "PRODROMAL"])]
    r, lo, hi, n = _rho(b["score"], b["nsd_stage"])
    if n >= 6:
        R.add(P, "PD + prodromal: Spearman score vs NSD-ISS stage (ordinal)", "TRAIN", "rho", r, lo, hi, n=n,
              note=f"MDE |rho| {_fmt(mde_rho(n))}")
        if best[3] == "none" or best[3].startswith("PD") or best[3].startswith("HC"):
            best = (r, lo, hi, "score vs NSD-ISS stage (rho)", "rho")
    for g in ("PRODROMAL", "PD"):
        b = t[t["group"] == g]
        a, lo2, hi2, n1, n0 = _auroc(b["nsd_status"], b["score"])
        if n1 + n0:
            R.add(P, f"{g}: AUROC score for NSD+", "TRAIN", "auroc", a, lo2, hi2, n=n1 + n0, events=n1)
    for col, lab in (("saa_fmax", "Fmax"), ("saa_ttt", "time to threshold")):
        for g in ("PD", "PRODROMAL"):
            b = t[t["group"] == g]
            r, lo2, hi2, n = _rho(b["score"], b[col])
            if n >= 6:
                R.add(P, f"{g}: Spearman score vs SAA {lab}", "TRAIN", "rho", r, lo2, hi2, n=n)

    n_all = int(base["saa_status"].notna().sum()) or int(base["nsd_status"].notna().sum())
    ev_all = int((base["saa_status"] == 1).sum()) or int((base["nsd_status"] == 1).sum())
    n_tr = int(t["saa_status"].notna().sum()) or int(t["nsd_status"].notna().sum())
    ev_tr = int((t["saa_status"] == 1).sum()) or int((t["nsd_status"] == 1).sum())
    # the informative analysis needs both classes; PD alone is ~90 % positive
    pro = base[base["group"] == "PRODROMAL"]
    n1, n0 = int((pro["saa_status"] == 1).sum()), int((pro["saa_status"] == 0).sum())
    if n1 + n0 == 0:
        pdb = base[base["group"] == "PD"]
        n1, n0 = int((pdb["saa_status"] == 1).sum()), int((pdb["saa_status"] == 0).sum())
    mde = mde_auroc(n1, n0)
    powered = min(n1, n0) >= _MIN_EVENTS
    nxt = ("request HC + prodromal SAA results; add 'saa' reader output to assembly and SAA / NSD targets to discovery"
           if seq else "add SAA / NSD targets to discovery with the clinical (Venuto-style) comparator; TEST once")
    return _summary_row(P, q, "partial" if seq else "yes", n_all, ev_all, n_tr, ev_tr, mde, "AUROC", best[0], best[1],
                        best[2], best[4], powered, next_step=nxt)


# ── P2  prodromal arms ──────────────────────────────────────────────────────

def _p2_prodromal(ctx, R: _Rows) -> Dict[str, Any]:
    P = "P2 prodromal conversion: arms and endpoints"
    q = "Where does the conversion hazard come from (RBD / hyposmia / carriers, DaT-negative), and are there enough events?"
    df = _prodromal_frame(ctx)
    if df is None or len(df) < 20:
        return _summary_row(P, q, "no", 0, 0, 0, 0, np.nan, "HR/SD", np.nan, np.nan, np.nan, "none", False,
                            next_step="set ppmi_clinical_dir and rebuild with --force_assembly (prodromal conversion table)")
    h = ctx["horizon"]
    for sp in ("ALL", "TRAIN", "TEST"):
        d = df if sp == "ALL" else df[df["split"] == sp]
        R.add(P, "prodromal with a sample", sp, "n", len(d), n=len(d), events=int(d["event"].sum()),
              note=f"median follow-up {d.loc[d['event'] == 0, 'months'].median():.0f} m; MDE HR/SD {_fmt(mde_hr(int(d['event'].sum())))}")
    for col, lab in (("enrlrbd", "RBD arm"), ("enrlhpsm", "hyposmia arm"), ("enrllrrk2", "LRRK2 carriers"),
                     ("enrlgba", "GBA carriers")):
        d = df[df[col] == 1]
        if len(d):
            R.add(P, lab, "ALL", "n", len(d), n=len(d), events=int(d["event"].sum()),
                  note=f"MDE HR/SD {_fmt(mde_hr(int(d['event'].sum())))}")
    if df["converted_code"].notna().any():
        for code, n in df.loc[df["event"] == 1, "converted_code"].value_counts().items():
            R.add(P, f"conversion diagnosis code {int(code)}", "ALL", "events", int(n),
                  note="1 = idiopathic PD; other codes only if listed in ppmi_conversion_codes")
    neg = df[df["datscan_visual_baseline"] == 0]
    if len(neg):
        R.add(P, "DaT visual read negative at baseline", "ALL", "n", len(neg), n=len(neg), events=int(neg["event"].sum()),
              note="the subgroup where imaging is uninformative; MDE HR/SD "
                   f"{_fmt(mde_hr(int(neg['event'].sum())))}")
    cur = ctx["extras"].get("curated")
    if cur is not None and "death_status" in cur.columns:
        dd = df.merge(cur[["participant_id", "death_status"]], on="participant_id", how="left")
        R.add(P, "deaths among prodromal (competing risk)", "ALL", "events", int((dd["death_status"] == 1).sum()))

    t = df[df["split"] == "TRAIN"]
    y = _horizon_label(t, h)
    a, lo, hi, n1, n0 = _auroc(y, t["score"])
    R.add(P, f"all arms: AUROC score, conversion within {h:.0f} m", "TRAIN", "auroc", a, lo, hi, n=n1 + n0, events=n1,
          note=f"MDE AUROC {_fmt(mde_auroc(n1, n0))}")
    for col, lab in (("enrlrbd", "RBD arm"), ("enrlhpsm", "hyposmia arm"), ("enrllrrk2", "LRRK2 carriers"), ("enrlgba", "GBA carriers")):
        s = t[t[col] == 1]
        if len(s) >= 15:
            ys = _horizon_label(s, h)
            a2, lo2, hi2, n1s, n0s = _auroc(ys, s["score"])
            if n1s >= 3 and n0s >= 3:
                R.add(P, f"{lab}: AUROC score, conversion within {h:.0f} m", "TRAIN", "auroc", a2, lo2, hi2, n=n1s + n0s, events=n1s)
    s = t[t["datscan_visual_baseline"] == 0]
    if len(s) >= 15:
        ys = _horizon_label(s, h)
        a2, lo2, hi2, n1s, n0s = _auroc(ys, s["score"])
        if n1s >= 3 and n0s >= 3:
            R.add(P, f"DaT-negative: AUROC score, conversion within {h:.0f} m", "TRAIN", "auroc", a2, lo2, hi2, n=n1s + n0s, events=n1s)
    cx = _cox(t, ["score"])
    if "hr" in cx:
        R.add(P, "Cox HR per SD score (unadjusted)", "TRAIN", "hr_per_sd", cx["hr"], cx["lo"], cx["hi"], n=cx["n"], events=cx["events"],
              note=f"C = {cx['c']:.3f}")
    ev_all = int(df["event"].sum())
    return _summary_row(P, q, "yes", len(df), ev_all, len(t), int(t["event"].sum()), mde_hr(ev_all), "HR/SD", a, lo, hi,
                        "auroc", ev_all >= _MIN_EVENTS,
                        next_step="after the tiered run: endpoint variants (any synucleinopathy), arm-stratified Cox, "
                                  "competing risk of death, decision curve at 36 / 60 m")


# ── P3  dynamic prodromal ───────────────────────────────────────────────────

def _p3_dynamic(ctx, R: _Rows) -> Dict[str, Any]:
    P = "P3 dynamic prodromal proteomics"
    q = "Are there enough serial pre-conversion samples to model the score's trajectory (ALS-style)?"
    conv, rows = ctx["conv"], ctx["rows"]
    if conv is None:
        return _summary_row(P, q, "no", 0, 0, 0, 0, np.nan, "AUROC", np.nan, np.nan, np.nan, "none", False,
                            next_step="needs the prodromal conversion table (ppmi_clinical_dir)")
    c = conv[~conv["prevalent_at_baseline"].astype(bool)].dropna(subset=["months"])
    pro = rows[rows["group"] == "PRODROMAL"].merge(c[["participant_id", "event", "months"]], on="participant_id", how="inner")
    pro = pro.rename(columns={"months": "event_month"})
    pre = pro[(pro["event"] == 0) | (pro["month"] < pro["event_month"])]      # samples before conversion
    per = (pre.groupby("participant_id")
              .agg(n_samples=("month", "size"), span=("month", lambda m: m.max() - m.min()),
                   event=("event", "first"), split=("split", "first"), first=("month", "min"))
              .reset_index())
    multi = per[per["n_samples"] >= 2]
    R.add(P, "prodromal with >= 2 pre-conversion samples", "ALL", "n", len(multi), n=len(multi), events=int(multi["event"].sum()),
          note=f"median span {multi['span'].median():.0f} m" if len(multi) else "")
    R.add(P, "converters with >= 2 pre-conversion samples", "ALL", "n", int(multi["event"].sum()))
    R.add(P, "prodromal with >= 3 pre-conversion samples", "ALL", "n", int((per["n_samples"] >= 3).sum()))
    for sp in ("TRAIN", "TEST"):
        m = multi[multi["split"] == sp]
        R.add(P, "prodromal with >= 2 pre-conversion samples", sp, "n", len(m), n=len(m), events=int(m["event"].sum()))

    # exploratory: slope of the transported score (TRAIN) vs conversion
    t = pre[(pre["split"] == "TRAIN") & np.isfinite(pre["score"])]
    sl = (t.groupby("participant_id")
            .apply(lambda g: pd.Series({"slope": _slope(g["month"].values, g["score"].values),
                                        "baseline": g.sort_values("month")["score"].iloc[0],
                                        "event": g["event"].iloc[0], "months": g["event_month"].iloc[0] - g["month"].min()}))
            .reset_index()) if len(t) else pd.DataFrame(columns=["slope", "baseline", "event", "months"])
    sl = sl.dropna(subset=["slope"]) if len(sl) else sl
    a = lo = hi = np.nan
    n1 = n0 = 0
    if len(sl) >= 10:
        a, lo, hi, n1, n0 = _auroc(sl["event"], sl["slope"])
        R.add(P, "AUROC score slope (per year) for conversion", "TRAIN", "auroc", a, lo, hi, n=n1 + n0, events=n1,
              note=f"MDE AUROC {_fmt(mde_auroc(n1, n0))}")
        a2, lo2, hi2, _, _ = _auroc(sl["event"], sl["baseline"])
        R.add(P, "AUROC baseline score in the same participants", "TRAIN", "auroc", a2, lo2, hi2, n=n1 + n0, events=n1)
        cx = _cox(sl, ["slope", "baseline"])
        if "hr" in cx:
            R.add(P, "Cox HR per SD of slope, adjusted for baseline score", "TRAIN", "hr_per_sd", cx["hr"], cx["lo"], cx["hi"],
                  n=cx["n"], events=cx["events"])
    ev_all = int(multi["event"].sum())
    n_conv, n_non = ev_all, int(len(multi) - ev_all)
    return _summary_row(P, q, "yes" if len(multi) else "no", len(multi), ev_all, len(sl), int(sl["event"].sum()) if len(sl) else 0,
                        mde_auroc(n_conv, n_non), "AUROC", a, lo, hi, "auroc", ev_all >= _MIN_EVENTS,
                        next_step="landmark Cox with the time-updated score; years-to-conversion regression "
                                  "(needs >= 30 converters with serial samples)")


# ── P4  cognition ───────────────────────────────────────────────────────────

def _p4_cognition(ctx, R: _Rows) -> Dict[str, Any]:
    P = "P4 cognition: MoCA, PD-MCI / dementia"
    q = "Does the plasma score predict cognitive decline, PD-MCI or dementia (outcomes UPDRS cannot address)?"
    ex, base = ctx["extras"], ctx["base"]
    moca, cog, gen, cur = ex.get("moca"), ex.get("cogcat"), ex.get("genetics"), ex.get("curated")
    if moca is None and cog is None:
        return _summary_row(P, q, "no", 0, 0, 0, 0, np.nan, "HR/SD", np.nan, np.nan, np.nan, "none", False,
                            next_step="download Montreal_Cognitive_Assessment and Cognitive_Categorization "
                                      "(Non-motor Assessments) plus iu_genetic_consensus (APOE)")
    pdb = base[base["group"] == "PD"].copy()
    rec: List[Dict[str, Any]] = []
    for _, r in pdb.iterrows():
        pid, m0 = r["participant_id"], r["month"]
        d = {"participant_id": pid, "split": r["split"], "score": r["score"], "age": r["age"], "y": r["y"]}
        if moca is not None:
            mv = moca[moca["participant_id"] == pid].dropna(subset=["visit_month"]).sort_values("visit_month")
            near = mv[(mv["visit_month"] - m0).abs() <= 6]
            d["moca_bl"] = near.iloc[(near["visit_month"] - m0).abs().argmin()]["moca"] if len(near) else np.nan
            fu = mv[mv["visit_month"] >= m0 - 6]
            d["moca_visits"] = len(fu)
            d["moca_span"] = (fu["visit_month"].max() - fu["visit_month"].min()) if len(fu) else np.nan
            d["moca_slope"] = _slope(fu["visit_month"].values, fu["moca"].values, min_span=12.0) if len(fu) >= 2 else np.nan
            after = mv[mv["visit_month"] > m0]
            if np.isfinite(d["moca_bl"]) and d["moca_bl"] >= 21 and len(after):
                hit = after[after["moca"] < 21]
                d["moca21_event"] = float(len(hit) > 0)
                d["moca21_months"] = (hit["visit_month"].iloc[0] if len(hit) else after["visit_month"].max()) - m0
        if cog is not None:
            cv = cog[cog["participant_id"] == pid].dropna(subset=["visit_month"]).sort_values("visit_month")
            near = cv[(cv["visit_month"] - m0).abs() <= 6]
            d["cog_bl"] = near.iloc[(near["visit_month"] - m0).abs().argmin()]["cogstate"] if len(near) else np.nan
            after = cv[cv["visit_month"] > m0]
            if d["cog_bl"] == 1 and len(after):
                hit = after[after["cogstate"] >= 2]
                d["mci_event"] = float(len(hit) > 0)
                d["mci_months"] = (hit["visit_month"].iloc[0] if len(hit) else after["visit_month"].max()) - m0
            if np.isfinite(d.get("cog_bl", np.nan)) and d["cog_bl"] < 3 and len(after):
                hit = after[after["cogstate"] == 3]
                d["dem_event"] = float(len(hit) > 0)
                d["dem_months"] = (hit["visit_month"].iloc[0] if len(hit) else after["visit_month"].max()) - m0
        rec.append(d)
    df = pd.DataFrame(rec)
    for c in ("moca_bl", "moca_slope", "moca21_event", "moca21_months", "cog_bl", "mci_event", "mci_months",
              "dem_event", "dem_months", "moca_visits"):
        if c not in df.columns:
            df[c] = np.nan
    if gen is not None and "apoe_e4_count" in gen.columns:
        df = df.merge(gen[["participant_id", "apoe_e4_count"]], on="participant_id", how="left")
    else:
        df["apoe_e4_count"] = np.nan
    if cur is not None and "educyrs" in cur.columns:
        df = df.merge(cur[["participant_id", "educyrs"]], on="participant_id", how="left")
    else:
        df["educyrs"] = np.nan

    for sp in ("ALL", "TRAIN", "TEST"):
        d = df if sp == "ALL" else df[df["split"] == sp]
        R.add(P, "PD with baseline MoCA (±6 m of the sample)", sp, "n", int(d["moca_bl"].notna().sum()), n=len(d))
        R.add(P, "PD with a MoCA slope (>= 2 visits, >= 12 m)", sp, "n", int(d["moca_slope"].notna().sum()))
        R.add(P, "PD cognitively normal at baseline with follow-up categorisation", sp, "n",
              int(d["mci_event"].notna().sum()), events=int((d["mci_event"] == 1).sum()),
              note=f"MDE HR/SD {_fmt(mde_hr(int((d['mci_event'] == 1).sum())))}")
        R.add(P, "PD not demented at baseline with follow-up categorisation", sp, "n",
              int(d["dem_event"].notna().sum()), events=int((d["dem_event"] == 1).sum()))
        R.add(P, "PD with baseline MoCA >= 21 and later MoCA", sp, "n", int(d["moca21_event"].notna().sum()),
              events=int((d["moca21_event"] == 1).sum()),
              note=f"MoCA < 21 events; MDE HR/SD {_fmt(mde_hr(int((d['moca21_event'] == 1).sum())))}")
    R.add(P, "PD with APOE genotype", "ALL", "n", int(df["apoe_e4_count"].notna().sum()),
          events=int((df["apoe_e4_count"] >= 1).sum()), note="events = e4 carriers")
    R.add(P, "PD with years of education", "ALL", "n", int(df["educyrs"].notna().sum()))

    t = df[df["split"] == "TRAIN"]
    best = (np.nan, np.nan, np.nan, "none")
    r, lo, hi, n = _rho(t["score"], t["moca_bl"])
    if n >= 6:
        R.add(P, "Spearman score vs baseline MoCA", "TRAIN", "rho", r, lo, hi, n=n, note=f"MDE |rho| {_fmt(mde_rho(n))}")
        rp, lop, hip, npar = _rho(_resid(t["score"], [t["age"]]), _resid(t["moca_bl"], [t["age"]]))
        if npar >= 6:
            R.add(P, "partial Spearman score vs baseline MoCA | age", "TRAIN", "rho", rp, lop, hip, n=npar)
        best = (r, lo, hi, "rho")
    r, lo, hi, n = _rho(t["score"], t["moca_slope"])
    if n >= 6:
        R.add(P, "Spearman score vs MoCA slope (points / year)", "TRAIN", "rho", r, lo, hi, n=n, note=f"MDE |rho| {_fmt(mde_rho(n))}")
    for ev, mo, lab in (("mci_event", "mci_months", "PD-MCI or dementia"), ("dem_event", "dem_months", "dementia"),
                        ("moca21_event", "moca21_months", "MoCA < 21")):
        d = t.rename(columns={ev: "event", mo: "months"})
        cx = _cox(d, ["score"])
        if "hr" in cx:
            R.add(P, f"Cox HR per SD score -> {lab}", "TRAIN", "hr_per_sd", cx["hr"], cx["lo"], cx["hi"], n=cx["n"], events=cx["events"],
                  note=f"C = {cx['c']:.3f}")
            cx2 = _cox(d, ["score", "age", "moca_bl"]) if d["moca_bl"].notna().sum() > 25 else {}
            if "hr" in cx2:
                R.add(P, f"Cox HR per SD score -> {lab} | age, baseline MoCA", "TRAIN", "hr_per_sd", cx2["hr"], cx2["lo"], cx2["hi"],
                      n=cx2["n"], events=cx2["events"])
            yh = _horizon_label(d, ctx["horizon"])
            a, alo, ahi, n1, n0 = _auroc(yh, d["score"])
            if n1 >= 3 and n0 >= 3:
                R.add(P, f"AUROC score, {lab} within {ctx['horizon']:.0f} m", "TRAIN", "auroc", a, alo, ahi, n=n1 + n0, events=n1)
                if lab == "PD-MCI or dementia":
                    best = (a, alo, ahi, "auroc")           # the event-based headline beats baseline MoCA

    ev_all = int((df["mci_event"] == 1).sum()) or int((df["moca21_event"] == 1).sum())
    n_all = int(df["moca_bl"].notna().sum()) or int(df["cog_bl"].notna().sum())
    powered = ev_all >= _MIN_EVENTS or (ev_all == 0 and n_all >= _MIN_N)
    mde, unit = (mde_hr(ev_all), "HR/SD") if ev_all else (mde_rho(n_all), "|ρ|")
    return _summary_row(P, q, "yes", n_all, ev_all, int(t["moca_bl"].notna().sum()), int((t["mci_event"] == 1).sum()),
                        mde, unit, best[0], best[1], best[2], best[3], powered,
                        next_step="add MoCA / COGSTATE readers to assembly, cognitive milestones to time_to_event, "
                                  "targets to discovery with age, education, baseline MoCA, APOE e4 as comparators")


# ── P5  comparator proteins and published panels ────────────────────────────

def _p5_comparators(ctx, R: _Rows) -> Dict[str, Any]:
    P = "P5 comparators: NfL, GFAP, published panels"
    q = "How does the 5,000-protein score compare with NfL alone and with published incident-PD signatures?"
    prot_cols, X, M, base = ctx["prot_cols"], ctx["X"], ctx["M"], ctx["base"]
    ann: Dict[str, str] = {}
    p = TAB / "protein_annotation.csv"
    if p.exists():
        a = pd.read_csv(p)
        if {"uniprot", "gene"} <= set(a.columns):
            ann = {str(u).upper(): str(g).upper() for u, g in zip(a["uniprot"], a["gene"])}
    by_acc = {c.split(":", 1)[-1].split("-")[0].upper(): i for i, c in enumerate(prot_cols)}
    by_gene = {ann.get(acc, ""): i for acc, i in by_acc.items() if ann.get(acc)}
    found: Dict[str, int] = {}
    for gene, (acc, _, _) in _COMPARATORS.items():
        j = by_gene.get(gene, by_acc.get(acc))
        if j is not None:
            found[gene] = j
    missing = [g for g in _COMPARATORS if g not in found]
    R.add(P, "comparator proteins on the panel", "ALL", "n", len(found), note=", ".join(sorted(found)) or "none")
    if missing:
        R.add(P, "comparator proteins absent from the panel", "ALL", "n", len(missing), note=", ".join(missing))

    pdt = base[(base["group"] == "PD") & (base["split"] == "TRAIN")]
    pro = _prodromal_frame(ctx)
    prot = pro[pro["split"] == "TRAIN"] if pro is not None else None
    h = ctx["horizon"]

    def _vals(j: int, pos: np.ndarray) -> np.ndarray:
        v = X[pos, j].astype(float)
        v[M[pos, j] == 0] = np.nan
        return v

    def _report(label: str, v_pd: np.ndarray, v_pro: Optional[np.ndarray], source: str = ""):
        r, lo, hi, n = _rho(v_pd, pdt["y"])
        if n >= 6:
            R.add(P, f"{label}: Spearman vs UPDRS within PD", "TRAIN", "rho", r, lo, hi, n=n, note=source)
        if np.isfinite(pdt["caudate"]).sum() >= 20:
            r2, lo2, hi2, n2 = _rho(v_pd, pdt["caudate"])
            if n2 >= 6:
                R.add(P, f"{label}: Spearman vs caudate DaT within PD", "TRAIN", "rho", r2, lo2, hi2, n=n2)
        if prot is not None and v_pro is not None and len(prot):
            a, alo, ahi, n1, n0 = _auroc(_horizon_label(prot, h), v_pro)
            if n1 >= 3 and n0 >= 3:
                R.add(P, f"{label}: AUROC conversion within {h:.0f} m (prodromal, raw direction)", "TRAIN", "auroc",
                      a, alo, ahi, n=n1 + n0, events=n1)
        return (r, lo, hi, n)

    ref = _report("severity score (reference)", pdt["score"].values,
                  prot["score"].values if prot is not None else None, "the pipeline's primary model")
    nefl = (np.nan, np.nan, np.nan, 0)
    for gene in sorted(found, key=lambda g: list(_COMPARATORS).index(g)):
        j = found[gene]
        res = _report(gene, _vals(j, pdt["pos"].values), _vals(j, prot["pos"].values) if prot is not None else None,
                      _COMPARATORS[gene][2])
        if gene == "NEFL":
            nefl = res
    avail = [g for g in _UKB_PANEL if g in found]
    if len(avail) >= 3:
        def _comp(pos):
            Z = np.column_stack([_vals(found[g], pos) * _COMPARATORS[g][1] for g in avail])
            return np.nanmean(Z, axis=1)
        _report(f"UKB 2025 direction-weighted panel ({len(avail)}/{len(_UKB_PANEL)} proteins)", _comp(pdt["pos"].values),
                _comp(prot["pos"].values) if prot is not None else None, "Jiao et al. 2025 directions, z-mean")
    for b in EXTRA_BIOMARKERS:
        name = str(b.get("name", ""))
        if name and name in ctx["clin"].columns:
            col = pd.to_numeric(ctx["clin"][name], errors="coerce").values
            if np.isfinite(col[pdt["pos"].values]).mean() >= 0.3:
                _report(f"extra biomarker {name}", col[pdt["pos"].values], col[prot["pos"].values] if prot is not None else None,
                        "extra_biomarkers")

    sig = f"score ρ {_fmt(ref[0])} vs NEFL ρ {_fmt(nefl[0])} (UPDRS, TRAIN PD)" if np.isfinite(nefl[0]) else \
          f"score ρ {_fmt(ref[0])}; NEFL not on panel"
    return _summary_row(P, q, "yes" if found else "no", len(pdt) + (len(prot) if prot is not None else 0), np.nan,
                        len(pdt), np.nan, np.nan, "", nefl[0], nefl[1], nefl[2], "rho", True,
                        verdict="feasible; low effort" if found else "needs data",
                        next_step="published panels with directions in literature_proteins.yaml; nfl_only / published_panel "
                                  "feature sets in the prodromal benchmark; extra_biomarkers NEFL / GFAP from Project 318",
                        signal_text=sig)


# ── P6  plasma vs CSF overlap ───────────────────────────────────────────────

def _p6_csf(ctx, R: _Rows) -> Dict[str, Any]:
    P = "P6 plasma vs CSF (other tissue on disk)"
    q = "How many participants have both tissues, for a head-to-head severity / conversion comparison?"
    from .config import _cfg, _data_path
    from .olink_io import table_columns, read_table, ppmi_pid_to_amp, event_to_month
    cfg = _cfg("proteomics_panels", None)
    nested = isinstance(cfg, dict) and cfg and all(isinstance(v, dict) for v in cfg.values())
    others = [t for t in (cfg.keys() if nested else []) if str(t).upper() not in PROTEOMICS_TISSUES]
    if not others:
        return _summary_row(P, q, "not configured", 0, 0, 0, 0, np.nan, "|ρ|", np.nan, np.nan, np.nan, "none", False,
                            verdict="not configured",
                            next_step="add the CSF release under proteomics_panels: {CSF: {explore_ht: <277 parquet>}} "
                                      "and run with --tissue CSF / PLA+CSF (compare_runs on matched participants)")
    active = ctx["rows"][["participant_id", "month", "group", "split"]].drop_duplicates()
    best = (np.nan, 0)
    for t in others:
        pairs = []
        files = []
        for name, v in cfg[t].items():
            files += [str(x) for x in (v if isinstance(v, (list, tuple)) else [v])]
        for f in files:
            path = _data_path(f)
            if not Path(path).exists():
                R.add(P, f"{t}: {Path(f).name}", "ALL", "note", note="file not on disk")
                continue
            cols = table_columns(path)
            pc = next((c for c in cols if c.upper() in ("PATNO", "PARTICIPANT_ID")), None)
            ec = next((c for c in cols if c.upper() in ("EVENT_ID", "VISIT_NAME", "VISIT")), None)
            if pc is None:
                R.add(P, f"{t}: {Path(f).name}", "ALL", "note", note="no participant column")
                continue
            use = [pc] + ([ec] if ec else []) + [c for c in cols if c == "SampleType"]
            d = read_table(path, columns=use)
            if "SampleType" in d.columns:
                d = d[d["SampleType"].astype(str).str.upper() == "SAMPLE"]
            pid = ppmi_pid_to_amp(d[pc]) if pc.upper() == "PATNO" else d[pc].astype(str).map(map_participant_id)
            mon = event_to_month(d[ec]) if ec else pd.Series(0.0, index=d.index)
            pairs.append(pd.DataFrame({"participant_id": pid.values, "month": pd.to_numeric(mon, errors="coerce").values}).dropna())
        if not pairs:
            continue
        other = pd.concat(pairs).drop_duplicates()
        both = active.merge(other[["participant_id"]].drop_duplicates(), on="participant_id")
        same = active.merge(other, on=["participant_id", "month"])
        n_both = both["participant_id"].nunique()
        R.add(P, f"{t}: participants with a sample", "ALL", "n", other["participant_id"].nunique())
        R.add(P, f"{t} + {'/'.join(PROTEOMICS_TISSUES)}: participants with both tissues", "ALL", "n", n_both,
              note=f"MDE |rho| for a paired comparison {_fmt(mde_rho(n_both))}")
        R.add(P, f"{t} + {'/'.join(PROTEOMICS_TISSUES)}: same-visit sample pairs", "ALL", "n", len(same))
        for g in ("PD", "HC", "PRODROMAL"):
            for sp in ("TRAIN", "TEST"):
                b = both[(both["group"] == g) & (both["split"] == sp)]
                if len(b):
                    R.add(P, f"{t}: {g} with both tissues", sp, "n", b["participant_id"].nunique())
        if n_both > best[1]:
            best = (t, n_both)
    n = best[1]
    return _summary_row(P, q, "yes" if n else "no", n, np.nan, np.nan, np.nan, mde_rho(n) if n else np.nan, "|ρ|",
                        np.nan, np.nan, np.nan, "none", n >= _MIN_N,
                        verdict=("feasible; low effort (counts only)" if n >= _MIN_N else ("underpowered on disk" if n else "needs data")),
                        next_step="run --tissue CSF and PLA+CSF with the same holdout; compare_runs on the matched participants")


# ── P7  subgroups ───────────────────────────────────────────────────────────

def _p7_subgroups(ctx, R: _Rows) -> Dict[str, Any]:
    P = "P7 genetic and sex subgroups"
    q = "Does the severity signal hold in LRRK2 / GBA carriers and in both sexes; are non-manifest carriers covered?"
    ex, base = ctx["extras"], ctx["base"].copy()
    gen, cur, conv = ex.get("genetics"), ex.get("curated"), ctx["conv"]
    base["lrrk2"] = np.nan; base["gba"] = np.nan
    src = "none"
    if gen is not None and {"lrrk2_carrier", "gba_carrier"} & set(gen.columns):
        g = gen.set_index("participant_id")
        for k in ("lrrk2", "gba"):
            if f"{k}_carrier" in g.columns:
                base[k] = base["participant_id"].map(g[f"{k}_carrier"])
        src = "genetics file"
    if base["lrrk2"].isna().all() and cur is not None and "subgroup" in cur.columns:
        s = cur.set_index("participant_id")["subgroup"].astype(str).str.upper()
        base["lrrk2"] = base["participant_id"].map(s.str.contains("LRRK2").astype(float))
        base["gba"] = base["participant_id"].map(s.str.contains("GBA").astype(float))
        src = "curated subgroup"
    if base["lrrk2"].isna().all() and conv is not None and "enrllrrk2" in conv.columns:
        c = conv.set_index("participant_id")
        base["lrrk2"] = base["participant_id"].map(c["enrllrrk2"]); base["gba"] = base["participant_id"].map(c["enrlgba"])
        src = "prodromal enrolment flags (prodromal only)"
    R.add(P, "carrier status source", "ALL", "note", note=src)

    pdb = base[base["group"] == "PD"]
    for lab, mask in (("female", pdb["male"] == 0), ("male", pdb["male"] == 1),
                      ("LRRK2 carriers", pdb["lrrk2"] == 1), ("GBA carriers", pdb["gba"] == 1),
                      ("non-carriers", (pdb["lrrk2"] == 0) & (pdb["gba"] == 0))):
        for sp in ("TRAIN", "TEST"):
            n = int((mask & (pdb["split"] == sp)).sum())
            R.add(P, f"PD {lab}", sp, "n", n, n=n, note=f"MDE |rho| {_fmt(mde_rho(n))}" if n > 6 else "")
    pro = base[base["group"] == "PRODROMAL"]
    for lab, mask in (("LRRK2 non-manifest carriers", pro["lrrk2"] == 1), ("GBA non-manifest carriers", pro["gba"] == 1)):
        R.add(P, lab, "ALL", "n", int(mask.sum()))

    t = pdb[pdb["split"] == "TRAIN"]
    best = (np.nan, np.nan, np.nan, 0)
    for lab, mask in (("female", t["male"] == 0), ("male", t["male"] == 1), ("LRRK2 carriers", t["lrrk2"] == 1),
                      ("GBA carriers", t["gba"] == 1), ("non-carriers", (t["lrrk2"] == 0) & (t["gba"] == 0))):
        s = t[mask]
        r, lo, hi, n = _rho(s["score"], s["y"])
        if n >= 20:
            R.add(P, f"PD {lab}: Spearman OOF score vs UPDRS", "TRAIN", "rho", r, lo, hi, n=n)
            if lab in ("LRRK2 carriers", "GBA carriers") and (not np.isfinite(best[0]) or n > best[3]):
                best = (r, lo, hi, n)
    if pro is not None and ctx["conv"] is not None:
        pf = _prodromal_frame(ctx)
        if pf is not None:
            pf = pf.merge(base[["participant_id", "lrrk2", "gba"]], on="participant_id", how="left")
            for lab, col in (("LRRK2 non-manifest carriers", "lrrk2"), ("GBA non-manifest carriers", "gba")):
                s = pf[(pf[col] == 1) & (pf["split"] == "TRAIN")]
                if len(s) >= 15:
                    a, lo, hi, n1, n0 = _auroc(_horizon_label(s, ctx["horizon"]), s["score"])
                    if n1 >= 3 and n0 >= 3:
                        R.add(P, f"{lab}: AUROC score, conversion within {ctx['horizon']:.0f} m", "TRAIN", "auroc", a, lo, hi,
                              n=n1 + n0, events=n1)
    n_car = int(((pdb["lrrk2"] == 1) | (pdb["gba"] == 1)).sum())
    return _summary_row(P, q, "yes" if src != "none" else "partial", n_car, np.nan, best[3], np.nan,
                        mde_rho(n_car) if n_car > 6 else np.nan, "|ρ|", best[0], best[1], best[2], "rho", n_car >= _MIN_N,
                        next_step="config-level stratified runs (sex, carrier status); carrier-specific coefficients; "
                                  "score vs DaT in non-manifest carriers")


# ── figure and report ───────────────────────────────────────────────────────

def _figure(summ: pd.DataFrame, made: List[str]) -> None:
    from .figures_advanced import _style, _save, ORD3, C_TRAIN, MUTED, INK2
    plt = _style()
    s = summ.copy()
    s["label"] = s["path"].str.replace(r"^P(\d) ", lambda m: f"P{m.group(1)}  ", regex=True)
    n = len(s)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.5, 0.55 * n + 1.6), gridspec_kw={"width_ratios": [1.1, 1.0]})
    y = np.arange(n)[::-1]
    nn_raw = pd.to_numeric(s["n_all"], errors="coerce").values
    ev_raw = pd.to_numeric(s["events_all"], errors="coerce").values
    nn, ev = np.nan_to_num(nn_raw), np.nan_to_num(ev_raw)
    ax0.barh(y + 0.18, nn, height=0.34, color=ORD3[0], label="participants on disk (all splits)")
    ax0.barh(y - 0.18, ev, height=0.34, color=ORD3[2], label="events on disk")
    for yi, a, b, ar, br in zip(y, nn, ev, nn_raw, ev_raw):
        txt = ("—" if not np.isfinite(ar) else f"{int(ar):,}") + " / " + ("—" if not np.isfinite(br) else f"{int(br):,}")
        ax0.text(max(a, b) + max(nn.max(), 1) * 0.015, yi, txt, va="center", fontsize=7, color=INK2)
    ax0.set_yticks(y); ax0.set_yticklabels(s["label"], fontsize=8)
    ax0.set_xlim(0, max(nn.max(), 1) * 1.28)
    ax0.set_ylim(-0.7, n - 0.3)
    ax0.set_xlabel("participants / events available")
    ax0.set_title("Coverage (all splits)", loc="left")
    ax0.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=7)
    ax0.grid(axis="y", visible=False)

    eff = pd.to_numeric(s["exploratory_signal"], errors="coerce").values
    lo = pd.to_numeric(s["signal_lo"], errors="coerce").values
    hi = pd.to_numeric(s["signal_hi"], errors="coerce").values
    mde = pd.to_numeric(s["mde_80pct"], errors="coerce").values
    kinds = s["signal_kind"].astype(str).values
    units = s["mde_unit"].astype(str).values
    for yi, e, l, h, k, m, u, v in zip(y, eff, lo, hi, kinds, mde, units, s["verdict"]):
        mk = "o"
        if k == "auroc":                           # AUROC -> Gini so it shares the rho axis
            e, l, h, mk = 2 * e - 1, 2 * l - 1, 2 * h - 1, "s"
        elif k != "rho":                           # HR / text rows are not plotted on this axis
            e = np.nan
        tick = (2 * m - 1 if u == "AUROC" else m) if (np.isfinite(m) and u in ("AUROC", "|ρ|")) else np.nan
        if np.isfinite(tick):
            for xx in (-tick, tick):
                ax1.plot([xx, xx], [yi - 0.28, yi + 0.28], color=MUTED, lw=0.9)
        if np.isfinite(e):
            if np.isfinite(l) and np.isfinite(h):
                ax1.plot([l, h], [yi, yi], color=C_TRAIN, lw=1.4, solid_capstyle="butt")
            ax1.plot([e], [yi], marker=mk, ms=6.5, color=C_TRAIN, mec="white", mew=0.8, ls="none")
        else:
            ax1.text(0.03, yi, str(v), va="center", fontsize=7, color=MUTED)
    ax1.axvline(0, color=MUTED, lw=0.8, ls="--")
    ax1.set_yticks(y); ax1.set_yticklabels([""] * n)
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-0.7, n - 0.3)
    ax1.set_xlabel("effect on TRAIN rows (ρ, or Gini = 2·AUROC − 1)")
    ax1.set_title("Exploratory signal (TRAIN only; holdout never scored)", loc="left")
    ax1.grid(axis="y", visible=False)
    from matplotlib.lines import Line2D
    ax1.legend(handles=[Line2D([], [], marker="o", color=C_TRAIN, ls="none", label="Spearman ρ"),
                        Line2D([], [], marker="s", color=C_TRAIN, ls="none", label="Gini (2·AUROC − 1)"),
                        Line2D([], [], color=MUTED, lw=0.9, label="± minimum detectable effect, 80 % power")],
               loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=7)
    fig.suptitle("Path exploration: what the data on disk can support", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    _save(fig, "fig_path_exploration.png", made)


def _markdown(summ: pd.DataFrame, rows: pd.DataFrame, extras: Dict[str, Any]) -> str:
    L = ["# Path exploration — what the data on disk can support", "",
         "Exploratory only. Coverage counts use every split; effects are computed on TRAIN rows "
         "(out-of-fold severity score for PD / HC rows, transported score for prodromal rows) and the "
         "holdout is never scored. Minimum detectable effects (MDE) are for 80 % power at two-sided α = 0.05 "
         "with the coverage shown. A feasible path still needs its own pre-specified analysis with the holdout "
         "evaluated once. See `docs/literature_review_2026_gaps.md` for the rationale of each path.", ""]
    files = extras.get("files", {}) if extras else {}
    if files:
        L += ["PPMI extras found: " + "; ".join(f"{k}: {'yes' if v else 'no'}" for k, v in files.items()), ""]
    L += ["## Summary", "", "| path | question | data | n (all) | events (all) | MDE | exploratory signal (TRAIN) | verdict | next step |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in summ.iterrows():
        L.append(f"| {r['path']} | {r['question']} | {r['data_on_disk']} | {_fmt(r['n_all'], 0)} | {_fmt(r['events_all'], 0)} | "
                 f"{_mde_str(r)} | {_signal_str(r)} | **{r['verdict']}** | {r['next_step']} |")
    L += ["", "![paths](figures/fig_path_exploration.png)", ""]
    for path in summ["path"]:
        d = rows[rows["path"] == path]
        if d.empty:
            continue
        L += [f"## {path}", "", "| item | scope | metric | value | 95 % CI | n | events | note |", "|---|---|---|---|---|---|---|---|"]
        for _, r in d.iterrows():
            val = _fmt(r["value"], 0 if r["metric"] in ("n", "events") else 3) if r["metric"] != "note" else ""
            L.append(f"| {r['item']} | {r['scope']} | {r['metric']} | {val} | {_ci(r['ci_lo'], r['ci_hi'], 3).strip()} | "
                     f"{_fmt(r['n'], 0)} | {_fmt(r['events'], 0)} | {r['note']} |")
        L.append("")
    return "\n".join(L)


# ── stage entry ─────────────────────────────────────────────────────────────

def run_explore_paths(clin, z_prot, X, M, y_all, tv, cohort, PANEL_COL_INDICES, USE_PANEL_AWARE,
                      PRIMARY_LABEL, prot_cols, ppmi_dir: Optional[str] = None) -> Dict[str, Any]:
    print(f"\n{'=' * 60}")
    print("PATH EXPLORATION (feasibility of the candidate extensions; TRAIN-only, exploratory)")
    print(f"{'=' * 60}")
    from .ppmi_clinical import load_ppmi_extras
    extras = load_ppmi_extras(ppmi_dir if ppmi_dir is not None else PPMI_CLINICAL_DIR, log=print)
    ctx = _context(clin, X, M, y_all, tv, cohort, PANEL_COL_INDICES, USE_PANEL_AWARE, PRIMARY_LABEL, prot_cols, extras)
    R = _Rows()
    summ = []
    for fn in (_p1_saa, _p2_prodromal, _p3_dynamic, _p4_cognition, _p5_comparators, _p6_csf, _p7_subgroups):
        try:
            summ.append(fn(ctx, R))
        except Exception as e:                      # noqa: BLE001 - one path must not sink the others
            import traceback
            traceback.print_exc()
            summ.append(_summary_row(fn.__name__, "", "error", 0, 0, 0, 0, np.nan, "", np.nan, np.nan, np.nan, "none", False,
                                     verdict=f"error: {type(e).__name__}: {str(e)[:60]}"))
    rows = R.frame()
    summary = pd.DataFrame(summ)
    rows.to_csv(TAB / "path_feasibility.csv", index=False)
    summary.to_csv(TAB / "path_exploration_summary.csv", index=False)
    made: List[str] = []
    try:
        _figure(summary, made)
    except Exception as e:                          # noqa: BLE001
        print(f"  [figure] {type(e).__name__}: {e}")
    (OUT / "PATH_EXPLORATION.md").write_text(_markdown(summary, rows, extras), encoding="utf-8")

    print(f"\n  {'path':<44} {'data':<14} {'n(all)':>6} {'events':>7}  {'MDE (80 % power)':<16} {'signal (TRAIN only)':<44} verdict")
    for _, r in summary.iterrows():
        print(f"  {r['path']:<44.44} {str(r['data_on_disk']):<14} {_fmt(r['n_all'], 0):>6} {_fmt(r['events_all'], 0):>7}  "
              f"{_mde_str(r):<16} {_signal_str(r):<44.44} {r['verdict']}")
    print(f"\n  -> {TAB / 'path_feasibility.csv'}, {TAB / 'path_exploration_summary.csv'}, {OUT / 'PATH_EXPLORATION.md'}"
          + (f", figures/{made[0]}" if made else ""))
    out = {"paths": [{"path": r["path"], "data": r["data_on_disk"], "n_all": r["n_all"], "events_all": r["events_all"],
                      "verdict": r["verdict"]} for _, r in summary.iterrows()]}
    summary_update({"explore_paths": out})
    return out


def main() -> None:
    """Standalone: rebuild the aligned inputs from a finished run (cached matrix) and explore."""
    from .config import print_banner, SEVERITY_POPULATION
    from .data_loading import load_clinical, build_cohort_masks, load_proteomics
    from .features import build_aligned_matrices
    from .severity_model import setup_targets_and_cv
    print_banner()
    clin, *_ = load_clinical()
    cohort = build_cohort_masks(clin)
    z_prot, panel_protein_map = load_proteomics(clin, cohort["is_train"])
    feat = build_aligned_matrices(z_prot, None, clin.index, panel_protein_map, False)
    is_train, is_test = cohort["is_train"], cohort["is_test"]
    if SEVERITY_POPULATION == "pd_only":
        is_train_sev, is_test_sev = is_train & cohort["is_pd_flag"], is_test & cohort["is_pd_flag"]
    elif SEVERITY_POPULATION == "pd_hc":
        keep = cohort["is_pd_flag"] | cohort["is_control"]
        is_train_sev, is_test_sev = is_train & keep, is_test & keep
    else:
        is_train_sev, is_test_sev = is_train, is_test
    tv = setup_targets_and_cv(clin, z_prot, feat["M_prot"], is_train_sev, is_test_sev)
    label = "RidgeSVD"
    p = TAB / "summary.json"
    if p.exists():
        try:
            label = json.loads(p.read_text()).get("primary_model") or label
        except Exception:                           # noqa: BLE001
            pass
    print(f"  primary model label from the finished run: {label}")
    run_explore_paths(clin, z_prot, feat["X_prot"], feat["M_prot"], tv["y_all"], tv, cohort,
                      feat["PANEL_COL_INDICES"], feat["USE_PANEL_AWARE"], label, feat["prot_cols"])


if __name__ == "__main__":
    main()

"""
§0b — Assemble ``results/tables/clinical_unified.csv`` from the raw AMP-PD
release files.

Inputs (all resolved from ``config.CLINICAL_FILES``):
    MDS-UPDRS Part I-IV   -> updrs_total (one row per participant-visit)
    UPSIT                 -> upsit_total
    Demographics          -> sex, age_at_baseline
    amp_pd_case_control   -> case_control (CASE / CONTROL / OTHER)

Output columns (consumed by ``data_loading.load_clinical``):
    participant_id, visit_name, visit_month, cohort, case_control,
    diagnosis, updrs_total, mds_updrs_part_{i,ii,iii,iv}_total,
    upsit_total, sex, age_at_baseline, age, site, race, ethnicity

Column names in the release files are resolved by regex so minor naming
differences between AMP-PD releases do not break the build; every
resolution is printed so it can be audited.
"""

from __future__ import annotations

import re
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import (
    TAB, CLINICAL_FILES, TRAIN_PREFIX, TEST_PREFIX, ASSEMBLY,
)

CLINICAL_UNIFIED = TAB / "clinical_unified.csv"

_PID_PATTERNS   = [r"^participant_id$", r"^participant$", r"^patno$",
                   r"^subject_id$", r"^guid$"]
_VISIT_PATTERNS = [r"^visit_name$", r"^visit$", r"^event_name$"]
_MONTH_PATTERNS = [r"^visit_month$", r"^months?$", r"^month_of_visit$"]


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  helpers                                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _read(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required AMP-PD file not found: {path}\n"
            f"  -> check 'data_dir' / 'clinical_files' in config.yaml")
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig",
                     dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _find_col(df: pd.DataFrame, patterns: List[str],
              label: str, required: bool = True) -> Optional[str]:
    cols = list(df.columns)
    for pat in patterns:
        rx = re.compile(pat, flags=re.I)
        for c in cols:
            if rx.search(c):
                return c
    if required:
        raise KeyError(
            f"Could not resolve '{label}' column.\n"
            f"  tried patterns: {patterns}\n"
            f"  available columns: {cols}")
    return None


def _norm_pid(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def _visit_key(df: pd.DataFrame, label: str) -> pd.Series:
    """Canonical visit label 'M<month>' built from visit_month when present,
    otherwise parsed from visit_name."""
    mcol = _find_col(df, _MONTH_PATTERNS, "visit_month", required=False)
    vcol = _find_col(df, _VISIT_PATTERNS, "visit_name", required=False)
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    if mcol is not None:
        mn = pd.to_numeric(df[mcol], errors="coerce")
        out = mn.round().map(lambda z: f"M{int(z)}" if pd.notna(z) else pd.NA)
    if vcol is not None:
        v = df[vcol].astype(str).str.upper().str.strip()
        tok = v.str.extract(r"\b(M\d{1,3})\b", expand=False)
        fb  = v.str.extract(r"MONTH\s*(\d{1,3})", expand=False)
        fb  = fb.map(lambda x: f"M{int(x)}" if pd.notna(x) else pd.NA)
        bl  = v.where(v.str.contains(r"BASELINE|^BL$|^SC$|SCREEN", regex=True))
        bl  = bl.map(lambda x: "M0" if pd.notna(x) else pd.NA)
        out = out.fillna(tok).fillna(fb).fillna(bl)
    n_ok = out.notna().sum()
    print(f"    [{label}] visit key: month_col='{mcol}', name_col='{vcol}' "
          f"-> {n_ok}/{len(df)} rows resolved")
    return out.astype("object")


def _month_from_key(vk: pd.Series) -> pd.Series:
    return pd.to_numeric(vk.astype(str).str.extract(r"M(-?\d+)", expand=False),
                         errors="coerce")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  loaders                                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _load_updrs_part(path: str, part: str) -> pd.DataFrame:
    """Return (participant_id, visit_key, mds_updrs_part_<part>_total)."""
    label = f"UPDRS-{part.upper()}"
    print(f"  {label}: {Path(path).name}")
    df = _read(path)
    pid = _find_col(df, _PID_PATTERNS, "participant_id")
    score = _find_col(
        df,
        [rf"^mds_updrs_part_{part}_summary_score$",
         rf"^mds_updrs_part_{part}.*(summary|total).*score",
         rf"part_?{part}\b.*(summary|total)",
         r"summary_score$", r"total_score$"],
        f"{label} summary score")
    print(f"    [{label}] id='{pid}', score='{score}'")

    out = pd.DataFrame({
        "participant_id": _norm_pid(df[pid]),
        "visit_key": _visit_key(df, label),
        "score": pd.to_numeric(df[score], errors="coerce"),
    })

    extra_cols: List[str] = []
    if part == "iii":
        # Hoehn & Yahr stage lives in the Part III file
        hy = _find_col(df, [r"hoehn", r"\bhy\b.*stage", r"^upd2hy"],
                       "Hoehn & Yahr", required=False)
        if hy is not None:
            out["hoehn_yahr"] = pd.to_numeric(df[hy], errors="coerce")
            extra_cols.append("hoehn_yahr")
            print(f"    [{label}] Hoehn & Yahr col='{hy}' "
                  f"({int(out['hoehn_yahr'].notna().sum())} values)")
        # Visit-level "on PD medication?" flag (PPMI upd23a) — gives an
        # untreated -> treated contrast within participants
        med = _find_col(df, [r"^upd23a", r"medication_for_pd", r"on_pd_med", r"pd_medication"],
                        "on PD medication (visit)", required=False)
        if med is not None:
            s = df[med].astype(str).str.strip().str.lower()
            num = pd.to_numeric(df[med], errors="coerce")
            out["pd_medicated"] = np.where(num.notna(), (num > 0).astype(float),
                                  np.where(s.isin(["yes", "y", "true", "1"]), 1.0,
                                  np.where(s.isin(["no", "n", "false", "0"]), 0.0, np.nan)))
            extra_cols.append("pd_medicated")
            print(f"    [{label}] visit-level medication col='{med}' "
                  f"({int((out['pd_medicated'] == 1).sum())} medicated, "
                  f"{int((out['pd_medicated'] == 0).sum())} unmedicated rows)")
        # Medication state of the exam (ON / OFF / unknown)
        state = _find_col(
            df, [r"clinical_state", r"on_off", r"med.*state", r"^upd23b"],
            "Part III medication state", required=False)
        if state is not None:
            st = df[state].astype(str).str.upper()
            out["updrs3_state"] = np.where(st.str.contains("OFF", na=False), "OFF",
                                  np.where(st.str.contains("ON", na=False), "ON", "UNK"))
            extra_cols.append("updrs3_state")
            # Part III is often recorded twice per visit (ON / OFF medication).
            if ASSEMBLY["updrs3_prefer_off_state"]:
                out["is_off"] = (out["updrs3_state"] == "OFF").astype(int)
                n_off = int(out["is_off"].sum())
                print(f"    [{label}] medication-state col='{state}' "
                      f"({n_off} OFF rows) -> preferring OFF per visit")
                has_off = (out.groupby(["participant_id", "visit_key"])["is_off"]
                           .transform("max"))
                out = out[(out["is_off"] == 1) | (has_off == 0)]
                out = out.drop(columns=["is_off"])
        else:
            out["updrs3_state"] = "UNK"
            extra_cols.append("updrs3_state")

    out = out.dropna(subset=["visit_key", "score"])
    aggs = {"score": "mean"}
    for c in extra_cols:
        aggs[c] = "mean" if c == "hoehn_yahr" else ("max" if c == "pd_medicated" else "first")
    agg = (out.groupby(["participant_id", "visit_key"], as_index=False)
              .agg(aggs)
              .rename(columns={"score": f"mds_updrs_part_{part}_total"}))
    print(f"    [{label}] {len(agg)} participant-visits, "
          f"{agg['participant_id'].nunique()} participants")
    return agg


def _load_upsit(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        print(f"  UPSIT: not found ({path}) -> upsit_total will be NaN")
        return None
    print(f"  UPSIT: {Path(path).name}")
    df = _read(path)
    pid = _find_col(df, _PID_PATTERNS, "participant_id")
    score = _find_col(
        df,
        [r"^upsit_total_score$", r"upsit.*total", r"upsit.*score",
         r"^total_score$", r"^upsit$"],
        "UPSIT total", required=False)
    if score is None:
        print(f"    [UPSIT] no total-score column found in {list(df.columns)}"
              f" -> skipped")
        return None
    print(f"    [UPSIT] id='{pid}', score='{score}'")
    out = pd.DataFrame({
        "participant_id": _norm_pid(df[pid]),
        "visit_key": _visit_key(df, "UPSIT"),
        "upsit_total": pd.to_numeric(df[score], errors="coerce"),
    }).dropna(subset=["upsit_total"])
    out["visit_key"] = out["visit_key"].fillna("M0")
    agg = (out.groupby(["participant_id", "visit_key"], as_index=False)
              ["upsit_total"].max())
    print(f"    [UPSIT] {len(agg)} participant-visits, "
          f"{agg['participant_id'].nunique()} participants")
    return agg


def _load_demographics(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        print(f"  Demographics: not found ({path}) -> sex/age will be NaN")
        return None
    print(f"  Demographics: {Path(path).name}")
    df = _read(path)
    pid = _find_col(df, _PID_PATTERNS, "participant_id")
    sex = _find_col(df, [r"^sex$", r"^gender$"], "sex", required=False)
    age = _find_col(df, [r"^age_at_baseline$", r"^age_baseline$",
                         r"^age$", r"age.*enrol"], "age", required=False)
    race = _find_col(df, [r"^race$"], "race", required=False)
    eth  = _find_col(df, [r"^ethnicity$"], "ethnicity", required=False)
    print(f"    [Demographics] id='{pid}', sex='{sex}', age='{age}', "
          f"race='{race}', ethnicity='{eth}'")
    out = pd.DataFrame({"participant_id": _norm_pid(df[pid])})
    out["sex"] = (df[sex].astype(str).str.strip().str.title()
                  if sex else np.nan)
    out["age_at_baseline"] = (pd.to_numeric(df[age], errors="coerce")
                              if age else np.nan)
    out["race"] = df[race].astype(str).str.strip() if race else np.nan
    out["ethnicity"] = df[eth].astype(str).str.strip() if eth else np.nan
    out = (out.sort_values("participant_id")
              .groupby("participant_id", as_index=False).first())
    print(f"    [Demographics] {len(out)} participants")
    return out


def _load_case_control(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        print(f"  Case/control: not found ({path}) -> inferred from cohort")
        return None
    print(f"  Case/control: {Path(path).name}")
    df = _read(path)
    pid = _find_col(df, _PID_PATTERNS, "participant_id")
    cc_lt = _find_col(df, [r"^case_control_other_latest$",
                           r"case_control.*latest", r"^case_control$"],
                      "case_control", required=False)
    cc_bl = _find_col(df, [r"^case_control_other_at_baseline$",
                           r"case_control.*baseline"],
                      "case_control baseline", required=False)
    dx_lt = _find_col(df, [r"^diagnosis_latest$", r"diagnosis.*latest",
                           r"^diagnosis$"], "diagnosis", required=False)
    dx_bl = _find_col(df, [r"^diagnosis_at_baseline$",
                           r"diagnosis.*baseline"], "diagnosis baseline",
                      required=False)
    print(f"    [Case/control] id='{pid}', latest='{cc_lt}', "
          f"baseline='{cc_bl}', diagnosis='{dx_lt}'")

    def _label(series: pd.Series) -> pd.Series:
        s = series.astype(str).str.strip().str.upper()
        out = pd.Series("OTHER", index=s.index, dtype="object")
        out[s.str.contains(r"CONTROL|^HC$|HEALTHY", na=False)] = "CONTROL"
        out[s.str.contains(r"^CASE|^PD$|PARKINSON", na=False)] = "CASE"
        out[s.isin(["", "NAN", "NONE"])] = pd.NA
        return out

    out = pd.DataFrame({"participant_id": _norm_pid(df[pid])})
    lab = _label(df[cc_lt]) if cc_lt else pd.Series(pd.NA, index=df.index)
    if cc_bl:
        lab = lab.fillna(_label(df[cc_bl]))
    out["case_control"] = lab.values
    out["diagnosis"] = (df[dx_lt] if dx_lt else
                        (df[dx_bl] if dx_bl else np.nan))
    if dx_bl:
        out["diagnosis_at_baseline"] = df[dx_bl].values
    out = out.groupby("participant_id", as_index=False).first()
    print(f"    [Case/control] {len(out)} participants: "
          f"{out['case_control'].value_counts(dropna=False).to_dict()}")
    return out


def _load_med_history(path: str) -> Optional[pd.DataFrame]:
    """Participant-level age at PD diagnosis (for disease duration)."""
    if not os.path.exists(path):
        print(f"  PD medical history: not found ({Path(path).name}) -> "
              f"disease duration unavailable")
        return None
    print(f"  PD medical history: {Path(path).name}")
    df = _read(path)
    pid = _find_col(df, _PID_PATTERNS, "participant_id")
    age_dx = _find_col(df, [r"age.*diagnos", r"diagnos.*age", r"^age_at_dx$",
                            r"age.*onset", r"onset.*age"],
                       "age at diagnosis", required=False)
    yr_dx = _find_col(df, [r"diagnos.*year", r"year.*diagnos", r"^pd_dx_year$",
                           r"diagnosis_date", r"date.*diagnos"],
                      "diagnosis year/date", required=False)
    levo = _find_col(df, [r"^on_levodopa$", r"levodopa", r"l_?dopa"],
                     "levodopa use", required=False)
    ledd = _find_col(df, [r"^ledd$", r"levodopa.*equivalent", r"\bledd\b"],
                     "LEDD", required=False)
    dopa_any = _find_col(df, [r"on_dopamine_agonist", r"dopamine.*agonist",
                              r"on_other_pd_medications", r"pd_medication"],
                         "other dopaminergic medication", required=False)
    print(f"    [MedHx] id='{pid}', age_at_diagnosis='{age_dx}', "
          f"diagnosis_year='{yr_dx}', levodopa='{levo}', LEDD='{ledd}', "
          f"other_dopaminergic='{dopa_any}'")
    if age_dx is None and yr_dx is None and levo is None and ledd is None:
        print("    [MedHx] no diagnosis-age / medication column -> skipped")
        return None
    out = pd.DataFrame({"participant_id": _norm_pid(df[pid])})
    out["age_at_diagnosis"] = (pd.to_numeric(df[age_dx], errors="coerce")
                               if age_dx else np.nan)
    if yr_dx:
        yr = pd.to_numeric(df[yr_dx].astype(str).str.extract(r"(\d{4})", expand=False),
                           errors="coerce")
        out["diagnosis_year"] = yr

    def _yes(col):
        s = df[col].astype(str).str.strip().str.lower()
        num = pd.to_numeric(df[col], errors="coerce")
        return np.where(num.notna(), (num > 0).astype(float),
                        s.isin(["yes", "y", "true", "1", "on"]).astype(float))

    # participant-level medication exposure (any visit) — a reviewer-facing
    # confounder: plasma DDC rises with levodopa/DDC-inhibitor treatment
    if levo:
        out["on_levodopa"] = _yes(levo)
    if dopa_any:
        out["on_other_dopaminergic"] = _yes(dopa_any)
    if ledd:
        out["ledd"] = pd.to_numeric(df[ledd], errors="coerce")
    g = out.groupby("participant_id", as_index=False)
    aggs = {}
    for c in out.columns:
        if c == "participant_id":
            continue
        aggs[c] = "max" if c in ("on_levodopa", "on_other_dopaminergic") else "first"
    out = (out.sort_values("participant_id").groupby("participant_id", as_index=False)
              .agg(aggs))
    if "ledd" in out.columns:
        out["ledd"] = out["ledd"].fillna(np.nan)
    msg = f"    [MedHx] {int(out['age_at_diagnosis'].notna().sum())} with age at diagnosis"
    if "on_levodopa" in out.columns:
        msg += f"; {int((out['on_levodopa'] > 0).sum())} ever on levodopa"
    if "ledd" in out.columns:
        msg += f"; {int(out['ledd'].notna().sum())} with LEDD"
    print(msg)
    return out


def _load_datscan(path: str) -> Optional[pd.DataFrame]:
    """Visit-level striatal binding ratios (mean L/R putamen, caudate)."""
    if not os.path.exists(path):
        print(f"  DaTSCAN: not found ({Path(path).name}) -> skipped")
        return None
    print(f"  DaTSCAN: {Path(path).name}")
    df = _read(path)
    pid = _find_col(df, _PID_PATTERNS, "participant_id")
    put = [c for c in df.columns if re.search(r"putamen", c, re.I)]
    cau = [c for c in df.columns if re.search(r"caudate", c, re.I)]
    print(f"    [DaTSCAN] id='{pid}', putamen={put}, caudate={cau}")
    if not put and not cau:
        print("    [DaTSCAN] no putamen/caudate columns -> skipped")
        return None
    out = pd.DataFrame({"participant_id": _norm_pid(df[pid]),
                        "visit_key": _visit_key(df, "DaTSCAN")})
    # PPMI's diagnostic scan is done at screening, weeks *before* the baseline
    # visit; it may carry a negative visit_month (M-1, M-2 ...).  Key it as the
    # baseline scan so it lands on the M0 clinical row and on the baseline
    # proteomic sample.
    neg = out["visit_key"].astype(str).str.match(r"^M-\d+$")
    if neg.any():
        print(f"    [DaTSCAN] {int(neg.sum())} pre-baseline (negative-month) scans keyed as M0")
        out.loc[neg, "visit_key"] = "M0"
    vc = out["visit_key"].value_counts(dropna=False).head(8)
    print("    [DaTSCAN] visit keys: " + ", ".join(f"{k}={v}" for k, v in vc.items()))
    if put:
        out["datscan_putamen"] = df[put].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    if cau:
        out["datscan_caudate"] = df[cau].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    if put and cau:
        out["datscan_striatum"] = out[["datscan_putamen", "datscan_caudate"]].mean(axis=1)
    out["visit_key"] = out["visit_key"].fillna("M0")
    val_cols = [c for c in out.columns if c.startswith("datscan_")]
    out = out.dropna(subset=val_cols, how="all")
    agg = out.groupby(["participant_id", "visit_key"], as_index=False)[val_cols].mean()
    print(f"    [DaTSCAN] {len(agg)} participant-visits, "
          f"{agg['participant_id'].nunique()} participants")
    return agg


def _load_tte(path: str) -> Optional[pd.DataFrame]:
    """Time-to-event endpoints -> long table (participant_id, endpoint, time, event).

    Column pairs are taken from config ``tte_endpoints`` when given, otherwise
    detected by name: a column containing 'time'/'tte'/'years'/'months' is
    paired with a column sharing its stem that contains 'event'/'status'/
    'censor'/'reached'.
    """
    from .config import TTE_ENDPOINTS
    if not os.path.exists(path):
        print(f"  Time-to-event: not found ({Path(path).name}) -> skipped")
        return None
    print(f"  Time-to-event: {Path(path).name}")
    df = _read(path)
    pid = _find_col(df, _PID_PATTERNS, "participant_id")
    pairs = []
    if TTE_ENDPOINTS:
        for e in TTE_ENDPOINTS:
            pairs.append((e["name"], e["time_col"], e["event_col"]))
    else:
        low = {c: c.lower() for c in df.columns}
        time_cols = [c for c in df.columns
                     if re.search(r"time|tte|years|months|days", low[c]) and c != pid]
        for tc in time_cols:
            stem = re.sub(r"(time_to_|_time|tte_|_tte|_years|_months|_days|years_to_|"
                          r"months_to_|days_to_|time_)", "", low[tc])
            cand = [c for c in df.columns if c not in (tc, pid)
                    and re.search(r"event|status|censor|reached|occur", low[c])
                    and (stem in low[c] or re.sub(r"(event_|_event|status_|_status|"
                                                  r"_censor|censor_|_reached|reached_)",
                                                  "", low[c]) == stem)]
            if cand:
                pairs.append((stem or tc, tc, cand[0]))
    if not pairs:
        low = {c: c.lower() for c in df.columns}
        tcols = [c for c in df.columns if c != pid and re.search(r"time|tte|years|months|days", low[c])]
        ecols = [c for c in df.columns if c != pid and re.search(r"event|status|censor|reached|occur", low[c])]
        if len(tcols) == 1 and len(ecols) == 1:
            pairs.append(("endpoint", tcols[0], ecols[0]))
            print(f"    [TTE] single endpoint detected: time='{tcols[0]}', event='{ecols[0]}'")
    if not pairs:
        print(f"    [TTE] could not pair time/event columns in {list(df.columns)} -> "
              f"set 'tte_endpoints' in config.yaml")
        return None
    rows = []
    for name, tc, ec in pairs:
        t = pd.to_numeric(df[tc], errors="coerce")
        ev = df[ec].astype(str).str.strip().str.lower()
        e = pd.to_numeric(df[ec], errors="coerce")
        if e.isna().all():
            e = ev.isin(["1", "true", "yes", "event", "y", "reached"]).astype(float)
        e = (e > 0).astype(float)
        sub = pd.DataFrame({"participant_id": _norm_pid(df[pid]), "endpoint": name,
                            "time": t, "event": e}).dropna(subset=["time"])
        rows.append(sub)
        print(f"    [TTE] endpoint '{name}': time='{tc}', event='{ec}', "
              f"n={len(sub)}, events={int(sub['event'].sum())}")
    return pd.concat(rows, ignore_index=True)


def _load_extra_biomarkers() -> List[pd.DataFrame]:
    """Optional external biomarkers (config `extra_biomarkers`), one frame each
    with columns participant_id, visit_key, <name>."""
    from .config import EXTRA_BIOMARKERS, DATA_DIR
    out = []
    for spec in EXTRA_BIOMARKERS:
        name = str(spec.get("name", "")).strip()
        path = spec.get("file", "")
        path = str(path if Path(str(path)).is_absolute() else DATA_DIR / str(path))
        if not name or not os.path.exists(path):
            print(f"  Biomarker '{name}': not found ({path}) -> skipped")
            continue
        print(f"  Biomarker '{name}': {Path(path).name}")
        df = _read(path)
        pid = _find_col(df, _PID_PATTERNS, "participant_id")
        vc = spec.get("value_col")
        val = _find_col(df, [rf"^{re.escape(str(vc))}$"] if vc else [rf"^{re.escape(name)}$", name],
                        f"{name} value", required=False)
        if val is None:
            print(f"    [{name}] value column not found in {list(df.columns)} -> skipped")
            continue
        bm = pd.DataFrame({"participant_id": _norm_pid(df[pid]),
                           "visit_key": _visit_key(df, name),
                           name: pd.to_numeric(df[val], errors="coerce")}).dropna(subset=[name])
        bm["visit_key"] = bm["visit_key"].fillna("M0")
        if spec.get("log_transform", False):
            bm[name] = np.log1p(bm[name].clip(lower=0))
        agg = bm.groupby(["participant_id", "visit_key"], as_index=False)[name].mean()
        print(f"    [{name}] value col='{val}', {len(agg)} participant-visits, "
              f"{agg['participant_id'].nunique()} participants")
        out.append(agg)
    return out


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  main assembly                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _cohort_from_pid(pid: pd.Series) -> pd.Series:
    up = pid.astype(str).str.upper()
    out = pd.Series("OTHER", index=pid.index, dtype="object")
    out[up.str.startswith(TRAIN_PREFIX)] = "TRAIN"
    out[up.str.startswith(TEST_PREFIX)] = "TEST"
    return out


def assemble_clinical(force: bool = False) -> pd.DataFrame:
    """Build (or reuse) results/tables/clinical_unified.csv."""
    if CLINICAL_UNIFIED.exists() and not force:
        df = pd.read_csv(CLINICAL_UNIFIED)
        print(f"[Assembly] Reusing {CLINICAL_UNIFIED} "
              f"({len(df)} rows, {df['participant_id'].nunique()} participants)"
              f"  [use --force_assembly to rebuild]")
        return df

    print("=" * 70)
    print("ASSEMBLING clinical_unified.csv FROM RAW AMP-PD FILES")
    print("=" * 70)

    # ── UPDRS parts ─────────────────────────────────────────────────────
    parts = {}
    for part in ("i", "ii", "iii", "iv"):
        parts[part] = _load_updrs_part(CLINICAL_FILES[f"updrs_{part}"], part)

    base = parts["i"]
    for part in ("ii", "iii", "iv"):
        base = base.merge(parts[part], on=["participant_id", "visit_key"],
                          how="outer")

    p1, p2, p3, p4 = (f"mds_updrs_part_{p}_total" for p in ("i", "ii", "iii", "iv"))
    if ASSEMBLY["updrs_part_iv_missing_as_zero"]:
        n_fill = int(base[p4].isna().sum())
        base[p4] = base[p4].fillna(0.0)
        print(f"  UPDRS: Part IV missing on {n_fill} participant-visits "
              f"-> treated as 0 (updrs_part_iv_missing_as_zero=true)")
    core_ok = base[[p1, p2, p3]].notna().all(axis=1)
    base["updrs_total"] = np.where(
        core_ok & base[p4].notna(),
        base[[p1, p2, p3, p4]].sum(axis=1), np.nan)
    print(f"  UPDRS: {int(core_ok.sum())}/{len(base)} participant-visits have "
          f"Parts I+II+III; updrs_total finite on "
          f"{int(np.isfinite(base['updrs_total']).sum())}")

    # ── Cohort / case-control / demographics ────────────────────────────
    cc = _load_case_control(CLINICAL_FILES["case_control"])
    demo = _load_demographics(CLINICAL_FILES["demographics"])
    upsit = _load_upsit(CLINICAL_FILES["upsit"])

    # Participants known to the study but without UPDRS get one M0 row so
    # that healthy controls with proteomics still contribute to HC anchoring.
    known = set(base["participant_id"])
    extra_ids: List[str] = []
    for tbl in (cc, demo):
        if tbl is not None:
            extra_ids.extend(p for p in tbl["participant_id"] if p not in known)
    extra_ids = sorted(set(extra_ids))
    if extra_ids:
        base = pd.concat([base, pd.DataFrame({
            "participant_id": extra_ids, "visit_key": "M0"})],
            ignore_index=True)
        print(f"  Added {len(extra_ids)} participants without UPDRS "
              f"(single M0 row, updrs_total=NaN)")

    base["cohort"] = _cohort_from_pid(base["participant_id"])
    if cc is not None:
        base = base.merge(cc, on="participant_id", how="left")
    else:
        base["case_control"] = pd.NA
        base["diagnosis"] = pd.NA
    default_cc = pd.Series(
        np.where(base["cohort"].eq("TEST"), "CASE", "UNKNOWN"),
        index=base.index)
    base["case_control"] = base["case_control"].astype("object").fillna(default_cc)

    if demo is not None:
        base = base.merge(demo, on="participant_id", how="left")
    else:
        for c in ("sex", "age_at_baseline", "race", "ethnicity"):
            base[c] = np.nan

    if upsit is not None:
        base = base.merge(upsit, on=["participant_id", "visit_key"], how="left")
        if ASSEMBLY["upsit_fill_from_baseline"]:
            first = (upsit.assign(m=_month_from_key(upsit["visit_key"]))
                          .sort_values(["participant_id", "m"])
                          .drop_duplicates("participant_id")
                          .set_index("participant_id")["upsit_total"])
            n_before = int(base["upsit_total"].notna().sum())
            base["upsit_total"] = base["upsit_total"].fillna(
                base["participant_id"].map(first))
            print(f"  UPSIT: {n_before} visit-matched -> "
                  f"{int(base['upsit_total'].notna().sum())} after "
                  f"earliest-visit carry-forward")
    else:
        base["upsit_total"] = np.nan

    # ── Optional: disease duration, DaTSCAN, time-to-event ──────────────
    medhx = _load_med_history(CLINICAL_FILES.get("med_history", ""))
    if medhx is not None:
        base = base.merge(medhx, on="participant_id", how="left")
    dat = _load_datscan(CLINICAL_FILES.get("datscan", ""))
    if dat is not None:
        base = base.merge(dat, on=["participant_id", "visit_key"], how="left")
        bl = (dat.assign(m=_month_from_key(dat["visit_key"]))
                 .sort_values(["participant_id", "m"]).drop_duplicates("participant_id")
                 .set_index("participant_id"))
        for c in [c for c in dat.columns if c.startswith("datscan_")]:
            base[f"{c}_baseline"] = base["participant_id"].map(bl[c])
    tte = _load_tte(CLINICAL_FILES.get("time_to_event", ""))
    if tte is not None:
        tte.to_csv(TAB / "time_to_event.csv", index=False)
    for bm in _load_extra_biomarkers():
        name = [c for c in bm.columns if c not in ("participant_id", "visit_key")][0]
        base = base.merge(bm, on=["participant_id", "visit_key"], how="left")
        bl = (bm.assign(m=_month_from_key(bm["visit_key"]))
                .sort_values(["participant_id", "m"]).drop_duplicates("participant_id")
                .set_index("participant_id")[name])
        base[f"{name}_baseline"] = base["participant_id"].map(bl)

    # ── Final columns ───────────────────────────────────────────────────
    base = base.rename(columns={"visit_key": "visit_name"})
    base["visit_month"] = _month_from_key(base["visit_name"])
    base["age"] = base["age_at_baseline"] + base["visit_month"] / 12.0
    base["site"] = base["participant_id"].str.split("-").str[0]
    if "age_at_diagnosis" in base.columns:
        base["disease_duration_years"] = base["age"] - base["age_at_diagnosis"]
        base.loc[base["disease_duration_years"] < -1, "disease_duration_years"] = np.nan
        base.loc[base["case_control"].astype(str).str.upper() == "CONTROL",
                 "disease_duration_years"] = np.nan
        print(f"  Disease duration available on "
              f"{int(base['disease_duration_years'].notna().sum())} rows")

    if not ASSEMBLY["keep_other_cohorts"]:
        n_other = int(base["cohort"].eq("OTHER").sum())
        base = base[base["cohort"] != "OTHER"]
        print(f"  Dropped {n_other} rows from cohorts other than "
              f"{TRAIN_PREFIX}/{TEST_PREFIX} (keep_other_cohorts=false)")

    cols = ["participant_id", "visit_name", "visit_month", "cohort",
            "case_control", "diagnosis", "updrs_total", p1, p2, p3, p4,
            "hoehn_yahr", "updrs3_state", "pd_medicated", "upsit_total", "sex", "age_at_baseline",
            "age", "age_at_diagnosis", "disease_duration_years",
            "on_levodopa", "on_other_dopaminergic", "ledd", "site",
            "race", "ethnicity"]
    cols = [c for c in cols if c in base.columns]
    cols += [c for c in base.columns if c.startswith("datscan_")]
    from .config import EXTRA_BIOMARKERS
    for spec in EXTRA_BIOMARKERS:
        n = str(spec.get("name", ""))
        cols += [c for c in (n, f"{n}_baseline") if c in base.columns and c not in cols]
    base = (base[cols]
            .sort_values(["participant_id", "visit_month"])
            .reset_index(drop=True))
    base.to_csv(CLINICAL_UNIFIED, index=False)

    print("\n  clinical_unified.csv summary")
    print(f"    rows (participant-visits) : {len(base)}")
    print(f"    participants              : {base['participant_id'].nunique()}")
    for coh in ("TRAIN", "TEST"):
        m = base["cohort"].eq(coh)
        print(f"    {coh:<5} participants        : "
              f"{base.loc[m, 'participant_id'].nunique()}  "
              f"(rows={int(m.sum())}, with UPDRS="
              f"{int(np.isfinite(base.loc[m, 'updrs_total']).sum())}, "
              f"{base.loc[m, 'case_control'].value_counts().to_dict()})")
    print(f"    UPSIT available           : "
          f"{int(base['upsit_total'].notna().sum())} rows")
    print(f"  -> {CLINICAL_UNIFIED}")
    return base


if __name__ == "__main__":
    assemble_clinical(force=True)

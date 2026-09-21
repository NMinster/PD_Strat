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
    return pd.to_numeric(vk.astype(str).str.extract(r"M(\d+)", expand=False),
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

    # Part III is often recorded twice per visit (ON / OFF medication).
    if part == "iii" and ASSEMBLY["updrs3_prefer_off_state"]:
        state = _find_col(
            df, [r"clinical_state", r"on_off", r"med.*state", r"^upd23b"],
            "Part III medication state", required=False)
        if state is not None:
            st = df[state].astype(str).str.upper()
            out["is_off"] = st.str.contains("OFF", na=False).astype(int)
            n_off = int(out["is_off"].sum())
            print(f"    [{label}] medication-state col='{state}' "
                  f"({n_off} OFF rows) -> preferring OFF per visit")
            has_off = (out.groupby(["participant_id", "visit_key"])["is_off"]
                       .transform("max"))
            out = out[(out["is_off"] == 1) | (has_off == 0)]
            out = out.drop(columns=["is_off"])

    out = out.dropna(subset=["visit_key", "score"])
    agg = (out.groupby(["participant_id", "visit_key"], as_index=False)["score"]
              .mean()
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

    # ── Final columns ───────────────────────────────────────────────────
    base = base.rename(columns={"visit_key": "visit_name"})
    base["visit_month"] = _month_from_key(base["visit_name"])
    base["age"] = base["age_at_baseline"] + base["visit_month"] / 12.0
    base["site"] = base["participant_id"].str.split("-").str[0]

    if not ASSEMBLY["keep_other_cohorts"]:
        n_other = int(base["cohort"].eq("OTHER").sum())
        base = base[base["cohort"] != "OTHER"]
        print(f"  Dropped {n_other} rows from cohorts other than "
              f"{TRAIN_PREFIX}/{TEST_PREFIX} (keep_other_cohorts=false)")

    cols = ["participant_id", "visit_name", "visit_month", "cohort",
            "case_control", "diagnosis", "updrs_total", p1, p2, p3, p4,
            "upsit_total", "sex", "age_at_baseline", "age", "site",
            "race", "ethnicity"]
    cols = [c for c in cols if c in base.columns]
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

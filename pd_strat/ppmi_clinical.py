"""
PPMI-native clinical reader.

The AMP-PD v4 clinical tables stop in 2023, so PPMI participants enrolled or
followed up since then (PPMI LITE and the Project 222 / 314 additions) have
Olink samples but no UPDRS row to join.  This module reads the files PPMI
distributes through LONI "Study Data" and produces rows in the
clinical_unified schema that ``assemble_clinical`` appends for every
(participant, visit) the AMP-PD tables do not already cover.

Files looked for in ``ppmi_clinical_dir`` (names matched case-insensitively
on substrings, so any download date suffix is fine):

  MDS-UPDRS_Part_I.csv                          NP1RTOT   (clinician Part I)
  MDS-UPDRS_Part_I_Patient_Questionnaire.csv    NP1PTOT   (patient Part I)
  MDS_UPDRS_Part_II__Patient_Questionnaire.csv  NP2PTOT
  MDS-UPDRS_Part_III.csv                        NP3TOT, NHY, PDSTATE, PDTRTMNT
  MDS-UPDRS_Part_IV__Motor_Complications.csv    NP4TOT
  Participant_Status.csv                        COHORT / COHORT_DEFINITION, ENROLL_AGE
  Demographics.csv                              SEX, BIRTHDT
  Age_at_visit.csv                              AGE_AT_VISIT
  PD_Diagnosis_History.csv                      PDDXDT
  University_of_Pennsylvania_Smell_Identification_Test*.csv   TOTAL_CORRECT
  DaTScan_Analysis.csv                          DATSCAN_PUTAMEN_R/L, DATSCAN_CAUDATE_R/L
  LEDD_Concomitant_Medication_Log.csv           LEDTRT, LEDD

Every file is optional; a missing one is reported and its columns left NaN.
Item totals are summed from the item columns when a total column is absent.
``python -m pd_strat.ppmi_clinical S:/PPMI_clinical`` prints what would be
read without touching results.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .olink_io import PPMI_EVENT_MONTHS


def _name(p) -> str:
    """Display name for a path or a list of paths."""
    if isinstance(p, (list, tuple)):
        return " + ".join(Path(x).name for x in p)
    return Path(p).name

# logical name -> (filename regex, required)
_FILES: Dict[str, Tuple[str, bool]] = {
    "part1":   (r"MDS[-_ ]*UPDRS[-_ ]*Part[-_ ]*I(?![IV])(?!.*Patient)", True),
    "part1pq": (r"MDS[-_ ]*UPDRS[-_ ]*Part[-_ ]*I[-_ ]*(Patient|Pt)", False),
    "part2":   (r"MDS[-_ ]*UPDRS[-_ ]*Part[-_ ]*II(?!I)", True),
    "part3":   (r"MDS[-_ ]*UPDRS[-_ ]*Part[-_ ]*III(?!.*(ON_OFF|Dosing|Determination))", True),
    "part4":   (r"MDS[-_ ]*UPDRS[-_ ]*Part[-_ ]*IV", False),
    "status":  (r"Participant[-_ ]?Status", True),
    "demo":    (r"^Demographics", False),
    "age":     (r"Age[-_ ]?at[-_ ]?visit", False),
    "dx":      (r"PD[-_ ]?Diagnosis[-_ ]?History", False),
    "upsit":   (r"Smell[-_ ]?Identification|UPSIT", False),
    "datscan": (r"DaTScan[-_ ]?Analysis|DATSCAN[-_ ]?Analysis|Quant[-_ ]?SBR|SBR", False),
    "ledd":    (r"LEDD", False),
    # recognised now, used by the prodromal-conversion endpoint (next stage)
    "primary_dx":     (r"Primary[-_ ]?(Clinical|Research)[-_ ]?Diagnosis", False),
    "clinical_dx":    (r"^Clinical[-_ ]?Diagnosis", False),
    "prodromal":      (r"Prodromal[-_ ]?(Diagnostic|History)", False),
    "eligibility":    (r"Prodromal[-_ ]?Cohort[-_ ]?Eligibility", False),
    "datscan_visual": (r"Visual[-_ ]?Read|Visual[-_ ]?Interpretation", False),
    "curated":        (r"Curated[-_ ]?Data[-_ ]?Cut", False),
}
_TABULAR = (".csv", ".txt", ".tsv", ".xlsx", ".xls")
_LEVODOPA = re.compile(r"levodopa|sinemet|carbidopa|madopar|rytary|duopa|stalevo|benserazide|inbrija", re.I)
_AGONIST = re.compile(r"pramipexole|mirapex|ropinirole|requip|rotigotine|neupro|apomorphine|apokyn|piribedil", re.I)


def _read(path, nrows: Optional[int] = None) -> pd.DataFrame:
    if isinstance(path, (list, tuple)):
        parts = []
        for pth in path:
            d = _read_one(pth, nrows)
            d["_archived"] = int("archiv" in Path(pth).name.lower())
            parts.append(d)
        if len(parts) == 1:
            return parts[0].drop(columns="_archived")
        df = pd.concat(parts, ignore_index=True, sort=False)
        keys = [c for c in ("PATNO", "EVENT_ID") if c in df.columns]
        if keys:
            cur = df[df["_archived"] == 0]
            seen = set(map(tuple, cur[keys].astype(str).values))
            arch = df[df["_archived"] == 1]
            dup = arch[keys].astype(str).apply(tuple, axis=1).isin(seen) if len(arch) else pd.Series([], dtype=bool)
            df = pd.concat([cur, arch[~dup.values]] if len(arch) else [cur], ignore_index=True, sort=False)
        return df.drop(columns="_archived")
    return _read_one(path, nrows)


def _read_one(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    if Path(path).suffix.lower() in (".xlsx", ".xls"):
        xl = pd.ExcelFile(path)
        sheet = xl.sheet_names[0]
        for sh in xl.sheet_names:   # the Curated Data Cut: the data sheet is the one with PATNO
            cols = [str(c).strip().upper() for c in pd.read_excel(xl, sheet_name=sh, nrows=0).columns]
            if "PATNO" in cols:
                sheet = sh
                break
        df = pd.read_excel(xl, sheet_name=sheet, dtype=str, nrows=nrows)
    else:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", dtype=str, nrows=nrows)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _col(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        rx = re.compile(pat, re.I)
        for c in df.columns:
            if rx.search(str(c)):
                return c
    return None


def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _date(s: pd.Series) -> pd.Series:
    """PPMI dates are MM/YYYY (sometimes MM/DD/YYYY)."""
    s = s.astype(str).str.strip()
    out = pd.to_datetime(s, format="%m/%Y", errors="coerce")
    miss = out.isna()
    if miss.any():
        out[miss] = pd.to_datetime(s[miss], errors="coerce")
    return out


def find_files(root: str) -> Dict[str, Optional[str]]:
    rootp = Path(root)
    found: Dict[str, Optional[str]] = {k: None for k in _FILES}
    if not rootp.is_dir():
        return found
    cands = sorted(p for p in rootp.rglob("*") if p.is_file() and p.suffix.lower() in _TABULAR)
    for key, (rx, _) in _FILES.items():
        r = re.compile(rx, re.I)
        hits = [p for p in cands if r.search(p.stem)]
        if hits:
            # archived (legacy CRF) versions first, current versions last: when the
            # same participant-visit appears in both, the current file wins
            hits = sorted(hits, key=lambda p: (0 if "archiv" in p.name.lower() else 1, p.stat().st_mtime, p.name))
            found[key] = [str(p) for p in hits]
    return found


def _pid(df: pd.DataFrame) -> pd.Series:
    c = _col(df, [r"^PATNO$", r"^participant_id$", r"^patno"])
    if c is None:
        raise KeyError(f"no PATNO column in {list(df.columns)[:10]}")
    v = df[c].astype(str).str.strip().str.upper().str.replace(r"^PPMI[-_ ]?", "", regex=True)
    v = v.str.replace(r"\.0$", "", regex=True)
    return "PP-" + v


_BL_DATES: Dict[str, pd.Timestamp] = {}     # participant -> baseline (BL/SC) visit date, set by load_ppmi_native


def _visit(df: pd.DataFrame) -> pd.Series:
    """Visit key 'M<months>': scheduled EVENT_IDs from the PPMI schedule; any other code
    (remote R01.., unscheduled U01..) from the row's INFODT relative to the participant's
    baseline date when that is known."""
    c = _col(df, [r"^EVENT_ID$", r"^visit_name$", r"^event"])
    if c is None:
        return pd.Series("M0", index=df.index)
    ev = df[c].astype(str).str.strip().str.upper()
    m = pd.to_numeric(ev.map(PPMI_EVENT_MONTHS), errors="coerce")
    dcol = _col(df, [r"^INFODT$"])
    if dcol is not None and _BL_DATES and m.isna().any():
        pid = _pid(df)
        bl = pid.map(_BL_DATES)
        d = _date(df[dcol])
        derived = ((d - bl).dt.days / 30.4375).round()
        fill = m.isna() & derived.notna() & (derived >= 0)
        m = m.where(~fill, derived)
    return m.map(lambda x: f"M{int(x)}" if pd.notna(x) else pd.NA)


def _baseline_dates(paths: List[Optional[str]]) -> Dict[str, pd.Timestamp]:
    """Participant -> date of the BL visit (else SC, else earliest dated visit)."""
    rows = []
    for pth in paths:
        if not pth:
            continue
        try:
            df = _read(pth)
        except Exception:
            continue
        ev = _col(df, [r"^EVENT_ID$"]); dc = _col(df, [r"^INFODT$"])
        if ev is None or dc is None:
            continue
        rows.append(pd.DataFrame({"participant_id": _pid(df), "ev": df[ev].astype(str).str.upper(), "d": _date(df[dc])}))
    if not rows:
        return {}
    r = pd.concat(rows).dropna(subset=["d"])
    r["rank"] = r["ev"].map({"BL": 0, "SC": 1}).fillna(2)
    r = r.sort_values(["participant_id", "rank", "d"]).drop_duplicates("participant_id")
    return dict(zip(r["participant_id"], r["d"]))


# MDS-UPDRS items are scored 0-4; PPMI codes "unable to rate" as 101.  A section
# total is prorated from the available items when at most `allow_missing` items
# are missing (Goetz et al. 2015: Part I/II/IV one item, Part III seven items),
# otherwise it is missing.  Provided totals are only trusted within the section
# range and when no item carries a code.
_SECTION = {  # prefix regex -> (n_items, allow_missing, max_total)
    "p1r": (6, 1, 24), "p1p": (7, 1, 28), "p2": (13, 1, 52), "p3": (33, 7, 132), "p4": (6, 1, 24),
}


def _total(df: pd.DataFrame, total_pats: List[str], item_prefix: str, exclude: List[str],
           section: Optional[str] = None, log=None) -> pd.Series:
    """Section total from the items (0-4 each; codes > 4 = missing), prorated per
    the MDS-UPDRS rules; the provided total column is used only where the items
    are unavailable and it lies within the section range."""
    n_items, allow, max_total = _SECTION.get(section, (None, 0, 260))
    items = [k for k in df.columns if re.match(item_prefix, str(k), re.I)
             and not any(re.search(e, str(k), re.I) for e in exclude)]
    tot = pd.Series(np.nan, index=df.index)
    n_coded = 0
    if items:
        vals = df[items].apply(_num)
        coded = (vals > 4) | (vals < 0)
        n_coded = int(coded.any(axis=1).sum())
        vals = vals.mask(coded)
        n_avail = vals.notna().sum(axis=1)
        k = n_items or len(items)
        ok = (n_avail >= k - allow) & (n_avail > 0)
        tot = (vals.mean(axis=1) * k).where(ok)
    c = _col(df, total_pats)
    if c is not None:
        given = _num(df[c])
        given = given.where((given >= 0) & (given <= max_total))
        if items:
            # use the given total where the row carries no items at all (a file version
            # with totals only) or where every item was clean and complete
            k = n_items or len(items)
            tot = tot.where(tot.notna(), given.where((n_avail == 0) | (n_avail == k)))
        else:
            tot = given
    if log and n_coded:
        log(f"      {n_coded:,} rows had item codes outside 0-4 (e.g. 101 = unable to rate) -> treated as missing, "
            f"total prorated when <= {allow} item(s) missing")
    return tot


def _agg_visit(frame: pd.DataFrame, value: str, how: str = "mean") -> pd.DataFrame:
    frame = frame.dropna(subset=["visit_key"])
    return frame.groupby(["participant_id", "visit_key"], as_index=False)[value].agg(how)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  loaders                                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _updrs_tables(files: Dict[str, Optional[str]], prefer_off: bool, log) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Visit-level Part totals (+ H&Y, state, medicated) and visit dates."""
    pieces: List[pd.DataFrame] = []
    dates: List[pd.DataFrame] = []

    def _dates_from(df):
        c = _col(df, [r"^INFODT$", r"^visit_date", r"^date"])
        if c is not None:
            d = pd.DataFrame({"participant_id": _pid(df), "visit_key": _visit(df), "visit_date": _date(df[c])})
            dates.append(d.dropna(subset=["visit_key", "visit_date"]))

    # Part I = clinician-rated (NP1R) + patient questionnaire (NP1P)
    if files["part1"]:
        df = _read(files["part1"]); _dates_from(df)
        t = _total(df, [r"^NP1RTOT$", r"^NP1TOT$"], r"^NP1(COG|HALL|DPRS|ANXS|APAT|DDS)$", [], "p1r", log)
        p1 = _agg_visit(pd.DataFrame({"participant_id": _pid(df), "visit_key": _visit(df), "p1r": t}), "p1r")
        log(f"    [Part I clinician] {_name(files['part1'])}: {len(p1)} visits "
            f"(total col: {_col(df, [r'^NP1RTOT$', r'^NP1TOT$']) or 'summed items'})")
        if files["part1pq"]:
            dq = _read(files["part1pq"]); _dates_from(dq)
            tq = _total(dq, [r"^NP1PTOT$"], r"^NP1(SLPN|SLPD|PAIN|URIN|CNST|LTHD|FATG)$", [], "p1p", log)
            pq = _agg_visit(pd.DataFrame({"participant_id": _pid(dq), "visit_key": _visit(dq), "p1p": tq}), "p1p")
            log(f"    [Part I patient]   {_name(files['part1pq'])}: {len(pq)} visits")
            p1 = p1.merge(pq, on=["participant_id", "visit_key"], how="outer")
            p1["mds_updrs_part_i_total"] = p1["p1r"] + p1["p1p"]
        else:
            log("    [Part I patient]   not found -> Part I total will be clinician items only "
                "(6/13 items); updrs_total will be systematically low for these rows")
            p1["mds_updrs_part_i_total"] = p1["p1r"]
        pieces.append(p1[["participant_id", "visit_key", "mds_updrs_part_i_total"]])

    if files["part2"]:
        df = _read(files["part2"]); _dates_from(df)
        t = _total(df, [r"^NP2PTOT$", r"^NP2TOT$"], r"^NP2[A-Z]+$", [r"TOT"], "p2", log)
        p2 = _agg_visit(pd.DataFrame({"participant_id": _pid(df), "visit_key": _visit(df),
                                      "mds_updrs_part_ii_total": t}), "mds_updrs_part_ii_total")
        log(f"    [Part II]          {_name(files['part2'])}: {len(p2)} visits")
        pieces.append(p2)

    if files["part3"]:
        df = _read(files["part3"]); _dates_from(df)
        t = _total(df, [r"^NP3TOT$"], r"^NP3[A-Z]+$", [r"TOT"], "p3", log)
        fr = pd.DataFrame({"participant_id": _pid(df), "visit_key": _visit(df), "mds_updrs_part_iii_total": t})
        hy = _col(df, [r"^NHY$", r"hoehn"])
        fr["hoehn_yahr"] = _num(df[hy]).where(lambda v: (v >= 0) & (v <= 5)) if hy else np.nan
        st = _col(df, [r"^PDSTATE$", r"^ON_OFF", r"state"])
        state = pd.Series("UNK", index=df.index, dtype="object")
        if st:
            s = df[st].astype(str).str.upper()
            state = pd.Series(np.where(s.str.contains("OFF"), "OFF", np.where(s.str.contains(r"\bON\b"), "ON", "UNK")),
                              index=df.index, dtype="object")
        pag = _col(df, [r"^PAG_NAME$"])
        if pag:   # user guide: blank PDSTATE -> PAG_NAME NUPDR3OF (off) / NUPDR3ON (on)
            pg = df[pag].astype(str).str.upper()
            state = state.where(state != "UNK", np.where(pg.str.contains("3OF"), "OFF",
                                                         np.where(pg.str.contains("3ON"), "ON", "UNK")))
        fr["updrs3_state"] = state.values
        med = _col(df, [r"^PDTRTMNT$", r"^PDMEDYN$", r"^PDMEDUSE$"])
        if med:
            m = _num(df[med])
            s = df[med].astype(str).str.strip().str.lower()
            fr["pd_medicated"] = np.where(m.notna(), (m > 0).astype(float),
                                 np.where(s.isin(["y", "yes", "1"]), 1.0, np.where(s.isin(["n", "no", "0"]), 0.0, np.nan)))
        else:
            fr["pd_medicated"] = np.nan
        evc = _col(df, [r"^EVENT_ID$"])
        if evc is not None:
            unm = df.loc[fr["visit_key"].isna(), evc].astype(str).value_counts().head(10)
            if len(unm):
                log(f"    [Part III]         EVENT_IDs without a fixed month, rows dropped: {unm.to_dict()}")
        fr = fr.dropna(subset=["visit_key"])
        if prefer_off:
            fr["_off"] = (fr["updrs3_state"] == "OFF").astype(int)
            has_off = fr.groupby(["participant_id", "visit_key"])["_off"].transform("max")
            fr = fr[(fr["_off"] == 1) | (has_off == 0)].drop(columns="_off")
        p3 = (fr.groupby(["participant_id", "visit_key"], as_index=False)
                .agg(mds_updrs_part_iii_total=("mds_updrs_part_iii_total", "mean"),
                     hoehn_yahr=("hoehn_yahr", "mean"), updrs3_state=("updrs3_state", "first"),
                     pd_medicated=("pd_medicated", "max")))
        log(f"    [Part III]         {_name(files['part3'])}: {len(p3)} visits "
            f"(H&Y col: {hy}, state col: {st}, medicated col: {med}; "
            f"{int((p3['updrs3_state'] == 'OFF').sum())} OFF visits kept preferentially)")
        pieces.append(p3)

    if files["part4"]:
        df = _read(files["part4"]); _dates_from(df)
        t = _total(df, [r"^NP4TOT$"], r"^NP4[A-Z]+$", [r"TOT", r"DEN$", r"NUM$", r"PCT$"], "p4", log)
        p4 = _agg_visit(pd.DataFrame({"participant_id": _pid(df), "visit_key": _visit(df),
                                      "mds_updrs_part_iv_total": t}), "mds_updrs_part_iv_total")
        log(f"    [Part IV]          {_name(files['part4'])}: {len(p4)} visits")
        pieces.append(p4)

    if not pieces:
        return pd.DataFrame(columns=["participant_id", "visit_key"]), pd.DataFrame()
    base = pieces[0]
    for p in pieces[1:]:
        base = base.merge(p, on=["participant_id", "visit_key"], how="outer")
    vd = (pd.concat(dates).groupby(["participant_id", "visit_key"], as_index=False)["visit_date"].min()
          if dates else pd.DataFrame(columns=["participant_id", "visit_key", "visit_date"]))
    return base, vd


def _status(path: Optional[str], log) -> Optional[pd.DataFrame]:
    if not path:
        return None
    df = _read(path)
    out = pd.DataFrame({"participant_id": _pid(df)})
    cdef = _col(df, [r"^COHORT_DEFINITION$", r"cohort.*def", r"^APPRDX$", r"^diagnosis"])
    ccode = _col(df, [r"^COHORT$"])
    label = pd.Series("OTHER", index=df.index, dtype="object")
    diag = pd.Series(pd.NA, index=df.index, dtype="object")
    if cdef:
        s = df[cdef].astype(str)
        diag = s
        up = s.str.upper()
        label = np.where(up.str.contains("PARKINSON|^PD$|\\bPD\\b|EARLY IMAGING"), "CASE",
                np.where(up.str.contains("HEALTHY|CONTROL|^HC$"), "CONTROL", "OTHER"))
        label = pd.Series(label, index=df.index)
    elif ccode:
        code = _num(df[ccode])
        label = pd.Series(np.where(code == 1, "CASE", np.where(code == 2, "CONTROL", "OTHER")), index=df.index)
        diag = code.map({1: "Parkinson's Disease", 2: "Healthy Control", 3: "SWEDD", 4: "Prodromal"})
    out["case_control"] = label.values
    out["diagnosis"] = diag.values
    ea = _col(df, [r"^ENROLL_AGE$", r"enroll.*age"])
    out["enroll_age"] = _num(df[ea]) if ea else np.nan
    es = _col(df, [r"^ENROLL_STATUS$", r"enroll.*status"])
    out["enroll_status"] = df[es].astype(str).str.strip().str.lower().values if es else "enrolled"
    out = out.drop_duplicates("participant_id")
    log(f"    [Participant status] {_name(path)}: {len(out)} participants "
        f"{out['case_control'].value_counts().to_dict()} (cohort col: {cdef or ccode})")
    if cdef:
        log(f"      cohort definitions: {df[cdef].astype(str).value_counts().head(8).to_dict()}")
    if es:
        log(f"      enrol status: {df[es].astype(str).str.lower().value_counts().head(10).to_dict()}")
    return out


def _demographics(path: Optional[str], log) -> Optional[pd.DataFrame]:
    if not path:
        return None
    df = _read(path)
    out = pd.DataFrame({"participant_id": _pid(df)})
    sx = _col(df, [r"^SEX$", r"^gender$"])
    if sx:
        s = df[sx].astype(str).str.strip().str.upper()
        out["sex"] = np.where(s.isin(["1", "M", "MALE"]), "Male", np.where(s.isin(["0", "2", "F", "FEMALE"]), "Female", pd.NA))
    bd = _col(df, [r"^BIRTHDT$", r"birth"])
    out["birth_date"] = _date(df[bd]) if bd else pd.NaT
    out = out.drop_duplicates("participant_id")
    log(f"    [Demographics]     {_name(path)}: {len(out)} participants (sex col: {sx}, birth col: {bd})")
    return out


def _age_at_visit(path: Optional[str], log) -> Optional[pd.DataFrame]:
    if not path:
        return None
    df = _read(path)
    c = _col(df, [r"^AGE_AT_VISIT$", r"^age"])
    if c is None:
        return None
    out = _agg_visit(pd.DataFrame({"participant_id": _pid(df), "visit_key": _visit(df), "age": _num(df[c])}), "age")
    log(f"    [Age at visit]     {_name(path)}: {len(out)} visits")
    return out


def _diagnosis_date(path: Optional[str], log) -> Optional[pd.DataFrame]:
    if not path:
        return None
    df = _read(path)
    c = _col(df, [r"^PDDXDT$", r"dx.*dt", r"diagnosis.*date"])
    if c is None:
        return None
    out = pd.DataFrame({"participant_id": _pid(df), "dx_date": _date(df[c])}).dropna()
    out = out.sort_values("dx_date").drop_duplicates("participant_id")
    log(f"    [PD diagnosis]     {_name(path)}: {len(out)} participants with a diagnosis date")
    return out


def _upsit(path: Optional[str], log) -> Optional[pd.DataFrame]:
    if not path:
        return None
    df = _read(path)
    c = _col(df, [r"^TOTAL_CORRECT$", r"^UPSIT_TOTAL", r"total.*correct", r"^upsit_total"])
    if c is None:
        bk = [k for k in df.columns if re.match(r"^UPSITBK\d$", str(k), re.I)]
        if not bk:
            log(f"    [UPSIT]            {_name(path)}: no total / booklet columns -> skipped")
            return None
        tot = df[bk].apply(_num).sum(axis=1, min_count=len(bk))
    else:
        tot = _num(df[c])
    tot = tot.where((tot >= 0) & (tot <= 40))
    out = _agg_visit(pd.DataFrame({"participant_id": _pid(df), "visit_key": _visit(df), "upsit_total": tot}), "upsit_total")
    log(f"    [UPSIT]            {_name(path)}: {len(out)} visits (col: {c or 'booklets'})")
    return out


def _datscan(path: Optional[str], log) -> Optional[pd.DataFrame]:
    if not path:
        return None
    df = _read(path)
    # whole-region left/right columns only: AMP-PD style DATSCAN_PUTAMEN_R/L or the Xing
    # core-lab re-analysis PUTAMEN_L_REF_CWM / PUTAMEN_R_REF_CWM (sub-regions such as
    # PRECOMMISSURAL_PUTAMEN or PRECAUDATE are excluded)
    def _lr(region):
        rx = re.compile(rf"^(DATSCAN_)?{region}_(R|L|RIGHT|LEFT)(_REF_CWM)?$", re.I)
        return [c for c in df.columns if rx.match(str(c))]
    put, cau = _lr("PUTAMEN"), _lr("CAUDATE")
    if not put and not cau:
        return None
    cwm = any("CWM" in str(c).upper() for c in put + cau)
    if cwm:
        log("    [DaTSCAN] Xing core-lab SBR (cerebellar-white-matter reference): not on the same scale as "
            "the AMP-PD occipital-referenced SBR -> applied to ALL PPMI rows when ppmi_datscan_override is on")
    an = _col(df, [r"^DATSCAN_ANALYZED$"])
    if an:
        ok = df[an].astype(str).str.strip().str.lower().isin(["yes", "1", "true"])
        df = df[ok]
    out = pd.DataFrame({"participant_id": _pid(df), "visit_key": _visit(df)})
    if put:
        out["datscan_putamen"] = df[put].apply(_num).mean(axis=1)
    if cau:
        out["datscan_caudate"] = df[cau].apply(_num).mean(axis=1)
    if put and cau:
        out["datscan_striatum"] = out[["datscan_putamen", "datscan_caudate"]].mean(axis=1)
    out["visit_key"] = out["visit_key"].fillna("M0")
    vals = [c for c in out.columns if c.startswith("datscan_")]
    out = out.dropna(subset=vals, how="all").groupby(["participant_id", "visit_key"], as_index=False)[vals].mean()
    out.attrs["cwm"] = cwm
    log(f"    [DaTSCAN]          {_name(path)}: {len(out)} scans (putamen {put}, caudate {cau})")
    return out


def _ledd(path: Optional[str], log) -> Optional[pd.DataFrame]:
    if not path:
        return None
    df = _read(path)
    drug = _col(df, [r"^LEDTRT$", r"^CMTRT$", r"drug|medication|treatment"])
    if drug is None:
        return None
    pid = _pid(df)
    d = df[drug].astype(str)
    out = pd.DataFrame({"participant_id": pid,
                        "on_levodopa": d.str.contains(_LEVODOPA).astype(float),
                        "on_other_dopaminergic": d.str.contains(_AGONIST).astype(float)})
    out = out.groupby("participant_id", as_index=False).max()
    led = _col(df, [r"^LEDD$"])
    sd, ed = _col(df, [r"^STARTDT$"]), _col(df, [r"^STOPDT$"])
    spans = None
    n_nonnum = 0
    if led and sd:
        lv = _num(df[led])
        n_nonnum = int(df[led].notna().sum() - lv.notna().sum())   # e.g. "LD x 0.33" (COMT inhibitors)
        spans = pd.DataFrame({"participant_id": pid, "ledd": lv, "start": _date(df[sd]),
                              "stop": _date(df[ed]) if ed else pd.NaT}).dropna(subset=["ledd", "start"])
    log(f"    [LEDD log]         {_name(path)}: {int(out['on_levodopa'].sum())} ever on levodopa, "
        f"{int(out['on_other_dopaminergic'].sum())} on an agonist"
        + (f"; {len(spans)} dated medication spans with numeric LEDD"
           f"{f' ({n_nonnum} non-numeric LEDD entries skipped)' if n_nonnum else ''}" if spans is not None else ""))
    out.attrs["ledd_spans"] = spans
    return out


def _ledd_at_visits(spans: pd.DataFrame, vdates: pd.DataFrame) -> pd.DataFrame:
    """Total LEDD active at each dated visit (a span is active from STARTDT through the
    month of STOPDT, open-ended if STOPDT is blank) -> participant_id, visit_key, ledd."""
    if spans is None or vdates is None or not len(spans) or not len(vdates):
        return pd.DataFrame(columns=["participant_id", "visit_key", "ledd"])
    m = vdates.merge(spans, on="participant_id", how="inner")
    stop = m["stop"].fillna(pd.Timestamp("2100-01-01")) + pd.offsets.MonthEnd(0)
    act = (m["visit_date"] >= m["start"]) & (m["visit_date"] <= stop)
    tot = m[act].groupby(["participant_id", "visit_key"], as_index=False)["ledd"].sum()
    # dated visits with no active span are untreated (LEDD 0), not missing
    base = vdates[["participant_id", "visit_key"]].drop_duplicates()
    return base.merge(tot, on=["participant_id", "visit_key"], how="left").fillna({"ledd": 0.0})


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  assembly                                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def load_ppmi_native(root: str, prefer_off: bool = True, part_iv_missing_as_zero: bool = True,
                     log=print) -> Optional[pd.DataFrame]:
    """Rows in the clinical_unified schema from PPMI-native exports, or None."""
    if not root:
        return None
    if not Path(root).is_dir():
        log(f"  PPMI-native clinical: directory not found ({root}) -> skipped")
        return None
    files = find_files(root)
    log(f"  PPMI-native clinical: {root}")
    for k, (rx, req) in _FILES.items():
        if files[k] is None and req:
            log(f"    [missing] {k}: no file matching /{rx}/ -> "
                + ("cannot build UPDRS rows" if k.startswith("part") else "case/control unknown"))
    if not (files["part1"] and files["part2"] and files["part3"]):
        log("    -> Parts I, II and III are all needed for updrs_total; nothing appended")
        return None

    global _BL_DATES
    _BL_DATES = _baseline_dates([files["part3"], files["part1"], files["part2"]])
    log(f"    baseline visit dates for {len(_BL_DATES):,} participants (unscheduled EVENT_IDs such as "
        f"R01.. are keyed by months since that date)")
    upd, vdates = _updrs_tables(files, prefer_off, log)
    for p in ("i", "ii", "iii", "iv"):
        c = f"mds_updrs_part_{p}_total"
        if c not in upd.columns:
            upd[c] = np.nan
    if part_iv_missing_as_zero:
        upd["mds_updrs_part_iv_total"] = upd["mds_updrs_part_iv_total"].fillna(0.0)
    core = upd[["mds_updrs_part_i_total", "mds_updrs_part_ii_total", "mds_updrs_part_iii_total"]].notna().all(axis=1)
    upd["updrs_total"] = np.where(core & upd["mds_updrs_part_iv_total"].notna(),
                                  upd[[f"mds_updrs_part_{p}_total" for p in ("i", "ii", "iii", "iv")]].sum(axis=1), np.nan)
    for c in ("hoehn_yahr", "updrs3_state", "pd_medicated"):
        if c not in upd.columns:
            upd[c] = np.nan if c != "updrs3_state" else "UNK"

    st = _status(files["status"], log)
    demo = _demographics(files["demo"], log)
    age = _age_at_visit(files["age"], log)
    dx = _diagnosis_date(files["dx"], log)
    ups = _upsit(files["upsit"], log)
    dat = _datscan(files["datscan"], log)
    led = _ledd(files["ledd"], log)

    base = upd.copy()
    # participants known to status/demographics but without UPDRS: one M0 row (HC anchoring)
    known = set(base["participant_id"])
    extra = []
    for tbl in (st, demo):
        if tbl is not None:
            extra += [p for p in tbl["participant_id"] if p not in known]
    extra = sorted(set(extra))
    if extra:
        base = pd.concat([base, pd.DataFrame({"participant_id": extra, "visit_key": "M0"})], ignore_index=True)

    if st is not None:
        base = base.merge(st, on="participant_id", how="left")
        # user guide: analyses should use Enrolled / Withdrew / Complete only
        es_ = base["enroll_status"].astype(str).str.lower()
        keep = es_.str.startswith(("enrolled", "withdr", "complete")) | es_.str.contains("withdraw") | es_.isin(["nan", ""])
        bad = ~keep & base["enroll_status"].notna()
        if bad.any():
            log(f"    [enroll status] dropping {base.loc[bad, 'participant_id'].nunique()} participants with status "
                f"{sorted(base.loc[bad, 'enroll_status'].astype(str).unique())[:6]}")
            base = base[~bad]
    else:
        base["case_control"] = "UNKNOWN"; base["diagnosis"] = pd.NA; base["enroll_age"] = np.nan
    base["case_control"] = base["case_control"].fillna("UNKNOWN")
    if demo is not None:
        base = base.merge(demo, on="participant_id", how="left")
    else:
        base["sex"] = pd.NA; base["birth_date"] = pd.NaT
    base = base.merge(vdates, on=["participant_id", "visit_key"], how="left")
    base["visit_month"] = pd.to_numeric(base["visit_key"].astype(str).str.extract(r"M(-?\d+)", expand=False), errors="coerce")

    # age: Age_at_visit > birth date + visit date > enroll age + months
    if age is not None:
        base = base.merge(age, on=["participant_id", "visit_key"], how="left")
    else:
        base["age"] = np.nan
    if "birth_date" in base.columns:
        est = (base["visit_date"] - base["birth_date"]).dt.days / 365.25
        base["age"] = base["age"].fillna(est)
    base["age"] = base["age"].fillna(base["enroll_age"] + base["visit_month"] / 12.0)
    bl_age = (base.sort_values(["participant_id", "visit_month"]).dropna(subset=["age"])
                  .drop_duplicates("participant_id").set_index("participant_id"))
    base["age_at_baseline"] = base["participant_id"].map(bl_age["age"] - bl_age["visit_month"] / 12.0)

    # age at diagnosis from the diagnosis date and the baseline visit date
    base["age_at_diagnosis"] = np.nan
    if dx is not None:
        bl_date = (base.dropna(subset=["visit_date"]).sort_values(["participant_id", "visit_month"])
                       .drop_duplicates("participant_id").set_index("participant_id"))
        tmp = dx.set_index("participant_id")["dx_date"]
        d0 = base["participant_id"].map(bl_date["visit_date"])
        m0 = base["participant_id"].map(bl_date["visit_month"])
        yrs_before_bl = (d0 - base["participant_id"].map(tmp)).dt.days / 365.25
        base["age_at_diagnosis"] = base["age_at_baseline"] + m0 / 12.0 - yrs_before_bl
    base["disease_duration_years"] = base["age"] - base["age_at_diagnosis"]
    base.loc[base["disease_duration_years"] < -1, "disease_duration_years"] = np.nan
    base.loc[base["case_control"].eq("CONTROL"), "disease_duration_years"] = np.nan

    if ups is not None:
        base = base.merge(ups, on=["participant_id", "visit_key"], how="left")
        first = (ups.assign(m=pd.to_numeric(ups["visit_key"].str.extract(r"M(-?\d+)", expand=False), errors="coerce"))
                    .sort_values(["participant_id", "m"]).drop_duplicates("participant_id")
                    .set_index("participant_id")["upsit_total"])
        base["upsit_total"] = base["upsit_total"].fillna(base["participant_id"].map(first))
    else:
        base["upsit_total"] = np.nan
    if dat is not None:
        base = base.merge(dat, on=["participant_id", "visit_key"], how="left")
        bl = (dat.assign(m=pd.to_numeric(dat["visit_key"].str.extract(r"M(-?\d+)", expand=False), errors="coerce"))
                 .sort_values(["participant_id", "m"]).drop_duplicates("participant_id").set_index("participant_id"))
        for c in [c for c in dat.columns if c.startswith("datscan_")]:
            base[f"{c}_baseline"] = base["participant_id"].map(bl[c])
    if led is not None:
        base = base.merge(led, on="participant_id", how="left")
        lv = _ledd_at_visits(led.attrs.get("ledd_spans"), vdates)
        if len(lv):
            base = base.merge(lv, on=["participant_id", "visit_key"], how="left")
            log(f"    [LEDD at visit]    {int(base['ledd'].notna().sum())} rows, "
                f"{int((base['ledd'] > 0).sum())} with LEDD > 0")
    for c in ("on_levodopa", "on_other_dopaminergic", "ledd", "race", "ethnicity"):
        if c not in base.columns:
            base[c] = np.nan

    base = base.rename(columns={"visit_key": "visit_name"})
    base["cohort"] = "TRAIN"           # PPMI ids carry the PP- prefix
    base["site"] = "PP"
    base["source"] = "ppmi_native"
    base = base.dropna(subset=["visit_name"])
    base.attrs["datscan_cwm"] = bool(dat.attrs.get("cwm", False)) if dat is not None else False
    try:
        from .config import PPMI_CONVERSION_CODES
    except Exception:
        PPMI_CONVERSION_CODES = [1]
    base.attrs["conversion"] = build_conversion_table(files, st, list(PPMI_CONVERSION_CODES), log)
    log(f"    -> {len(base)} participant-visits, {base['participant_id'].nunique()} participants, "
        f"updrs_total on {int(np.isfinite(base['updrs_total']).sum())} rows; "
        f"{base['case_control'].value_counts().to_dict()}")
    return base


def build_conversion_table(files: Dict[str, Optional[str]], status: Optional[pd.DataFrame],
                           codes: List[int], log=print) -> Optional[pd.DataFrame]:
    """Phenoconversion of prodromal enrollees.

    event = first visit after baseline at which the research or clinical diagnosis
    is one of `codes` (default 1 = idiopathic PD); censor = last dated visit in any
    diagnosis / UPDRS file.  Returns one row per prodromal participant.
    """
    if status is None or not files.get("primary_dx") and not files.get("clinical_dx"):
        return None
    pro = status[status["diagnosis"].astype(str).str.upper().str.contains("PRODROMAL")]
    if pro.empty:
        return None
    # subgroup flags from Participant_Status (hyposmia, RBD, genetic)
    st = _read(files["status"])
    flags = pd.DataFrame({"participant_id": _pid(st)})
    for c in ("ENRLHPSM", "ENRLRBD", "ENRLLRRK2", "ENRLGBA", "ENRLSNCA", "ENRLPRKN", "ENRLPINK1"):
        if c in st.columns:
            flags[c.lower()] = _num(st[c])
    flags = flags.drop_duplicates("participant_id")
    dx_rows, last_rows = [], []
    for key, col in (("primary_dx", "PRIMDIAG"), ("clinical_dx", "NEWDIAG")):
        if not files.get(key):
            continue
        d = _read(files[key])
        if col not in d.columns:
            continue
        dc = _col(d, [r"^INFODT$"]); dd = _col(d, [r"^DIAGDT$"])
        when = _date(d[dc]) if dc else pd.NaT
        if dd:
            when = _date(d[dd]).fillna(when)
        fr = pd.DataFrame({"participant_id": _pid(d), "code": _num(d[col]), "when": when, "source": key})
        last_rows.append(fr[["participant_id", "when"]])
        dx_rows.append(fr[fr["code"].isin(codes)])
    for key in ("part3", "part1", "part2"):
        if files.get(key):
            d = _read(files[key]); dc = _col(d, [r"^INFODT$"])
            if dc:
                last_rows.append(pd.DataFrame({"participant_id": _pid(d), "when": _date(d[dc])}))
    dx = pd.concat(dx_rows) if dx_rows else pd.DataFrame(columns=["participant_id", "code", "when", "source"])
    last = pd.concat(last_rows).dropna().groupby("participant_id")["when"].max()
    out = pro[["participant_id", "diagnosis"]].merge(flags, on="participant_id", how="left")
    out["baseline_date"] = out["participant_id"].map(_BL_DATES)
    first = dx.dropna(subset=["when"]).sort_values("when").drop_duplicates("participant_id").set_index("participant_id")
    out["event_date"] = out["participant_id"].map(first["when"])
    out["converted_code"] = out["participant_id"].map(first["code"])
    out["converted_source"] = out["participant_id"].map(first["source"])
    out["last_visit_date"] = out["participant_id"].map(last)
    months = lambda a, b: (a - b).dt.days / 30.4375
    out["event"] = out["event_date"].notna().astype(int)
    out["months"] = np.where(out["event"] == 1, months(out["event_date"], out["baseline_date"]),
                             months(out["last_visit_date"], out["baseline_date"]))
    out["prevalent_at_baseline"] = (out["event"] == 1) & (out["months"] <= 0)
    vis = files.get("datscan_visual")
    if vis:
        v = _read(vis); vc = _col(v, [r"VISINTRP"]); dc = _col(v, [r"DATSCAN_DATE", r"^INFODT$"])
        if vc:
            vv = pd.DataFrame({"participant_id": _pid(v), "read": v[vc].astype(str).str.lower(),
                               "when": _date(v[dc]) if dc else pd.NaT}).sort_values("when").drop_duplicates("participant_id")
            out["datscan_visual_baseline"] = out["participant_id"].map(
                vv.set_index("participant_id")["read"].map({"positive": 1.0, "negative": 0.0}))
    n_ev = int(((out["event"] == 1) & ~out["prevalent_at_baseline"]).sum())
    log(f"    [conversion]       {len(out)} prodromal enrollees; {n_ev} converted to codes {codes} after baseline "
        f"({int(out['prevalent_at_baseline'].sum())} already diagnosed at baseline, excluded); "
        f"median follow-up {np.nanmedian(out.loc[out['event'] == 0, 'months']):.0f} months among non-converters")
    return out


def merge_native(base: pd.DataFrame, native: pd.DataFrame, mode: str, log=print) -> pd.DataFrame:
    """Append PPMI-native rows the AMP-PD table lacks.

    mode = "fill":         add every (participant, visit) absent from AMP-PD
    mode = "participants": add only participants absent from AMP-PD altogether
    AMP-PD rows are never replaced.  Agreement on overlapping visits is printed
    as a check that both sources describe the same examinations.
    """
    key = ["participant_id", "visit_name"]
    amp_keys = set(map(tuple, base[key].astype(str).values))
    amp_pids = set(base["participant_id"].astype(str))
    nat = native.copy()
    nat["_key"] = list(map(tuple, nat[key].astype(str).values))
    overlap = nat[nat["_key"].isin(amp_keys)]
    if len(overlap):
        ov = overlap.merge(base[key + ["updrs_total", "hoehn_yahr"]], on=key, suffixes=("_nat", "_amp"))
        m = ov[["updrs_total_nat", "updrs_total_amp"]].dropna()
        if len(m) >= 10:
            r = float(np.corrcoef(m["updrs_total_nat"], m["updrs_total_amp"])[0, 1])
            d = (m["updrs_total_nat"] - m["updrs_total_amp"]).abs()
            log(f"    [agreement] {len(m)} overlapping visits with UPDRS in both sources: "
                f"r = {r:.3f}, median |diff| = {d.median():.1f}, >5 points on {int((d > 5).sum())} visits")
            if r < 0.95:
                log("    [agreement] WARNING r < 0.95 -> check Part I patient questionnaire / OFF-state "
                    "handling before trusting the appended rows")
    if mode == "participants":
        add = nat[~nat["participant_id"].astype(str).isin(amp_pids)]
    else:
        add = nat[~nat["_key"].isin(amp_keys)]
    add = add.drop(columns="_key")
    n_new_p = add.loc[~add["participant_id"].isin(amp_pids), "participant_id"].nunique()
    n_new_v = int(add["participant_id"].isin(amp_pids).sum())
    log(f"    [append] mode={mode}: +{len(add)} rows = {n_new_p} new participants "
        f"({int(np.isfinite(add.loc[~add['participant_id'].isin(amp_pids), 'updrs_total']).sum())} rows with UPDRS) "
        f"and {n_new_v} new visits for {add.loc[add['participant_id'].isin(amp_pids), 'participant_id'].nunique()} "
        f"AMP-PD participants; {len(overlap)} overlapping visits kept from AMP-PD")
    if "source" not in base.columns:
        base = base.assign(source="amp_pd")
    out = pd.concat([base, add], ignore_index=True, sort=False)
    return out


def describe(root: str) -> None:
    files = find_files(root)
    print(f"PPMI-native clinical directory: {root}")
    used = set()
    for k, (rx, req) in _FILES.items():
        print(f"  {k:<14} {'required' if req else 'optional':<8} -> {_name(files[k]) if files[k] else '(not found)'}")
        if files[k]:
            used.update(files[k])
            try:
                head = _read(files[k][-1], nrows=3)
                n = sum(sum(1 for _ in open(f, "rb")) - 1 for f in files[k] if Path(f).suffix.lower() == ".csv") or "?"
                print(f"      rows={n}  columns({len(head.columns)}): {list(head.columns)[:45]}")
                ex = head.iloc[0].to_dict() if len(head) else {}
                print(f"      first row: {dict(list(ex.items())[:14])}")
            except Exception as e:
                print(f"      (could not read: {type(e).__name__}: {e})")
    others = [p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in _TABULAR and str(p) not in used]
    if others:
        print("  not used:", ", ".join(p.name for p in others))
    print()
    # value distributions the phenoconversion endpoint will rely on
    for key, cols in (("primary_dx", ["PRIMDIAG", "DXLVL"]), ("clinical_dx", ["NEWDIAG", "DIAGSINLV"]),
                      ("prodromal", ["PRIMDIAG", "PSLVL2"]), ("datscan_visual", ["DATSCAN_VISINTRP"])):
        if files[key]:
            try:
                d = _read(files[key])
                for c in cols:
                    if c in d.columns:
                        print(f"  {key}.{c}: {d[c].astype(str).value_counts().head(25).to_dict()}")
                if key == "primary_dx" and "PRIMDIAG" in d.columns and "PATNO" in d.columns:
                    ev = d["EVENT_ID"].astype(str).value_counts().head(15).to_dict() if "EVENT_ID" in d.columns else {}
                    print(f"  primary_dx visits: {ev}")
            except Exception as e:
                print(f"  ({key}: {type(e).__name__}: {e})")
    if files["curated"]:
        try:
            xl = pd.ExcelFile(files["curated"])
            print(f"  curated workbook sheets: {xl.sheet_names}")
            for sh in xl.sheet_names:
                dd = pd.read_excel(xl, sheet_name=sh, dtype=str)
                if "Variable" in dd.columns and "Code" in dd.columns:
                    sub = dd[dd["Variable"].astype(str).str.upper().isin(["PRIMDIAG", "NEWDIAG", "COHORT", "COHORT_DEFINITION", "SUBGROUP", "CONPD", "PHENOCNV", "PHENOCONV", "DIAGSINLV"])]
                    for _, r in sub.iterrows():
                        code = str(r.get("Code")).replace("\n", " | "); dec = str(r.get("Decode")).replace("\n", " | ")
                        print(f"  dict {r['Variable']}: codes = {code[:600]}")
                        print(f"       decode = {dec[:1200]}")
                    break
        except Exception as e:
            print(f"  (curated: {type(e).__name__}: {e})")
    print()
    df = load_ppmi_native(root)
    if df is not None:
        print()
        print(df.groupby("case_control")["participant_id"].nunique().rename("participants").to_string())
        print()
        print("visits:", df["visit_name"].value_counts().head(12).to_dict())
        print("with UPDRS:", int(np.isfinite(df["updrs_total"]).sum()), "rows;",
              "age:", int(df["age"].notna().sum()), "rows;",
              "disease duration:", int(df["disease_duration_years"].notna().sum()), "rows")


if __name__ == "__main__":
    describe(sys.argv[1] if len(sys.argv) > 1 else ".")

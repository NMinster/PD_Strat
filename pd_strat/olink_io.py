"""
Readers and column normalisation for Olink files in the two layouts we meet:

* **AMP-PD harmonised** (releases_2023_v4 …_olink-explore_protein-expression_*.csv):
  participant_id, sample_id, visit_name, visit_month, UniProt, Assay, NPX,
  Cumulative_QC, panel

* **PPMI native** (Project 9000 / 196 / 222 CSV, Project 293 / 277 / 314 parquet,
  Project 318 Target-48 CSV, Project 214 prodromal):
  SAMPLEID, PATNO ("PPMI-3004" or 3004), EVENT_ID (BL, V04, …), OLINKID,
  UNIPROT, ASSAY, MISSINGFREQ, PANEL, PLATEID, QC_WARNING, LOD, NPX

``normalize_olink`` maps the second onto the first so every downstream step
(visit matching, QC filter, gene symbols) is layout-agnostic.  PPMI PATNOs are
rewritten to the AMP-PD ``PP-<patno>`` form so the clinical join works for
every participant present in the AMP-PD clinical tables.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# PPMI visit schedule (EVENT_ID -> months from baseline).  Screening (SC) is
# keyed to baseline; unscheduled / withdrawal visits carry no fixed month.
PPMI_EVENT_MONTHS: Dict[str, float] = {
    "SC": 0, "BL": 0, "RS1": 0,
    "V01": 3, "V02": 6, "V03": 9, "V04": 12, "V05": 18, "V06": 24, "V07": 30,
    "V08": 36, "V09": 42, "V10": 48, "V11": 54, "V12": 60, "V13": 72, "V14": 84,
    "V15": 96, "V16": 108, "V17": 120, "V18": 132, "V19": 144, "V20": 156,
    "V21": 168, "V22": 180, "V23": 192,
}

_COL_ALIASES = {
    # target name          : candidate source names (case-insensitive exact match)
    "participant_id": ["participant_id", "patno", "participant", "subject_id", "guid"],
    "sample_id":      ["sample_id", "sampleid", "sample", "sample_name"],
    "visit_name":     ["visit_name", "event_id", "visit", "clinical_event", "event"],
    "visit_month":    ["visit_month", "month", "months", "visit_months"],
    "UniProt":        ["uniprot", "uniprot_id", "uniprotid", "uniprot_accession"],
    "Assay":          ["assay", "gene", "gene_name", "gene_symbol", "symbol", "hgnc"],
    "NPX":            ["npx", "pcnormalizednpx", "extnpx", "value", "abundance"],
    "Cumulative_QC":  ["cumulative_qc", "qc_warning", "sampleqc", "sample_qc", "qc", "assayqc"],
    "panel":          ["panel", "panel_name", "block"],
    "OlinkID":        ["olinkid", "olink_id"],
}


def read_table(path: str, **kw) -> pd.DataFrame:
    """CSV / TSV / parquet / xlsx by extension."""
    ext = Path(path).suffix.lower()
    if ext in (".parquet", ".pq"):
        try:
            return pd.read_parquet(path)
        except ImportError as e:  # pragma: no cover
            raise ImportError("Reading .parquet needs pyarrow: conda install -c conda-forge pyarrow "
                              "(or pip install pyarrow)") from e
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name=kw.pop("sheet_name", 0))
    if ext in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t", low_memory=False, **kw)
    return pd.read_csv(path, low_memory=False, **kw)


def ppmi_pid_to_amp(s: pd.Series) -> pd.Series:
    """'PPMI-3004' / 'ppmi3004' / 3004 / '3004' -> 'PP-3004'; AMP-PD ids untouched."""
    v = s.astype(str).str.strip().str.upper()
    is_amp = v.str.match(r"^(PP|PD|BF|HB|LB|LC|SU|SY)-")
    digits = v.str.replace(r"^PPMI[-_ ]?", "", regex=True).str.replace(r"\.0$", "", regex=True)
    out = np.where(is_amp, v, "PP-" + digits)
    return pd.Series(out, index=s.index, dtype="object")


def event_to_month(s: pd.Series) -> pd.Series:
    """PPMI EVENT_ID -> month (NaN for unscheduled / unknown codes)."""
    v = s.astype(str).str.strip().str.upper()
    out = v.map(PPMI_EVENT_MONTHS)
    # already an M<n> / MONTH n token or a bare number
    tok = pd.to_numeric(v.str.extract(r"^M(-?\d{1,3})$", expand=False), errors="coerce")
    num = pd.to_numeric(v, errors="coerce")
    return pd.to_numeric(out, errors="coerce").fillna(tok).fillna(num)


def _pick(df: pd.DataFrame, target: str) -> Optional[str]:
    low = {str(c).strip().lower(): c for c in df.columns}
    for cand in _COL_ALIASES[target]:
        if cand in low:
            return low[cand]
    return None


def normalize_olink(df: pd.DataFrame, path: str = "", label: str = "") -> pd.DataFrame:
    """Return a copy with AMP-PD-style columns whatever the input layout.

    Adds ``layout`` in {"amp_pd", "ppmi"} to the attrs for logging.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    is_ppmi = _pick(df, "participant_id") is not None and \
        str(_pick(df, "participant_id")).lower() == "patno"
    ren = {}
    for tgt in _COL_ALIASES:
        src = _pick(df, tgt)
        if src is not None and src != tgt and tgt not in df.columns:
            ren[src] = tgt
    df = df.rename(columns=ren)
    if "participant_id" not in df.columns:
        raise KeyError(f"{Path(path).name}: no participant column (looked for "
                       f"{_COL_ALIASES['participant_id']}); columns = {list(df.columns)[:15]}")
    if is_ppmi:
        df["participant_id"] = ppmi_pid_to_amp(df["participant_id"])
        if "visit_name" in df.columns:
            m = event_to_month(df["visit_name"])
            if "visit_month" in df.columns:
                df["visit_month"] = pd.to_numeric(df["visit_month"], errors="coerce").fillna(m)
            else:
                df["visit_month"] = m
            unk = df.loc[df["visit_month"].isna(), "visit_name"].astype(str).value_counts().head(6)
            if len(unk):
                print(f"    [{label or Path(path).name}] EVENT_IDs without a fixed month "
                      f"(rows dropped from visit matching): {unk.to_dict()}")
    if "Cumulative_QC" in df.columns:
        q = df["Cumulative_QC"].astype(str).str.strip().str.upper()
        # Olink flags: PASS / WARN / MANUAL_WARN / EXCLUDED ; AMP-PD: PASS / FAIL
        df["Cumulative_QC"] = np.where(q.isin(["PASS", "OK", "NONE", "", "NAN"]), "PASS", q)
    if "NPX" in df.columns:
        df["NPX"] = pd.to_numeric(df["NPX"], errors="coerce")
    if "UniProt" in df.columns:
        df["UniProt"] = df["UniProt"].astype(str).str.strip()
    df.attrs["layout"] = "ppmi" if is_ppmi else "amp_pd"
    return df


def describe_table(path: str, n: int = 3) -> str:
    """Schema summary used by `python -m pd_strat.inspect_file`."""
    df = read_table(path)
    lines = [f"{path}", f"  rows={len(df):,}  cols={df.shape[1]}", "  columns / dtype / n_unique / example:"]
    for c in df.columns:
        s = df[c]
        try:
            nu = s.nunique(dropna=True)
        except Exception:
            nu = -1
        ex = s.dropna().astype(str).head(2).tolist()
        lines.append(f"    {c:<28} {str(s.dtype):<10} {nu:>9,}  {ex}")
    lines.append("  head:")
    lines.append(df.head(n).to_string(max_cols=20, max_colwidth=24))
    try:
        nd = normalize_olink(df, path)
        lines.append(f"  normalised layout: {nd.attrs.get('layout')}; "
                     f"participants={nd['participant_id'].nunique():,}"
                     + (f"; proteins={nd['UniProt'].nunique():,}" if 'UniProt' in nd else "")
                     + (f"; visit_month resolved on {int(nd['visit_month'].notna().sum()):,}/{len(nd):,} rows"
                        if 'visit_month' in nd else ""))
        if "visit_name" in nd:
            lines.append(f"  visits: {nd['visit_name'].astype(str).value_counts().head(12).to_dict()}")
        if "panel" in nd:
            lines.append(f"  panels: {nd['panel'].astype(str).value_counts().head(12).to_dict()}")
        if "Cumulative_QC" in nd:
            lines.append(f"  QC: {nd['Cumulative_QC'].astype(str).value_counts().to_dict()}")
    except Exception as e:
        lines.append(f"  (not an Olink long table: {type(e).__name__}: {e})")
    return "\n".join(lines)

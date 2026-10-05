"""
Readers and column normalisation for Olink files in the two layouts we meet:

* **AMP-PD harmonised** (releases_2023_v4 …_olink-explore_protein-expression_*.csv):
  participant_id, sample_id, visit_name, visit_month, UniProt, Assay, NPX,
  Cumulative_QC, panel

* **PPMI native** (Project 9000 / 196 / 222 CSV, Project 293 / 277 / 314 parquet,
  Project 318 Target-48 CSV, Project 214 prodromal):
  SAMPLEID, PATNO ("PPMI-3004" or 3004), EVENT_ID (BL, V04, …), OLINKID,
  UNIPROT, ASSAY, MISSINGFREQ, PANEL, PLATEID, QC_WARNING, LOD, NPX
  (Explore HT adds SampleType, AssayType, Block, SampleQC, AssayQC)

``normalize_olink`` maps the second onto the first so every downstream step
(visit matching, QC filter, gene symbols) is layout-agnostic.  PPMI PATNOs are
rewritten to the AMP-PD ``PP-<patno>`` form so the clinical join works for
every participant present in the AMP-PD clinical tables.

Memory: Explore HT releases are 5-22 million rows.  ``read_olink`` prunes to
the columns the pipeline uses, filters plate controls and control assays
inside pyarrow, and keeps string columns categorical; ``normalize_olink``
works on unique values and never expands a categorical to object.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional

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
    "panel":          ["panel", "panel_name"],
    "OlinkID":        ["olinkid", "olink_id"],
}
# extra columns kept when present (filters / diagnostics)
_EXTRA_KEEP = ["sampletype", "assaytype", "assayqc", "assay_qc", "block", "plateid", "tissue_type"]
_CTRL_ASSAY_RE = re.compile(r"^(EXT|INC|AMP|DET|CTRL)\d*$", re.I)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  readers                                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _wanted_columns(all_cols: List[str]) -> List[str]:
    low = {str(c).strip().lower(): c for c in all_cols}
    keep: List[str] = []
    for cands in _COL_ALIASES.values():
        for c in cands:
            if c in low and low[c] not in keep:
                keep.append(low[c])
    for c in _EXTRA_KEEP:
        if c in low and low[c] not in keep:
            keep.append(low[c])
    return keep


def read_table(path: str, columns: Optional[List[str]] = None, **kw) -> pd.DataFrame:
    """CSV / TSV / parquet / xlsx by extension (full table unless `columns`)."""
    ext = Path(path).suffix.lower()
    if ext in (".parquet", ".pq"):
        try:
            return pd.read_parquet(path, columns=columns)
        except ImportError as e:  # pragma: no cover
            raise ImportError("Reading .parquet needs pyarrow: conda install -c conda-forge pyarrow "
                              "(or pip install pyarrow)") from e
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name=kw.pop("sheet_name", 0), usecols=columns)
    sep = "\t" if ext in (".tsv", ".txt") else ","
    return pd.read_csv(path, sep=sep, low_memory=False, usecols=columns, **kw)


def table_columns(path: str) -> List[str]:
    ext = Path(path).suffix.lower()
    if ext in (".parquet", ".pq"):
        import pyarrow.parquet as pq
        return list(pq.read_schema(path).names)
    if ext in (".xlsx", ".xls"):
        return list(pd.read_excel(path, nrows=0).columns)
    sep = "\t" if ext in (".tsv", ".txt") else ","
    return list(pd.read_csv(path, sep=sep, nrows=0).columns)


def read_olink(path: str) -> pd.DataFrame:
    """Memory-light read of an Olink long table: only the needed columns,
    plate controls / control assays filtered at read time, strings categorical."""
    ext = Path(path).suffix.lower()
    cols = _wanted_columns(table_columns(path))
    low = {str(c).lower(): c for c in cols}
    if ext in (".parquet", ".pq"):
        import pyarrow.parquet as pq
        import pyarrow.compute as pc
        import pyarrow.dataset as ds
        filt = None
        if "sampletype" in low:
            filt = pc.field(low["sampletype"]) == "SAMPLE"
        if "assaytype" in low:
            f2 = pc.field(low["assaytype"]) == "assay"
            filt = f2 if filt is None else (filt & f2)
        tbl = ds.dataset(path, format="parquet").to_table(columns=cols, filter=filt)
        df = tbl.to_pandas(strings_to_categorical=True, self_destruct=True)
        del tbl
        return df
    df = read_table(path, columns=cols)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype("category")
    return df


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  normalisation                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _map_unique(s: pd.Series, fn: Callable[[object], object]) -> pd.Series:
    """Apply fn to the unique values only and map back (categorical-safe)."""
    if isinstance(s.dtype, pd.CategoricalDtype):
        cats = s.cat.categories
        mapping = {c: fn(c) for c in cats}
        out = s.map(mapping)
        return out
    uniq = pd.unique(s)
    mapping = {u: fn(u) for u in uniq}
    return s.map(mapping)


def _pid_to_amp_one(v) -> str:
    t = str(v).strip().upper()
    if t in ("", "NAN", "NONE", "<NA>"):
        return ""
    if re.match(r"^(PP|PD|BF|HB|LB|LC|SU|SY)-", t):
        return t
    t = re.sub(r"^PPMI[-_ ]?", "", t)
    t = re.sub(r"\.0$", "", t)
    return f"PP-{t}"


def ppmi_pid_to_amp(s: pd.Series) -> pd.Series:
    """'PPMI-3004' / 'ppmi3004' / 3004.0 / '3004' -> 'PP-3004'; AMP-PD ids untouched."""
    return _map_unique(s, _pid_to_amp_one)


def _event_to_month_one(v) -> float:
    t = str(v).strip().upper()
    if t in PPMI_EVENT_MONTHS:
        return float(PPMI_EVENT_MONTHS[t])
    m = re.match(r"^M(-?\d{1,3})$", t)
    if m:
        return float(m.group(1))
    try:
        return float(t)
    except ValueError:
        return np.nan


def event_to_month(s: pd.Series) -> pd.Series:
    """PPMI EVENT_ID -> month (NaN for unscheduled / unknown codes)."""
    return pd.to_numeric(_map_unique(s, _event_to_month_one), errors="coerce")


def _pick(df: pd.DataFrame, target: str) -> Optional[str]:
    low = {str(c).strip().lower(): c for c in df.columns}
    for cand in _COL_ALIASES[target]:
        if cand in low:
            return low[cand]
    return None


def _upper_unique(s: pd.Series) -> pd.Series:
    return _map_unique(s, lambda v: str(v).strip().upper())


def normalize_olink(df: pd.DataFrame, path: str = "", label: str = "") -> pd.DataFrame:
    """Return a copy with AMP-PD-style columns whatever the input layout.

    Adds ``layout`` in {"amp_pd", "ppmi"} to the attrs for logging.
    """
    df = df.copy(deep=False)
    df.columns = [str(c).strip() for c in df.columns]
    pid_src = _pick(df, "participant_id")
    is_ppmi = pid_src is not None and str(pid_src).lower() == "patno"
    ren = {}
    for tgt in _COL_ALIASES:
        src = _pick(df, tgt)
        if src is not None and src != tgt and tgt not in df.columns:
            ren[src] = tgt
    df = df.rename(columns=ren)
    if "participant_id" not in df.columns:
        raise KeyError(f"{Path(path).name}: no participant column (looked for "
                       f"{_COL_ALIASES['participant_id']}); columns = {list(df.columns)[:15]}")
    low = {str(c).lower(): c for c in df.columns}

    # ── plate controls / control assays / no participant ───────────────
    n0 = len(df)
    keep = pd.Series(True, index=df.index)
    if "sampletype" in low:
        st = _upper_unique(df[low["sampletype"]])
        keep &= st.isin(["SAMPLE", "NAN", ""]).values
    if "assaytype" in low:
        at = _map_unique(df[low["assaytype"]], lambda v: str(v).strip().lower())
        keep &= at.isin(["assay", "nan", ""]).values
    if "UniProt" in df.columns:
        bad = {u for u in pd.unique(df["UniProt"].astype(object)) if _CTRL_ASSAY_RE.match(str(u))}
        if bad:
            keep &= ~df["UniProt"].isin(bad).values
    pid_txt = _map_unique(df["participant_id"], lambda v: str(v).strip().lower())
    keep &= ~pid_txt.isin(["", "nan", "none", "<na>"]).values
    if not keep.all():
        df = df[keep.values]
        print(f"    [{label or Path(path).name}] dropped {n0 - len(df):,} control-sample / "
              f"control-assay / no-participant rows ({len(df):,} kept)")

    if is_ppmi:
        df["participant_id"] = ppmi_pid_to_amp(df["participant_id"])
        if "visit_name" in df.columns:
            m = event_to_month(df["visit_name"])
            if "visit_month" in df.columns:
                df["visit_month"] = pd.to_numeric(df["visit_month"], errors="coerce").fillna(m)
            else:
                df["visit_month"] = m
            und = df.loc[df["visit_month"].isna(), "visit_name"]
            if len(und):
                unk = und.astype(str).value_counts().head(6)
                print(f"    [{label or Path(path).name}] EVENT_IDs without a fixed month "
                      f"(rows dropped from visit matching): {unk.to_dict()}")
    if "Cumulative_QC" in df.columns:
        q = _map_unique(df["Cumulative_QC"],
                        lambda v: "PASS" if str(v).strip().upper() in ("PASS", "OK", "NONE", "", "NAN")
                        else str(v).strip().upper())
        # Explore HT carries a separate assay-level flag; a WARN/FAIL assay is
        # dropped for every sample (it is the protein that failed, not the sample)
        aq_col = next((low[c] for c in ("assayqc", "assay_qc") if c in low
                       and low[c] in df.columns and low[c] != "Cumulative_QC"), None)
        if aq_col is not None:
            aq = _upper_unique(df[aq_col])
            bad = aq.isin(["WARN", "FAIL", "MANUAL_WARN", "EXCLUDED"]).values & (q.astype(object) == "PASS").values
            if bad.any():
                q = q.astype(object)
                q[bad] = "ASSAY_" + aq.astype(object)[bad]
        q = q.astype(object)
        q[pd.isna(q)] = "PASS"          # no flag recorded = not flagged
        df["Cumulative_QC"] = pd.Categorical(q)
    if "NPX" in df.columns:
        df["NPX"] = pd.to_numeric(df["NPX"], errors="coerce")
    if "UniProt" in df.columns:
        df["UniProt"] = _map_unique(df["UniProt"], lambda v: str(v).strip())
    df.attrs["layout"] = "ppmi" if is_ppmi else "amp_pd"
    return df


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  inspection                                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _sample_rows(path: str, n: int = 200_000) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext in (".parquet", ".pq"):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        return next(pf.iter_batches(batch_size=n)).to_pandas()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, nrows=n)
    sep = "\t" if ext in (".tsv", ".txt") else ","
    return pd.read_csv(path, sep=sep, nrows=n, low_memory=False)


def describe_table(path: str, n: int = 3) -> str:
    """Schema summary used by `python -m pd_strat.inspect_file`."""
    head = _sample_rows(path)
    lines = [f"{path}", f"  cols={head.shape[1]}  (column listing from the first {len(head):,} rows)",
             "  columns / dtype / n_unique(sample) / example:"]
    for c in head.columns:
        s = head[c]
        try:
            nu = s.nunique(dropna=True)
        except Exception:
            nu = -1
        ex = s.dropna().astype(str).head(2).tolist()
        lines.append(f"    {c:<28} {str(s.dtype):<10} {nu:>9,}  {ex}")
    lines.append("  head:")
    lines.append(head.head(n).to_string(max_cols=20, max_colwidth=24))
    try:
        nd = normalize_olink(read_olink(path), path)
        lines.append(f"  rows kept after control filtering: {len(nd):,}")
        lines.append(f"  normalised layout: {nd.attrs.get('layout')}; "
                     f"participants={nd['participant_id'].nunique():,}"
                     + (f"; proteins={nd['UniProt'].nunique():,}" if 'UniProt' in nd else "")
                     + (f"; visit_month resolved on {int(nd['visit_month'].notna().sum()):,}/{len(nd):,} rows"
                        if 'visit_month' in nd else ""))
        if "visit_name" in nd:
            lines.append(f"  visits: {nd['visit_name'].astype(str).value_counts().head(12).to_dict()}")
        if "participant_id" in nd and "visit_month" in nd:
            keys = ["participant_id", "visit_name"] if "visit_name" in nd else ["participant_id"]
            samp = nd[keys + ["visit_month"]].drop_duplicates(keys)
            n_ok = int(samp["visit_month"].notna().sum())
            lines.append(f"  samples (participant x visit): {len(samp):,}; with a fixed month: {n_ok:,}; "
                         f"participants with >= 2 dated samples: "
                         f"{int((samp.dropna(subset=['visit_month']).groupby('participant_id', observed=True).size() >= 2).sum()):,}")
            if "visit_name" in nd:
                und = samp[samp["visit_month"].isna()]["visit_name"].astype(str).value_counts().head(6)
                if len(und):
                    lines.append(f"  undated samples by EVENT_ID: {und.to_dict()}")
        if "panel" in nd:
            lines.append(f"  panels: {nd['panel'].astype(str).value_counts().head(12).to_dict()}")
        if "Cumulative_QC" in nd:
            lines.append(f"  QC: {nd['Cumulative_QC'].astype(str).value_counts().to_dict()}")
    except Exception as e:
        lines.append(f"  (not an Olink long table: {type(e).__name__}: {e})")
    return "\n".join(lines)

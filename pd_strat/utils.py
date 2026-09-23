"""
Shared utility functions: metrics, ID helpers, manifest I/O, summary tracking.

Every module imports from here rather than reimplementing these primitives.
"""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

from .config import (
    CFG, TAB, MANIFEST_PATH, META_PATH, SEED, Y_LO, Y_HI,
    PROT_COMPLETENESS_THRESHOLD, PROJECT_ROOT,
)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Run-summary helper                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

from .config import LOCKED

summary: Dict[str, Any] = {
    "locked_plan": {k: v for k, v in LOCKED.items()
                    if not isinstance(v, range)},
    "version": "2.1",
}


def summary_update(extra: dict):
    summary.update(extra)
    with open(TAB / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Sample-flow accountant                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

_FLOW: Dict[str, Dict[str, int]] = {"TRAIN": {}, "TEST": {}}


def flow_record(step: str, n_train: int, n_test: int):
    _FLOW["TRAIN"][step] = n_train
    _FLOW["TEST"][step]  = n_test


def save_flow_table() -> pd.DataFrame:
    rows = [{"step": s,
             "TRAIN_n": _FLOW["TRAIN"].get(s, ""),
             "TEST_n":  _FLOW["TEST"].get(s, "")}
            for s in _FLOW["TRAIN"]]
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "sample_flow_table.csv", index=False)
    print("\n" + "=" * 60)
    print("SAMPLE FLOW TABLE")
    print("=" * 60)
    print(df.to_string(index=False))
    return df


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  ID / index helpers                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _to_str_index(idx) -> pd.Index:
    if isinstance(idx, pd.MultiIndex):
        idx = idx.to_flat_index()
    return pd.Index([
        "" if (isinstance(x, float) and np.isnan(x)) or x is None
        else str(x) for x in idx
    ])


_ID_CANDIDATES = [
    "participant_id", "patno", "patient_id", "subject_id", "subject",
    "id", "record_id", "ppid", "pdid", "pid", "participant",
]


def choose_id(df: pd.DataFrame) -> str:
    """Pick the best participant-ID column from a clinical DataFrame."""
    present = [c for c in _ID_CANDIDATES if c in df.columns]
    for c in present:
        s = df[c]
        nunique = s.dropna().astype(str).nunique()
        n = len(s.dropna())
        if n >= 2 and nunique <= max(0.9 * n, n - 1):
            return c
    if present:
        return present[0]
    if "sample_id" in df.columns:
        return "sample_id"
    df["_tmp_id"] = np.arange(len(df))
    return "_tmp_id"


def map_participant_id(s: str) -> str:
    """Map a raw index value to a canonical participant-level ID."""
    pat = ((CFG.get("data_patterns") or {}).get("participant_pattern")
           or "split:-:0:2")
    s = str(s)
    if pat.startswith("split:"):
        try:
            _, sep, lo, hi = pat.split(":")
            lo, hi = int(lo), int(hi)
            bits = s.split(sep)
            lo = max(0, min(lo, len(bits)))
            hi = max(lo + 1, min(hi, len(bits)))
            return sep.join(bits[lo:hi])
        except Exception:
            return s
    return s


def extract_patno(idx_values) -> np.ndarray:
    """Extract participant-level ID from row index (strips visit suffixes)."""
    return np.array([map_participant_id(str(x)) for x in idx_values])


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Manifest / metadata I/O                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def short_sha(cols) -> str:
    if cols is None or len(cols) == 0:
        return "NA"
    return hashlib.sha1("\n".join(map(str, cols)).encode()).hexdigest()[:12]


_GENE_CACHE = PROJECT_ROOT / "docs" / "uniprot_gene_map.csv"
_CURATED = PROJECT_ROOT / "docs" / "literature_proteins.yaml"


def _base_accession(key: str) -> str:
    """'CSF:P06756-2' -> 'P06756' (tissue prefix and isoform suffix removed)."""
    k = str(key).split(":", 1)[-1].strip()
    return k.split("-", 1)[0].split(",", 1)[0]


def _read_gene_sources() -> Dict[str, str]:
    """Accession -> gene from every local source (Olink export, cache, curated list)."""
    m: Dict[str, str] = {}
    if _CURATED.exists():
        try:
            import yaml
            data = yaml.safe_load(open(_CURATED, encoding="utf-8")) or {}
            prots = data.get("proteins", {}) if isinstance(data, dict) else {}
            items = prots.items() if isinstance(prots, dict) else \
                [(e.get("uniprot"), e) for e in prots if isinstance(e, dict)]
            for u, entry in items:
                g = (entry or {}).get("gene") if isinstance(entry, dict) else None
                if u and g:
                    m[str(u)] = str(g)
        except Exception:
            pass
    if _GENE_CACHE.exists():
        df = pd.read_csv(_GENE_CACHE, dtype=str).fillna("")
        m.update({u: g for u, g in zip(df["uniprot"], df["gene"]) if g})
    p = TAB / "protein_annotation.csv"
    if p.exists():
        df = pd.read_csv(p, dtype=str).fillna("")
        m.update({u: g for u, g in zip(df["uniprot"], df["gene"]) if g})
    return m


def load_protein_annotation() -> Dict[str, str]:
    """UniProt accession -> gene symbol (Olink export, cached UniProt lookup, curated list).

    Keys with a tissue prefix ('PLA:', 'CSF:') resolve through the bare accession.
    """
    m = _read_gene_sources()
    if not m:
        return {}

    class _Annot(dict):
        def get(self, key, default=""):
            if key in self:
                return dict.get(self, key)
            base = _base_accession(key)
            if base in self:
                g = dict.get(self, base)
                return f"{str(key).split(':', 1)[0]}:{g}" if ":" in str(key) else g
            return default
    return _Annot(m)


def _uniprot_fetch(accessions: List[str], timeout: float = 30.0) -> Dict[str, str]:
    """Batch query rest.uniprot.org for primary gene symbols."""
    import urllib.request
    import urllib.parse
    out: Dict[str, str] = {}
    for i in range(0, len(accessions), 100):
        chunk = accessions[i:i + 100]
        q = " OR ".join(f"accession:{a}" for a in chunk)
        url = ("https://rest.uniprot.org/uniprotkb/search?"
               + urllib.parse.urlencode({"query": q, "fields": "accession,gene_primary",
                                         "format": "tsv", "size": 500}))
        req = urllib.request.Request(url, headers={"User-Agent": "pd_strat/2.1 (gene-symbol lookup)"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            txt = resp.read().decode("utf-8", errors="replace")
        for line in txt.splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) >= 2 and parts[0] and parts[1]:
                out[parts[0].strip()] = parts[1].strip().split(";")[0].strip()
    return out


def ensure_gene_annotation(prot_cols: List[str]) -> Dict[str, str]:
    """Make sure every modelled protein has a gene symbol where one can be had.

    Olink exports from AMP-PD carry only UniProt accessions.  Accessions not
    annotated by the export or the local cache are looked up on UniProt (when
    ``uniprot_lookup`` is on and the network allows) and cached in
    docs/uniprot_gene_map.csv so later runs and --report_only stay offline.
    """
    from .config import UNIPROT_LOOKUP
    have = _read_gene_sources()
    bases = sorted({_base_accession(c) for c in prot_cols})
    missing = [b for b in bases if b and b not in have]
    print(f"\n[Annotation] {len(bases)} accessions; {len(bases) - len(missing)} with gene symbol")
    fetched: Dict[str, str] = {}
    if missing and UNIPROT_LOOKUP:
        try:
            fetched = _uniprot_fetch(missing)
            print(f"  [Annotation] UniProt lookup: {len(fetched)}/{len(missing)} resolved")
        except Exception as e:  # offline / proxy / rate limit -> accessions only
            print(f"  [Annotation] UniProt lookup unavailable ({type(e).__name__}: {e}); "
                  f"tables will show accessions for {len(missing)} proteins. "
                  f"Set uniprot_lookup: false to silence.")
    elif missing:
        print(f"  [Annotation] uniprot_lookup off; {len(missing)} proteins without symbol")
    if fetched:
        rows = [{"uniprot": u, "gene": g, "source": "uniprot_rest"} for u, g in fetched.items()]
        if _GENE_CACHE.exists():
            old = pd.read_csv(_GENE_CACHE, dtype=str).fillna("")
            rows = old.to_dict("records") + [r for r in rows if r["uniprot"] not in set(old["uniprot"])]
        _GENE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).drop_duplicates("uniprot").sort_values("uniprot").to_csv(_GENE_CACHE, index=False)
        have.update(fetched)
    # results-local table used by the report and figures
    def _sym(c: str) -> str:
        g = have.get(_base_accession(c), "")
        return f"{c.split(':', 1)[0]}:{g}" if (g and ":" in c) else g
    rows = [{"uniprot": c, "gene": _sym(c)} for c in prot_cols]
    p = TAB / "protein_annotation.csv"
    if p.exists():
        old = pd.read_csv(p, dtype=str).fillna("")
        extra = old[~old["uniprot"].isin({r["uniprot"] for r in rows})]
        rows += extra[["uniprot", "gene"]].to_dict("records")
    pd.DataFrame(rows).drop_duplicates("uniprot").to_csv(p, index=False)
    return load_protein_annotation()


def load_json(p: Path) -> dict:
    if p.exists():
        try:
            return json.load(open(p))
        except Exception:
            return {}
    return {}


def save_manifest(prot_cols: List[str]):
    m = load_json(MANIFEST_PATH)
    m["prot_cols"]  = list(map(str, prot_cols))
    m["prot_sha12"] = short_sha(m["prot_cols"])
    json.dump(m, open(MANIFEST_PATH, "w"), indent=2)


def apply_manifest(Z: pd.DataFrame, key: str) -> pd.DataFrame:
    man = load_json(MANIFEST_PATH)
    want = man.get(f"{key}_cols")
    if not want:
        return Z
    inter = [c for c in want if c in set(Z.columns.astype(str))]
    return Z.loc[:, inter]


def stable_top_features(Z: pd.DataFrame, target_n: int, name: str,
                        train_mask: Optional[np.ndarray] = None
                        ) -> pd.DataFrame:
    """Deterministic top-N feature selection by missingness then variance."""
    if Z.empty or target_n <= 0 or Z.shape[1] <= target_n:
        return Z
    X = Z.values if train_mask is None else Z.values[train_mask]
    mrate = 1.0 - np.mean(pd.isna(X), axis=0)
    Xn = np.nan_to_num(X, nan=0.0)
    var = np.var(Xn, axis=0)
    keep = var > 0
    if keep.sum() == 0:
        return Z
    cols_k = np.array(Z.columns)[keep]
    m_k, v_k = mrate[keep], var[keep]
    order = np.lexsort((cols_k.astype(str), -v_k, -m_k))
    chosen = cols_k[order][:target_n].tolist()
    Z2 = Z.loc[:, chosen]
    print(f"  [Freeze/{name}] {Z.shape[1]} -> {Z2.shape[1]} features")
    return Z2


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Metric functions                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def spearman_np(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    return float(spearmanr(a[m], b[m]).statistic)


def pearson_np(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    return float(pearsonr(a[m], b[m]).statistic)


def full_metrics(pred, true, label=""):
    """Compute full metric set: Spearman, Pearson, MAE, RMSE, R2."""
    m = np.isfinite(pred) & np.isfinite(true)
    n = int(m.sum())
    if n < 3:
        return {"n": n, "spearman": np.nan, "pearson": np.nan,
                "mae": np.nan, "rmse": np.nan, "r2": np.nan}
    p, t = pred[m], true[m]
    mae  = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t)**2)))
    ss_res = np.sum((t - p)**2)
    ss_tot = np.sum((t - t.mean())**2)
    r2 = float(1 - ss_res / max(ss_tot, 1e-12))
    return {
        "n": n,
        "spearman": spearman_np(pred, true),
        "pearson":  pearson_np(pred, true),
        "mae": mae, "rmse": rmse, "r2": r2,
    }


def bootstrap_ci(positions: np.ndarray, predictions: np.ndarray,
                  y_vals: np.ndarray, clin_index: pd.Index,
                  n_boot: int = 2000,
                  ci: float = 0.95, seed: int = SEED
                  ) -> Dict[str, Dict[str, float]]:
    """Participant-level bootstrap 95% CIs for all metrics."""
    pids = np.array([map_participant_id(str(x))
                     for x in clin_index[positions]])
    unique_pids = np.unique(pids)
    rng = np.random.default_rng(seed)
    alpha = (1 - ci) / 2

    boot_metrics: Dict[str, List[float]] = {
        "spearman": [], "pearson": [], "mae": [], "rmse": [], "r2": []}

    for _ in range(n_boot):
        bs_pids = rng.choice(unique_pids, size=len(unique_pids), replace=True)
        bs_idx = []
        for pid in bs_pids:
            bs_idx.extend(np.where(pids == pid)[0])
        bs_idx = np.array(bs_idx)
        if len(bs_idx) < 5:
            continue

        p_bs = predictions[bs_idx]
        t_bs = y_vals[bs_idx]
        m_bs = np.isfinite(p_bs) & np.isfinite(t_bs)
        if m_bs.sum() < 5:
            continue

        p_, t_ = p_bs[m_bs], t_bs[m_bs]
        boot_metrics["spearman"].append(float(spearmanr(p_, t_).statistic))
        boot_metrics["pearson"].append(float(pearsonr(p_, t_).statistic))
        boot_metrics["mae"].append(float(np.mean(np.abs(p_ - t_))))
        boot_metrics["rmse"].append(float(np.sqrt(np.mean((p_ - t_)**2))))
        ss_res = np.sum((t_ - p_)**2)
        ss_tot = np.sum((t_ - t_.mean())**2)
        boot_metrics["r2"].append(float(1 - ss_res / max(ss_tot, 1e-12)))

    result: Dict[str, Dict[str, float]] = {}
    point = full_metrics(predictions, y_vals)
    for metric in boot_metrics:
        vals = np.array(boot_metrics[metric])
        if len(vals) < 100:
            result[metric] = {"point": point.get(metric, np.nan),
                              "lo": np.nan, "hi": np.nan}
        else:
            result[metric] = {
                "point": point.get(metric, np.nan),
                "lo": float(np.percentile(vals, 100 * alpha)),
                "hi": float(np.percentile(vals, 100 * (1 - alpha))),
            }
    return result


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Visit-code / demographic helpers                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def infer_visit_code(df: pd.DataFrame) -> pd.Series:
    vc = pd.Series(pd.NA, index=df.index, dtype="object")
    if "visit_name" in df.columns:
        v = df["visit_name"].astype(str).str.upper()
        tok = v.str.extract(r"\b(M\d{1,3})\b", expand=False)
        fb  = v.str.extract(r"\bMONTH\s*(\d{1,3})\b", expand=False)
        vc  = tok.fillna(fb.map(lambda x: f"M{x}" if pd.notna(x) else pd.NA))
    if "sample_id" in df.columns:
        s = df["sample_id"].astype(str)
        m_blm   = s.str.extract(r"\bBLM(\d{1,3})", flags=re.I, expand=False)
        m_plain = s.str.extract(
            r"(?<![A-Z0-9])M(\d{1,3})(?![A-Z0-9])", flags=re.I, expand=False)
        vc = vc.fillna(m_blm.map(lambda x: f"M{x}" if pd.notna(x) else pd.NA))
        vc = vc.fillna(m_plain.map(lambda x: f"M{x}" if pd.notna(x) else pd.NA))
    if "months" in df.columns:
        mn = pd.to_numeric(df["months"], errors="coerce")
        vc = vc.fillna(mn.round().astype("Int64").map(
            lambda z: f"M{int(z)}" if pd.notna(z) else pd.NA))
    return vc.astype("string")


def resolve_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def eta2(labels, values):
    """Effect-size: fraction of variance explained by cluster labels."""
    m = np.isfinite(values)
    if m.sum() < 10:
        return np.nan
    l, v = labels[m], values[m]
    grand = v.mean()
    ss_t = np.sum((v - grand)**2)
    if ss_t < 1e-12:
        return 0.0
    ss_b = sum(np.sum(l == k) * (v[l == k].mean() - grand)**2
               for k in np.unique(l))
    return float(ss_b / ss_t)

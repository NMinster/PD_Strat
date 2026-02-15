"""
§1-2c — Load clinical data, proteomics (Olink NPX), and RNA expression.

Responsible for:
  - Reading clinical_unified.csv and inferring ID/visit/cohort columns.
  - Loading and HC-anchored z-scoring of proteomics panels.
  - Loading and z-scoring RNA gene-expression data.
  - Completeness filter audit.
"""

from __future__ import annotations

import os
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

from .config import (
    CFG, TAB, TRAIN_PREFIX, TEST_PREFIX,
    PROTEOMICS_PANELS, PROT_QC_FILTER, PROT_COMPLETENESS_THRESHOLD,
    HC_MODE, N_SVD, CFG_PROT_TARGET_N, SEED,
    META_PATH, MANIFEST_PATH,
    RNA_COMPLETENESS_THRESHOLD, RNA_TARGET_N_GENES, RNA_SVD_NC,
    RNA_EXCLUDE_BATCHES,
)
from .utils import (
    _to_str_index, choose_id, map_participant_id, infer_visit_code,
    resolve_column, short_sha, load_json, save_manifest,
    apply_manifest, stable_top_features, flow_record, summary_update,
)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  §1  LOAD CLINICAL DATA & TARGETS                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _ensure_updrs_total(df: pd.DataFrame):
    """Create or validate the updrs_total column."""
    if ("updrs_total" in df.columns
            and pd.to_numeric(df["updrs_total"], errors="coerce").notna().sum() > 20):
        df["updrs_total"] = pd.to_numeric(df["updrs_total"], errors="coerce")
        return
    lc = {x.lower(): x for x in df.columns}
    for k in ("updrs_total", "mds_updrs_total", "mds_updrs_total_score",
              "mds_updrs_total__sum", "total_updrs"):
        if k in lc:
            df["updrs_total"] = pd.to_numeric(df[lc[k]], errors="coerce")
            return
    parts = []
    for aliases in (
        ("mds_updrs_part_i_total",   "updrs_part_i_total"),
        ("mds_updrs_part_ii_total",  "updrs_part_ii_total"),
        ("mds_updrs_part_iii_total", "updrs_part_iii_total"),
        ("mds_updrs_part_iv_total",  "updrs_part_iv_total"),
    ):
        for a in aliases:
            if a.lower() in lc:
                parts.append(lc[a.lower()])
                break
    if parts:
        df["updrs_total"] = pd.to_numeric(
            df[parts].sum(axis=1, min_count=1), errors="coerce")
    else:
        df["updrs_total"] = np.nan


def _ensure_upsit(df: pd.DataFrame):
    for k in ("upsit_total", "UPSIT_total", "UPSITTOTAL",
              "upsit_score", "upsit"):
        if k in df.columns:
            s = pd.to_numeric(df[k], errors="coerce")
            if np.isfinite(s).sum() > 0:
                df["upsit_total"] = s
                return
    df["upsit_total"] = np.nan


_SEX_CANDIDATES  = ["sex", "Sex", "SEX", "gender", "Gender"]
_AGE_CANDIDATES  = ["age_at_baseline", "age", "Age", "AGE",
                     "age_at_enrollment", "enroll_age"]
_SITE_CANDIDATES = ["site", "Site", "SITE", "center", "Centre",
                     "enrolling_site", "study_site"]


def load_clinical() -> Tuple[pd.DataFrame, str, str, str, str]:
    """Load and prepare clinical data.

    Returns
    -------
    clin : pd.DataFrame   — indexed by participant ID
    id_key, SEX_COL, AGE_COL, SITE_COL : str or None
    """
    clin_path = TAB / "clinical_unified.csv"
    if not clin_path.exists():
        raise FileNotFoundError(
            "Missing results/tables/clinical_unified.csv -- run assembly first.")
    clin = pd.read_csv(clin_path)

    # ── ID inference ────────────────────────────────────────────────────
    id_key = choose_id(clin)
    clin[id_key] = _to_str_index(clin[id_key])
    print(f"[Infer] id_key='{id_key}'  "
          f"(unique={clin[id_key].nunique()}, rows={len(clin)})")
    clin = clin.set_index(id_key, drop=True)
    clin.index = _to_str_index(clin.index)

    # ── UPDRS total ─────────────────────────────────────────────────────
    _ensure_updrs_total(clin)

    # ── Visit code ──────────────────────────────────────────────────────
    clin["visit_code"] = infer_visit_code(clin)

    # ── Cohort flags ────────────────────────────────────────────────────
    clin["cohort"] = (clin.get("cohort",
                               pd.Series(index=clin.index, dtype=object))
                      .astype(str).str.upper())

    # ── Case / control ──────────────────────────────────────────────────
    if "case_control" not in clin.columns:
        is_test = clin["cohort"].eq("TEST")
        clin["case_control"] = np.where(is_test, "CASE", "UNKNOWN")

    # ── UPSIT ───────────────────────────────────────────────────────────
    _ensure_upsit(clin)

    # ── Demographics ────────────────────────────────────────────────────
    SEX_COL  = resolve_column(clin, _SEX_CANDIDATES)
    AGE_COL  = resolve_column(clin, _AGE_CANDIDATES)
    SITE_COL = resolve_column(clin, _SITE_CANDIDATES)
    print(f"[Demographics] sex='{SEX_COL}', age='{AGE_COL}', site='{SITE_COL}'")

    clin["_sex"]  = clin[SEX_COL].astype(str) if SEX_COL else np.nan
    clin["_age"]  = pd.to_numeric(clin[AGE_COL], errors="coerce") if AGE_COL else np.nan
    clin["_site"] = clin[SITE_COL].astype(str) if SITE_COL else np.nan

    return clin, id_key, SEX_COL, AGE_COL, SITE_COL


def build_cohort_masks(clin: pd.DataFrame):
    """Derive train/test/PD/HC boolean masks from clinical data.

    Returns dict with keys: is_train, is_test, is_control, is_pd_flag,
    is_pd_train, is_pd_test.
    """
    is_train = clin["cohort"].eq("TRAIN")
    is_test  = clin["cohort"].eq("TEST")

    cc = clin["case_control"].astype(str).str.upper()
    is_control = cc.isin(["CONTROL", "CNTL", "HC", "HEALTHY"])
    is_test_prefix = np.fromiter(
        (str(x).upper().startswith(TEST_PREFIX) for x in clin.index),
        dtype=bool, count=len(clin))
    is_pd_flag = (cc.isin(["CASE", "PD", "PARKINSONS", "PATIENT", "DISEASE"])
                  | is_test_prefix) & ~is_control
    is_pd_train = is_pd_flag & is_train
    is_pd_test  = is_pd_flag & is_test

    flow_record("01_raw_clinical", int(is_train.sum()), int(is_test.sum()))

    return {
        "is_train": is_train, "is_test": is_test,
        "is_control": is_control, "is_pd_flag": is_pd_flag,
        "is_pd_train": is_pd_train, "is_pd_test": is_pd_test,
    }


def demographics_for(clin: pd.DataFrame, positions: np.ndarray) -> pd.DataFrame:
    sub = clin.iloc[positions]
    return pd.DataFrame({
        "participant_id": sub.index.values,
        "sex":    sub["_sex"].values,
        "age":    sub["_age"].values,
        "site":   sub["_site"].values,
        "cohort": sub["cohort"].values,
    })


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  §2  LOAD PROTEOMICS                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _normlower(s):
    return pd.Series(s, dtype="object").astype(str).str.strip().str.lower()


def _is_hc_rows(df_case: pd.DataFrame):
    diag_bl = _normlower(df_case.get("diagnosis_at_baseline"))
    c_bl    = _normlower(df_case.get("case_control_other_at_baseline"))
    c_lt    = _normlower(df_case.get("case_control_other_latest"))
    return ((diag_bl.str.contains("no pd nor other", na=False)
             | c_bl.eq("control") | c_lt.eq("control"))
            .fillna(False))


def _load_case_control_table() -> Optional[pd.DataFrame]:
    cc_path = CFG.get("case_control_path", "")
    if not cc_path or not os.path.exists(cc_path):
        return None
    df = pd.read_csv(cc_path, sep=None, engine="python", encoding="utf-8-sig")
    if "participant_id" not in df.columns:
        cand = next((c for c in df.columns
                     if c.strip().lower() in {"participant_id", "participant",
                                              "patno"}), None)
        if cand is None:
            raise ValueError("case_control file needs participant_id column")
        df = df.rename(columns={cand: "participant_id"})
    df["participant_id_mapped"] = (df["participant_id"].astype(str)
                                   .map(map_participant_id))
    return df.set_index("participant_id_mapped")


CASE_DF = _load_case_control_table()


def _select_train_hc_ids(raw_index: pd.Index, is_train: pd.Series,
                          min_n: int = 30) -> pd.Index:
    if CASE_DF is None:
        return pd.Index([])
    hc_ids_all = CASE_DF.index[_is_hc_rows(CASE_DF)]
    clin_index = is_train.index
    hc_ids = (pd.Index(hc_ids_all)
              .intersection(clin_index[is_train])
              .intersection(raw_index))
    return hc_ids if len(hc_ids) >= min_n else pd.Index([])


def _load_proteomics_panels(panels_dict: Dict[str, str],
                            qc_filter: str = "PASS"
                            ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Load panels, return (wide_df, panel_protein_map)."""
    panel_dfs = []
    panel_map: Dict[str, List[str]] = {}

    for name, path in panels_dict.items():
        if not os.path.exists(path):
            print(f"  [WARN] Panel '{name}' not found: {path}")
            continue
        df = pd.read_csv(path, low_memory=False)
        df["panel"] = name
        if "Cumulative_QC" in df.columns and qc_filter:
            n0 = len(df)
            df = df[df["Cumulative_QC"].astype(str).str.upper()
                    == qc_filter.upper()]
            print(f"  {name}: {n0:,}->{len(df):,} rows (QC), "
                  f"{df['UniProt'].nunique()} proteins")
        else:
            print(f"  {name}: {df.shape[0]:,} rows, "
                  f"{df['UniProt'].nunique()} proteins")
        panel_map[name] = sorted(df["UniProt"].dropna().unique().tolist())
        panel_dfs.append(df)

    if not panel_dfs:
        return pd.DataFrame(), {}

    df_all = pd.concat(panel_dfs, ignore_index=True)
    print(f"\n  Total: {df_all.shape[0]:,} rows, "
          f"{df_all['UniProt'].nunique()} proteins, "
          f"{df_all['participant_id'].nunique()} subjects")
    wide = (df_all.pivot_table(index="participant_id", columns="UniProt",
                               values="NPX", aggfunc="mean"))
    wide.index = [map_participant_id(str(i)) for i in wide.index]
    if wide.index.has_duplicates:
        wide = wide.groupby(wide.index).mean()
    print(f"  Wide: {wide.shape}")

    for pname in list(panel_map.keys()):
        panel_map[pname] = [c for c in panel_map[pname] if c in wide.columns]
        print(f"  Panel '{pname}': {len(panel_map[pname])} proteins in wide")

    return wide.astype(np.float32), panel_map


def load_proteomics(clin: pd.DataFrame,
                    is_train: pd.Series,
                    name_csv: str = "z_prot.csv",
                    ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Load (or build) z-scored proteomics.

    Returns (z_prot DataFrame aligned to clin.index, panel_protein_map).
    """
    csv_path = TAB / name_csv
    meta = load_json(META_PATH).get(name_csv, {})
    panel_map_path = TAB / "panel_protein_map.json"
    panel_protein_map: Dict[str, List[str]] = {}

    # Try cached
    if csv_path.exists() and meta.get("hc_mode") == HC_MODE:
        df = pd.read_csv(csv_path, index_col=0)
        def _pid_like(x):
            return str(x).upper().startswith(("PP-", "PD-"))
        if (sum(_pid_like(c) for c in df.columns) >
                0.6 * len(df.columns)):
            df = df.T
        df.index = [map_participant_id(i) for i in df.index.astype(str)]
        if df.index.has_duplicates:
            df = df[~df.index.duplicated(keep="first")]
        df = df.reindex(clin.index)
        df = apply_manifest(df, "prot")
        if panel_map_path.exists():
            pmap = json.load(open(panel_map_path))
            for pname in list(pmap.keys()):
                pmap[pname] = [c for c in pmap[pname]
                               if c in set(df.columns.astype(str))]
            panel_protein_map = pmap
        else:
            z_cols_set = set(df.columns.astype(str))
            for pname, ppath in PROTEOMICS_PANELS.items():
                if os.path.exists(ppath):
                    try:
                        raw_p = pd.read_csv(ppath, low_memory=False,
                                            usecols=["UniProt"], nrows=500000)
                        prots = sorted(raw_p["UniProt"].dropna().unique().tolist())
                        prots = [c for c in prots if c in z_cols_set]
                        if prots:
                            panel_protein_map[pname] = prots
                            print(f"    [Reconstruct] panel '{pname}': "
                                  f"{len(prots)} proteins")
                    except Exception as e:
                        print(f"    [WARN] Could not read panel '{pname}': {e}")
            if panel_protein_map:
                json.dump(panel_protein_map, open(panel_map_path, "w"), indent=2)
        print(f"  [Pin] {name_csv} shape={df.shape}, "
              f"sha={short_sha(df.columns)}, "
              f"panels={list(panel_protein_map.keys())}")
        return df, panel_protein_map

    # Build from panels
    raw, panel_map = _load_proteomics_panels(PROTEOMICS_PANELS, PROT_QC_FILTER)
    if raw.empty:
        print("  [WARN] No proteomics data -- returning empty DataFrame")
        return pd.DataFrame(index=clin.index), {}

    raw = raw.reindex(clin.index)
    print(f"  [PROT/raw] samples={raw.notna().any(axis=1).sum()}, "
          f"feats={raw.shape[1]}")

    # HC-anchored z-scoring
    hc_ids = _select_train_hc_ids(raw.dropna(how="all").index, is_train,
                                   min_n=30)
    if len(hc_ids) >= 30:
        mu = raw.loc[hc_ids].mean(axis=0)
        sd = raw.loc[hc_ids].std(axis=0).replace(0, np.nan).fillna(1.0)
        Z = (raw - mu) / sd
        print(f"  [HC] Global TRAIN anchoring (n_hc={len(hc_ids)})")
    else:
        tr_ids = clin.index[is_train]
        tr_raw = raw.loc[raw.index.intersection(tr_ids)]
        mu = tr_raw.median(axis=0)
        mad = (tr_raw - mu).abs().median(axis=0)
        sd = (mad * 1.4826).replace(0, np.nan).fillna(1.0)
        Z = (raw - mu) / sd
        print(f"  [HC] Endpoint-blind fallback: robust median/MAD on "
              f"TRAIN (n={len(tr_raw)})")

    Z = Z.clip(-10, 10).reindex(clin.index).astype(np.float32)
    Z = stable_top_features(Z, CFG_PROT_TARGET_N, "PROT",
                            train_mask=is_train.values)
    man = load_json(MANIFEST_PATH)
    if "prot_cols" in man:
        Z = apply_manifest(Z, "prot")
    else:
        save_manifest(list(Z.columns))

    z_cols_set = set(Z.columns.astype(str))
    for pname in list(panel_map.keys()):
        panel_map[pname] = [c for c in panel_map[pname] if c in z_cols_set]
    panel_protein_map = panel_map

    Z.to_csv(csv_path)
    json.dump(panel_map, open(panel_map_path, "w"), indent=2)
    sha = short_sha(Z.columns)
    meta_all = load_json(META_PATH)
    meta_all[name_csv] = dict(hc_mode=HC_MODE, shape=list(Z.shape), sha12=sha)
    json.dump(meta_all, open(META_PATH, "w"), indent=2)
    print(f"  [Pin] wrote {name_csv}: shape={Z.shape}, sha={sha}")
    return Z, panel_protein_map


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  §2b  COMPLETENESS FILTER AUDIT                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def completeness_audit(z_prot: pd.DataFrame,
                       is_train: pd.Series, is_test: pd.Series,
                       panel_protein_map: Dict[str, List[str]]):
    """Print and save per-sample / per-panel completeness diagnostics."""
    print(f"\n{'=' * 60}")
    print("COMPLETENESS FILTER AUDIT")
    print(f"{'=' * 60}")

    raw_miss = z_prot.isna().mean(axis=1)
    raw_obs  = 1.0 - raw_miss
    panel_presence: Dict[str, pd.Series] = {}
    for pname, pcols in panel_protein_map.items():
        if pcols:
            panel_presence[pname] = z_prot[pcols].notna().mean(axis=1)

    print(f"  Total samples: {len(z_prot)}")
    print(f"  Total proteins: {z_prot.shape[1]}")
    print(f"  Overall missingness: {raw_miss.mean()*100:.1f}%")
    print(f"\n  Per-sample completeness distribution:")
    for q in [0, 10, 25, 50, 75, 90, 100]:
        val = np.nanpercentile(raw_obs.values, q)
        print(f"    P{q:3d}: {val*100:.1f}%")

    print(f"\n  Samples passing completeness thresholds:")
    for thresh in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        n_pass_train = int((raw_obs[is_train] >= thresh).sum())
        n_pass_test  = int((raw_obs[is_test] >= thresh).sum())
        marker = " <<<" if thresh == PROT_COMPLETENESS_THRESHOLD else ""
        print(f"    >={thresh*100:.0f}%: TRAIN={n_pass_train}, "
              f"TEST={n_pass_test}{marker}")

    if panel_presence:
        print(f"\n  Per-panel presence:")
        for pname, pseries in panel_presence.items():
            has_panel = (pseries > 0).sum()
            full_panel = (pseries >= 0.9).sum()
            print(f"    {pname}: {has_panel} any data, {full_panel} >=90%")

    audit_df = pd.DataFrame({
        "participant_id": z_prot.index,
        "cohort": is_train.map({True: "TRAIN", False: ""}).values,
        "overall_completeness": raw_obs.values,
    })
    for pname, pseries in panel_presence.items():
        audit_df[f"panel_{pname}_completeness"] = pseries.values
    audit_df.to_csv(TAB / "completeness_audit.csv", index=False)

    n_with = z_prot.notna().any(axis=1).sum()
    n_tr   = (is_train & z_prot.notna().any(axis=1)).sum()
    n_te   = (is_test  & z_prot.notna().any(axis=1)).sum()
    miss   = z_prot.isna().mean().mean()
    print(f"\n  Samples with any data: {n_with} (TRAIN={n_tr}, TEST={n_te})")
    print(f"  Proteins: {z_prot.shape[1]}")
    print(f"  Overall missingness: {miss*100:.1f}%")
    flow_record("03_after_qc", int(n_tr), int(n_te))


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  §2c  RNA LOADING                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def load_rna(clin: pd.DataFrame,
             is_train: pd.Series) -> Tuple[pd.DataFrame, bool]:
    """Load optional RNA modality. Returns (z_rna, HAS_RNA)."""
    print(f"\n{'=' * 60}")
    print("RNA LOADING (optional modality)")
    print(f"{'=' * 60}")

    rna_path = TAB / "z_rna.csv"
    rna_raw_path = Path(CFG.get("rna_path",
        r"S:/AMP-PD/releases_2023_v4release_1027_rnaseq_gene_expression.csv"))

    z_rna = pd.DataFrame(index=clin.index)

    # Try cached
    if rna_path.exists():
        df = pd.read_csv(rna_path, index_col=0)
        def _pid(x):
            return str(x).upper().startswith(("PP-", "PD-"))
        if sum(_pid(c) for c in df.columns) > 0.6 * len(df.columns):
            df = df.T
        df.index = [map_participant_id(i) for i in df.index.astype(str)]
        if df.index.has_duplicates:
            df = df[~df.index.duplicated(keep="first")]
        df = df.reindex(clin.index)
        print(f"  [Pin] z_rna.csv: shape={df.shape}, "
              f"non-null={df.notna().any(axis=1).sum()}")
        z_rna = df
    elif rna_raw_path.exists():
        print(f"  Loading raw RNA from: {rna_raw_path}")
        try:
            raw = pd.read_csv(rna_raw_path, low_memory=False)
            if "participant_id" in raw.columns:
                id_col = "participant_id"
            elif "PATNO" in raw.columns:
                id_col = "PATNO"
            else:
                id_col = raw.columns[0]
            raw.index = [map_participant_id(str(x)) for x in raw[id_col].values]
            raw = raw.drop(columns=[id_col], errors="ignore")
            raw = raw.select_dtypes(include=[np.number])
            if raw.index.has_duplicates:
                raw = raw[~raw.index.duplicated(keep="first")]
            raw = raw.reindex(clin.index)
            print(f"  [RNA/raw] samples={raw.notna().any(axis=1).sum()}, "
                  f"genes={raw.shape[1]}")

            hc_ids = _select_train_hc_ids(raw.dropna(how="all").index,
                                           is_train, min_n=30)
            if len(hc_ids) >= 30:
                mu = raw.loc[hc_ids].mean(axis=0)
                sd = raw.loc[hc_ids].std(axis=0).replace(0, np.nan).fillna(1.0)
                Z = (raw - mu) / sd
                print(f"  [RNA/HC] Global TRAIN anchoring (n_hc={len(hc_ids)})")
            else:
                tr_ids = clin.index[is_train]
                tr_raw = raw.loc[raw.index.intersection(tr_ids)]
                mu = tr_raw.median(axis=0)
                mad = (tr_raw - mu).abs().median(axis=0)
                sd = (mad * 1.4826).replace(0, np.nan).fillna(1.0)
                Z = (raw - mu) / sd
                print(f"  [RNA/HC] Endpoint-blind median/MAD (n={len(tr_raw)})")

            Z = Z.clip(-10, 10).reindex(clin.index).astype(np.float32)
            Z = stable_top_features(Z, RNA_TARGET_N_GENES, "RNA",
                                    train_mask=is_train.values)
            Z.to_csv(rna_path)
            print(f"  [Pin] wrote z_rna.csv: shape={Z.shape}")
            z_rna = Z
        except Exception as e:
            print(f"  [WARN] RNA loading failed: {e}")
    else:
        print(f"  [SKIP] No RNA data found at {rna_raw_path}")

    # Batch QC for RNA
    if not z_rna.empty and RNA_EXCLUDE_BATCHES:
        def _rna_batch_prefix(pid: str) -> str:
            pid = str(pid).upper().strip()
            if pid.startswith("PP-"):
                digits = ''.join(c for c in pid[3:][:2] if c.isdigit())
                if digits:
                    return f"PP-{digits}"
            return "OTHER"
        bp = pd.Series([_rna_batch_prefix(p) for p in z_rna.index],
                       index=z_rna.index)
        excl = bp.isin(RNA_EXCLUDE_BATCHES)
        if excl.sum() > 0:
            z_rna.loc[excl, :] = np.nan
            print(f"  RNA batch QC: excluded {excl.sum()} samples "
                  f"from batches {RNA_EXCLUDE_BATCHES}")

    HAS_RNA = not z_rna.empty and z_rna.notna().any(axis=1).sum() >= 30
    print(f"  RNA available: {HAS_RNA} "
          f"({z_rna.notna().any(axis=1).sum()} samples with data)")
    return z_rna, HAS_RNA

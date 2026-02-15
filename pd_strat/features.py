"""
§3 + §5 — Build aligned matrices and SVD-based feature constructors.

Provides:
  - build_aligned_matrices()  — zero-fill + clip + mask matrices
  - build_panel_aware_features() — per-panel SVD + missingness indicators
  - build_monolithic_features() — single global SVD
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

from .config import N_SVD, PANEL_SVD_NC, SEED


def svd_n_components(X_train: np.ndarray, target: int = N_SVD) -> int:
    return max(1, min(target, X_train.shape[1] - 1, X_train.shape[0] - 1))


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  §3  BUILD ALIGNED MATRICES                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def build_aligned_matrices(z_prot: pd.DataFrame,
                           z_rna: pd.DataFrame,
                           clin_index: pd.Index,
                           panel_protein_map: Dict[str, List[str]],
                           HAS_RNA: bool,
                           ) -> dict:
    """Build zero-filled, clipped feature and mask matrices.

    Returns a dict with keys:
        X_prot, M_prot, d_prot, prot_cols,
        X_rna, M_rna, d_rna, rna_cols,
        PANEL_COL_INDICES, USE_PANEL_AWARE
    """
    all_ids = clin_index

    # Proteomics
    Zp = z_prot.replace([np.inf, -np.inf], np.nan).reindex(all_ids)
    X_prot = Zp.values.astype(np.float32)
    M_prot = (~pd.isna(Zp)).values.astype(np.float32)
    X_prot[np.isnan(X_prot)] = 0.0
    np.clip(X_prot, -10, 10, out=X_prot)
    d_prot = X_prot.shape[1]
    prot_cols = list(Zp.columns)
    print(f"[Aligned] d_prot={d_prot}")

    # RNA
    if HAS_RNA:
        Zr = z_rna.replace([np.inf, -np.inf], np.nan).reindex(all_ids)
        X_rna = Zr.values.astype(np.float32)
        M_rna = (~pd.isna(Zr)).values.astype(np.float32)
        X_rna[np.isnan(X_rna)] = 0.0
        np.clip(X_rna, -10, 10, out=X_rna)
        d_rna = X_rna.shape[1]
        rna_cols = list(Zr.columns)
        print(f"[Aligned] d_rna={d_rna}")
    else:
        X_rna = np.zeros((len(all_ids), 0), dtype=np.float32)
        M_rna = np.zeros((len(all_ids), 0), dtype=np.float32)
        d_rna = 0
        rna_cols = []
        print("[Aligned] d_rna=0 (no RNA data)")

    # Panel column indices
    prot_cols_set = {c: i for i, c in enumerate(prot_cols)}
    PANEL_COL_INDICES: Dict[str, List[int]] = {}
    for pname, pcols in panel_protein_map.items():
        PANEL_COL_INDICES[pname] = [prot_cols_set[c] for c in pcols
                                     if c in prot_cols_set]
        print(f"  Panel '{pname}': {len(PANEL_COL_INDICES[pname])} col indices")

    USE_PANEL_AWARE = (len(PANEL_COL_INDICES) >= 2
                       and all(len(v) > 0 for v in PANEL_COL_INDICES.values()))
    if not USE_PANEL_AWARE and PANEL_COL_INDICES:
        PANEL_COL_INDICES = {k: v for k, v in PANEL_COL_INDICES.items()
                             if len(v) > 0}
        USE_PANEL_AWARE = len(PANEL_COL_INDICES) >= 2

    tag = "ENABLED" if USE_PANEL_AWARE else "DISABLED"
    print(f"  -> Panel-aware modeling {tag} "
          f"({len(PANEL_COL_INDICES)} panels)")

    return {
        "X_prot": X_prot, "M_prot": M_prot,
        "d_prot": d_prot, "prot_cols": prot_cols,
        "X_rna": X_rna, "M_rna": M_rna,
        "d_rna": d_rna, "rna_cols": rna_cols,
        "PANEL_COL_INDICES": PANEL_COL_INDICES,
        "USE_PANEL_AWARE": USE_PANEL_AWARE,
        "prot_cols_set": prot_cols_set,
    }


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  §5  PANEL-AWARE & MONOLITHIC SVD FEATURE BUILDERS                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def build_panel_aware_features(X: np.ndarray, M: np.ndarray,
                               train_positions: np.ndarray,
                               panel_col_indices: Dict[str, List[int]],
                               n_svd_per_panel: int = PANEL_SVD_NC,
                               fit_objects: Optional[Dict] = None,
                               seed: int = SEED,
                               ) -> Tuple[np.ndarray, Dict]:
    """Build panel-aware feature matrix with per-panel SVD + missingness."""
    parts = []
    col_names = []
    fit_objs = fit_objects or {}
    is_transform_only = fit_objects is not None

    for pname, cidx in panel_col_indices.items():
        if len(cidx) == 0:
            continue
        Xp = X[:, cidx]
        Mp = M[:, cidx]
        panel_obs_frac = Mp.mean(axis=1)

        parts.append(panel_obs_frac.reshape(-1, 1))
        col_names.append(f"miss_{pname}")

        if is_transform_only and pname in fit_objs:
            sc = fit_objs[pname]["scaler"]
            svd = fit_objs[pname]["svd"]
            Xp_s = sc.transform(Xp)
            Zp = svd.transform(Xp_s)
        else:
            tr_mask = np.zeros(X.shape[0], dtype=bool)
            tr_mask[train_positions] = True
            panel_ok = panel_obs_frac >= 0.3
            fit_mask = tr_mask & panel_ok

            sc = StandardScaler(with_mean=False)
            if fit_mask.sum() >= 5:
                sc.fit(Xp[fit_mask])
            else:
                sc.fit(Xp[tr_mask] if tr_mask.sum() >= 2 else Xp)
            Xp_s = sc.transform(Xp)

            nc = svd_n_components(
                Xp_s[fit_mask] if fit_mask.sum() >= 5 else Xp_s[tr_mask],
                n_svd_per_panel)
            svd = TruncatedSVD(n_components=nc, random_state=seed)
            if fit_mask.sum() >= 5:
                svd.fit(Xp_s[fit_mask])
            else:
                svd.fit(Xp_s[tr_mask] if tr_mask.sum() >= 2 else Xp_s)
            Zp = svd.transform(Xp_s)
            fit_objs[pname] = {"scaler": sc, "svd": svd}

        parts.append(Zp)
        col_names.extend([f"{pname}_pc{i}" for i in range(Zp.shape[1])])

    global_obs = M.mean(axis=1)
    parts.append(global_obs.reshape(-1, 1))
    col_names.append("miss_global")

    Z_panel = np.hstack(parts).astype(np.float32)
    return Z_panel, fit_objs


def build_monolithic_features(X: np.ndarray, train_positions: np.ndarray,
                              n_svd: int = N_SVD,
                              fit_objects: Optional[Dict] = None,
                              ) -> Tuple[np.ndarray, Dict]:
    """Original monolithic scaler+SVD (for comparison / ablation)."""
    fit_objs = fit_objects or {}
    is_transform_only = fit_objects is not None

    if is_transform_only and "mono" in fit_objs:
        sc = fit_objs["mono"]["scaler"]
        svd = fit_objs["mono"]["svd"]
        X_s = sc.transform(X)
        Z = svd.transform(X_s)
    else:
        sc = StandardScaler(with_mean=False)
        sc.fit(X[train_positions])
        X_s = sc.transform(X)
        nc = svd_n_components(X_s[train_positions], n_svd)
        svd = TruncatedSVD(n_components=nc, random_state=SEED)
        svd.fit(X_s[train_positions])
        Z = svd.transform(X_s)
        fit_objs["mono"] = {"scaler": sc, "svd": svd}

    return Z.astype(np.float32), fit_objs

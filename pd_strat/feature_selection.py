"""
§2a — Leakage-safe, outcome-blind protein feature selection.

Three stages, all computed on TRAIN rows that carry any proteomics
("effective" training mask) and never on the outcome:

  1. MAD / observation filter  — drop proteins with MAD < ``min_mad`` or
     observed fraction < ``min_obs_frac``.
  2. Redundancy pruning        — greedy removal of near-duplicate pairs
     (|Pearson r| > ``corr_thresh``); the member with the higher MAD is kept.
  3. Cap                       — if more than ``cap`` proteins survive, keep
     the top ``cap`` by MAD.

The selection log is written to results/tables/feature_selection_log.json so
the manuscript numbers (n dropped at each stage) are reproducible.
"""

from __future__ import annotations

import json
import warnings
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from .config import TAB
from .utils import summary_update


def select_protein_features(Z: pd.DataFrame,
                            train_mask: np.ndarray,
                            min_obs_frac: float = 0.30,
                            min_mad: float = 0.01,
                            corr_thresh: float = 0.95,
                            cap: int = 1168,
                            name: str = "PROT",
                            ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return (Z restricted to selected columns, selection log)."""
    log: Dict[str, Any] = {"n_input": int(Z.shape[1])}
    if Z.empty or Z.shape[1] < 2:
        return Z, log

    X = Z.values.astype(np.float64)[np.asarray(train_mask, dtype=bool)]
    eff = np.isfinite(X).any(axis=1)          # rows that actually carry data
    X = X[eff]
    n_eff = int(X.shape[0])
    log["n_effective_train_rows"] = n_eff
    if n_eff < 10:
        print(f"  [Select/{name}] only {n_eff} effective TRAIN rows -- "
              f"selection skipped")
        return Z, log

    # ── stage 1: observation fraction + MAD ─────────────────────────────
    obs = np.mean(np.isfinite(X), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(X, axis=0)
        mad = np.nanmedian(np.abs(X - med), axis=0)
    mad = np.nan_to_num(mad, nan=0.0)
    keep1 = (obs >= min_obs_frac) & (mad >= min_mad)
    log["n_dropped_low_obs"] = int((obs < min_obs_frac).sum())
    log["n_dropped_low_mad"] = int(((obs >= min_obs_frac) & (mad < min_mad)).sum())
    log["n_after_stage1"] = int(keep1.sum())
    if keep1.sum() < 2:
        print(f"  [Select/{name}] stage 1 left <2 features -- selection skipped")
        return Z, log

    cols = np.array(Z.columns)[keep1]
    mad_k = mad[keep1]
    Xk = X[:, keep1]

    # ── stage 2: greedy |r| pruning (higher-MAD member kept) ─────────────
    col_mean = np.nanmean(Xk, axis=0)
    Xi = np.where(np.isfinite(Xk), Xk, col_mean[None, :])
    Xi = Xi - Xi.mean(axis=0)
    sd = Xi.std(axis=0)
    sd[sd == 0] = 1.0
    Xi = Xi / sd
    C = (Xi.T @ Xi) / n_eff
    np.fill_diagonal(C, 0.0)

    order = np.argsort(-mad_k, kind="stable")
    kept: list = []
    n_redundant = 0
    for j in order:
        if kept and np.any(np.abs(C[j, kept]) > corr_thresh):
            n_redundant += 1
            continue
        kept.append(int(j))
    kept_arr = np.array(kept, dtype=int)
    log["n_dropped_redundant"] = int(n_redundant)
    log["n_after_stage2"] = int(len(kept_arr))

    # ── stage 3: cap by MAD ─────────────────────────────────────────────
    if cap and len(kept_arr) > cap:
        kept_arr = kept_arr[np.argsort(-mad_k[kept_arr], kind="stable")[:cap]]
        log["n_dropped_cap"] = int(log["n_after_stage2"] - cap)
    else:
        log["n_dropped_cap"] = 0
    final_cols = list(cols[np.sort(kept_arr)])       # keep original order
    log["n_selected"] = int(len(final_cols))
    log["params"] = dict(min_obs_frac=min_obs_frac, min_mad=min_mad,
                         corr_thresh=corr_thresh, cap=cap)

    print(f"  [Select/{name}] {log['n_input']} -> "
          f"{log['n_after_stage1']} (obs>={min_obs_frac:.0%}, MAD>={min_mad}) -> "
          f"{log['n_after_stage2']} (|r|<={corr_thresh}) -> "
          f"{log['n_selected']} (cap {cap})   "
          f"[effective TRAIN rows={n_eff}]")

    json.dump(log, open(TAB / f"feature_selection_log_{name}.json", "w"), indent=2)
    summary_update({f"feature_selection_{name}": log})
    return Z.loc[:, final_cols], log

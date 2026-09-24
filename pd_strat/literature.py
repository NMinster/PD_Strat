"""
§11b — Overlap of the model's proteins with the published PD proteomic
literature.

The curated list lives in ``docs/literature_proteins.yaml`` (UniProt ->
gene, source, note, verify flag) so it can be extended without touching code.
For every curated protein we report whether it is on the assayed panel, its
full-TRAIN importance rank, bootstrap inclusion frequency, and membership in
the locked-40, stability-selected and k* sets.  Proteins flagged
``treatment_responsive`` (e.g. plasma DDC under levodopa) are listed
separately because replication of such a protein across cohorts does not
establish it as a severity marker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from .config import TAB, ROB, PROJECT_ROOT
from .utils import summary_update, load_protein_annotation

CURATED_PATH = PROJECT_ROOT / "docs" / "literature_proteins.yaml"


def _load_curated() -> List[Dict[str, Any]]:
    if not CURATED_PATH.exists():
        return []
    import yaml
    data = yaml.safe_load(open(CURATED_PATH, encoding="utf-8")) or {}
    out = []
    for acc, v in (data.get("proteins") or {}).items():
        v = dict(v or {})
        v["uniprot"] = str(acc).strip().upper()
        out.append(v)
    return out


def run_literature_overlap(prot_cols: List[str]) -> Dict[str, Any]:
    print(f"\n{'=' * 60}")
    print("LITERATURE OVERLAP (docs/literature_proteins.yaml)")
    print(f"{'=' * 60}")
    cur = _load_curated()
    if not cur:
        print("  [SKIP] no curated list found")
        return {}
    annot = load_protein_annotation()
    # multi-tissue runs prefix proteins with "PLA:" / "CSF:" — match on the accession
    _acc = lambda c: str(c).upper().split(":")[-1]
    panel = {_acc(c) for c in prot_cols}
    _strip = lambda s: {_acc(x) for x in s}

    stab = pd.read_csv(ROB / "stability_selection.csv") if (ROB / "stability_selection.csv").exists() else None
    locked = (set(pd.read_csv(ROB / "confirmatory_protein_list.csv")["protein"].astype(str))
              if (ROB / "confirmatory_protein_list.csv").exists() else set())
    kstar = (set(pd.read_csv(ROB / "reduced_panel_k_star.csv")["protein"].astype(str))
             if (ROB / "reduced_panel_k_star.csv").exists() else set())
    conf = (pd.read_csv(ROB / "confirmatory_severity.csv").set_index("protein")
            if (ROB / "confirmatory_severity.csv").exists() else None)
    stable = set()
    if stab is not None:
        stab = stab.copy(); stab["protein"] = stab["protein"].astype(str).map(_acc)
        stab = stab[~stab["protein"].duplicated()].set_index("protein")
        stable = set(stab.index[stab["incl_freq_k50"] >= 0.8].astype(str))
    locked, kstar = _strip(locked), _strip(kstar)
    if conf is not None:
        conf = conf.copy(); conf.index = conf.index.astype(str).map(_acc)
        conf = conf[~conf.index.duplicated()]

    rows = []
    for c in cur:
        u = c["uniprot"]
        r: Dict[str, Any] = {
            "uniprot": u, "gene": c.get("gene", annot.get(u, "")),
            "source": c.get("source", ""), "note": c.get("note", ""),
            "treatment_responsive": bool(c.get("treatment_responsive", False)),
            "verify": bool(c.get("verify", False)),
            "on_panel": u in panel,
            "in_locked_list": u in locked, "in_stable_set": u in stable, "in_k_star_panel": u in kstar,
        }
        if stab is not None and u in stab.index:
            r["rank_full"] = int(stab.loc[u, "rank_full"])
            r["incl_freq_k50"] = float(stab.loc[u, "incl_freq_k50"])
            r["w_boot_median"] = float(stab.loc[u, "w_boot_median"])
        if conf is not None and u in conf.index:
            for k in ("train_beta", "train_p", "test_beta", "test_p", "test_levodopa_p"):
                if k in conf.columns:
                    r[k] = conf.loc[u, k]
        rows.append(r)
    df = pd.DataFrame(rows)
    for col in ("rank_full", "incl_freq_k50", "w_boot_median"):
        if col not in df.columns:
            df[col] = np.nan
    df = df.sort_values(["on_panel", "rank_full"], ascending=[False, True],
                        na_position="last")
    df.to_csv(TAB / "literature_overlap.csv", index=False)

    # which of *our* highlighted proteins have literature support?
    hits = {}
    for name, s in (("locked_list", locked), ("stable_set", stable), ("k_star_panel", kstar)):
        lit = df[df["on_panel"]]
        hits[name] = sorted(set(s) & set(lit["uniprot"]))
    summ = {
        "n_curated": int(len(df)), "n_on_panel": int(df["on_panel"].sum()),
        "curated_in_locked_list": hits["locked_list"],
        "curated_in_stable_set": hits["stable_set"],
        "curated_in_k_star_panel": hits["k_star_panel"],
        "treatment_responsive_in_locked_list": sorted(
            set(df.loc[df["treatment_responsive"], "uniprot"]) & locked),
    }
    print(f"  Curated proteins on panel: {summ['n_on_panel']}/{summ['n_curated']}")
    on = df[df["on_panel"]]
    for _, r in on.iterrows():
        flags = [k for k in ("in_locked_list", "in_stable_set", "in_k_star_panel") if r[k]]
        print(f"    {r['uniprot']:<8} {str(r['gene']):<8} rank={r.get('rank_full', np.nan):>5} "
              f"freq50={r.get('incl_freq_k50', np.nan):.2f}  {','.join(flags) or '-'}"
              f"{'  [treatment-responsive]' if r['treatment_responsive'] else ''}")
    if summ["treatment_responsive_in_locked_list"]:
        print(f"  ! Treatment-responsive proteins in the locked list: "
              f"{summ['treatment_responsive_in_locked_list']} -> report the prot_exclude re-run")
    summary_update({"literature_overlap": summ})
    return summ

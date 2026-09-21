"""
Compare two (or more) completed runs — e.g. plasma vs CSF vs PLA+CSF — on the
SAME participants.

    python -m pd_strat.compare_runs results results_CSF results_PLA_CSF
    python -m pd_strat.compare_runs results results_CSF --labels plasma csf --out results_comparison

Fewer participants have CSF than plasma, so headline numbers from separate
runs are not comparable.  This tool intersects the PD participants that have
predictions in every run (`predictions_pd_only.csv`), recomputes the
participant-level within-PD Spearman ρ for each run on that common set, and
reports paired bootstrap Δρ (run_k − run_1) for OOF and TEST.  It also tabulates
the headline summary numbers and the discovery-benchmark TEST deltas of each
run, and draws a dumbbell figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _rho(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[m], b[m]).statistic) if m.sum() >= 3 else np.nan


def _load_pred(d: Path) -> pd.DataFrame:
    p = d / "tables" / "predictions_pd_only.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found (run the pipeline in {d} first)")
    df = pd.read_csv(p)
    df["pid"] = df["participant_id"].astype(str).str.split("-").str[:2].str.join("-")
    return df.groupby(["split", "pid"])[["pred", "y"]].mean().reset_index()


def _summary(d: Path) -> dict:
    p = d / "tables" / "summary.json"
    return json.load(open(p, encoding="utf-8")) if p.exists() else {}


def compare(dirs: List[Path], labels: List[str], out: Path, B: int = 2000, seed: int = 42) -> str:
    out.mkdir(parents=True, exist_ok=True)
    preds = {l: _load_pred(d) for l, d in zip(labels, dirs)}
    sums = {l: _summary(d) for l, d in zip(labels, dirs)}
    rng = np.random.default_rng(seed)
    rows, lines = [], [f"# Run comparison: {' vs '.join(labels)}", ""]

    # ── headline numbers as reported by each run (different participants!) ──
    lines += ["## Headline numbers as reported by each run (NOT the same participants)", ""]
    hdr = ["quantity"] + labels
    tbl = [hdr, ["---"] * len(hdr)]
    for key, getter in (
        ("d_prot", lambda s: s.get("d_prot")),
        ("primary model", lambda s: s.get("primary_model")),
        ("OOF ρ (all rows)", lambda s: s.get("oof_spearman")),
        ("TEST ρ (all rows)", lambda s: s.get("test_spearman")),
        ("within-PD ρ OOF (participant)", lambda s: s.get("validity", {}).get("severity_vs_diagnosis", {}).get("OOF", {}).get("rho_within_pd_participant")),
        ("within-PD ρ TEST (participant)", lambda s: s.get("validity", {}).get("severity_vs_diagnosis", {}).get("TEST", {}).get("rho_within_pd_participant")),
        ("AUROC PD vs HC TEST", lambda s: s.get("validity", {}).get("severity_vs_diagnosis", {}).get("TEST", {}).get("auroc_pd_vs_hc_participant")),
        ("PD-only refit ρ TEST", lambda s: s.get("validity", {}).get("refit_pd_only", {}).get("TEST_participant", {}).get("spearman")),
        ("n PD participants TRAIN", lambda s: s.get("validity", {}).get("severity_vs_diagnosis", {}).get("OOF", {}).get("n_pd_participants")),
        ("n PD participants TEST", lambda s: s.get("validity", {}).get("severity_vs_diagnosis", {}).get("TEST", {}).get("n_pd_participants")),
    ):
        vals = []
        for l in labels:
            try:
                v = getter(sums[l])
            except Exception:
                v = None
            vals.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        tbl.append([key] + vals)
    lines += ["| " + " | ".join(r) + " |" for r in tbl] + [""]

    # ── discovery deltas per run ─────────────────────────────────────────
    disc = {l: sums[l].get("discovery", {}).get("best", []) for l in labels}
    if any(disc.values()):
        lines += ["## Discovery benchmark: TEST Δ vs clinical (best proteomic config per target)", ""]
        targets = list(dict.fromkeys(r["target"] for l in labels for r in disc[l]))
        tbl = [["target"] + [f"{l}: Δ [95% CI]" for l in labels], ["---"] * (len(labels) + 1)]
        for t in targets:
            row = [t]
            for l in labels:
                r = next((x for x in disc[l] if x["target"] == t), None)
                if r and "test_delta_vs_clinical" in r:
                    row.append(f"{r['test_delta_vs_clinical']:+.3f} [{r.get('test_best_delta_ci_lo', np.nan):+.3f}, "
                               f"{r.get('test_best_delta_ci_hi', np.nan):+.3f}] ({r['best_feature_set']}/{r['best_model']})")
                else:
                    row.append("–")
            tbl.append(row)
        lines += ["| " + " | ".join(r) + " |" for r in tbl] + [""]

    # ── same-participant comparison ──────────────────────────────────────
    lines += ["## Within-PD ρ on the SAME participants (participant-level, PD-only model)", ""]
    for split in ("TRAIN_OOF", "TEST"):
        common = None
        for l in labels:
            s = set(preds[l][preds[l]["split"] == split]["pid"])
            common = s if common is None else common & s
        common = sorted(common or [])
        if len(common) < 10:
            lines += [f"_{split}: fewer than 10 common participants_", ""]
            continue
        mats = {}
        for l in labels:
            s = preds[l][(preds[l]["split"] == split) & (preds[l]["pid"].isin(common))].set_index("pid").loc[common]
            mats[l] = (s["pred"].values, s["y"].values)
        y = mats[labels[0]][1]
        base = mats[labels[0]][0]
        tbl = [["run", "n common", "ρ", "Δρ vs " + labels[0], "95% CI", "p"], ["---"] * 6]
        for l in labels:
            p = mats[l][0]
            r = _rho(p, y)
            if l == labels[0]:
                tbl.append([l, str(len(common)), f"{r:.3f}", "–", "–", "–"])
                rows.append({"split": split, "run": l, "n_common": len(common), "rho": r})
                continue
            deltas = []
            for _ in range(B):
                ix = rng.integers(0, len(common), len(common))
                deltas.append(_rho(p[ix], y[ix]) - _rho(base[ix], y[ix]))
            deltas = np.array(deltas)
            lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
            pv = float(min(1.0, 2 * min(np.mean(deltas >= 0), np.mean(deltas <= 0))))
            tbl.append([l, str(len(common)), f"{r:.3f}", f"{r - _rho(base, y):+.3f}", f"[{lo:+.3f}, {hi:+.3f}]", f"{pv:.3f}"])
            rows.append({"split": split, "run": l, "n_common": len(common), "rho": r,
                         "delta_vs_first": r - _rho(base, y), "delta_ci_lo": lo, "delta_ci_hi": hi, "p": pv})
        lines += [f"**{split}**", ""] + ["| " + " | ".join(r) + " |" for r in tbl] + [""]

    # ── overlap of top proteins ──────────────────────────────────────────
    tops = {}
    for l, d in zip(labels, dirs):
        p = d / "robustness" / "stability_selection.csv"
        if p.exists():
            s = pd.read_csv(p).head(40)["protein"].astype(str).str.split(":").str[-1]
            tops[l] = set(s)
    if len(tops) >= 2:
        lines += ["## Overlap of top-40 stability-ranked proteins (by accession)", ""]
        ks = list(tops)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                inter = tops[ks[i]] & tops[ks[j]]
                lines += [f"- {ks[i]} ∩ {ks[j]}: {len(inter)} / 40 — {', '.join(sorted(inter)) or 'none'}"]
        lines += [""]

    pd.DataFrame(rows).to_csv(out / "same_participant_comparison.csv", index=False)
    text = "\n".join(lines)
    (out / "run_comparison.md").write_text(text, encoding="utf-8")
    _figure(rows, labels, out)
    print(text)
    print(f"\n[compare_runs] written to {out}/")
    return text


def _figure(rows, labels, out: Path):
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5.4, 0.55 * len(labels) + 1.6), dpi=300)
    cols = {"TRAIN_OOF": "#2a78d6", "TEST": "#eb6834"}
    for k, split in enumerate(("TRAIN_OOF", "TEST")):
        s = df[df["split"] == split].set_index("run").reindex(labels)
        if s["rho"].isna().all():
            continue
        yv = np.arange(len(labels)) + (0.15 if k else -0.15)
        ax.scatter(s["rho"], yv, s=40, color=cols[split], edgecolor="#fcfcfb", lw=1, zorder=3,
                   label=f"{split} (n={int(s['n_common'].dropna().iloc[0])})")
        for i, l in enumerate(labels):
            if i and np.isfinite(s.loc[l, "delta_ci_lo"]):
                r0 = s.loc[labels[0], "rho"]
                ax.plot([r0 + s.loc[l, "delta_ci_lo"], r0 + s.loc[l, "delta_ci_hi"]], [yv[i], yv[i]],
                        color=cols[split], lw=1.4, alpha=0.6)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set_xlabel("Within-PD Spearman ρ on the same participants (bar = 95% CI of Δ vs first run)")
    ax.grid(color="#e1e0d9", lw=0.6); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Compartment comparison on matched participants", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "fig_run_comparison.png", facecolor="#fcfcfb"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Compare completed pipeline runs on the same participants")
    ap.add_argument("dirs", nargs="+", help="results directories (first = reference)")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", default=None, help="output dir (default: results_comparison)")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    dirs = [Path(d) for d in a.dirs]
    labels = a.labels or [d.name for d in dirs]
    compare(dirs, labels, Path(a.out or "results_comparison"), B=a.boot)


if __name__ == "__main__":
    main()

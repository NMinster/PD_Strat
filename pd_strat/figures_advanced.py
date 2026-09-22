"""
§12b — Figures that carry the paper's argument (static PNG, 300 dpi).

All read the CSV tables written by the analysis modules, so they can be
re-generated stand-alone (`python -c "from pd_strat.figures_advanced import
run_advanced_figures; run_advanced_figures()"`).

Palette: reference data-viz palette (validated with the dataviz skill's
validator — categorical slots 1-3 pass all-pairs; ordinal blue ramp 250/450/650;
diverging blue-gray-red; status 'critical' reserved for treatment-responsive
proteins and always paired with a distinct marker + label).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .config import FIG, TAB, ROB
from .utils import spearman_np, summary

# ── palette ─────────────────────────────────────────────────────────────────
C_TRAIN, C_TEST, C_THIRD = "#2a78d6", "#eb6834", "#1baf7a"     # categorical 1-3
ORD3 = ["#86b6ef", "#2a78d6", "#104281"]                        # ordinal ramp (T1<T2<T3)
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV = ["#2a78d6", "#f0efec", "#e34948"]                         # blue - gray - red
C_CRIT = "#d03b3b"                                               # status: treatment-responsive
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300, "figure.facecolor": SURF, "axes.facecolor": SURF,
        "font.family": "sans-serif", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "grid.linestyle": "-", "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.labelcolor": INK2, "axes.titlecolor": INK, "legend.frameon": False,
        "legend.fontsize": 7, "lines.linewidth": 2, "lines.solid_capstyle": "round",
    })
    return plt


def _div_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("bgr", DIV)


def _seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("blues", SEQ)


def _save(fig, name, made: List[str]):
    fig.savefig(FIG / name, bbox_inches="tight", facecolor=SURF)
    import matplotlib.pyplot as plt
    plt.close(fig)
    made.append(name)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  1. discovery heatmap + 2. incremental dumbbell                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def fig_discovery(made):
    plt = _style()
    p = TAB / "discovery_grid.csv"
    if not p.exists():
        return
    g = pd.read_csv(p)
    targets = list(dict.fromkeys(g["target"]))
    fsets = ["clinical", "proteomics", "proteomics+clinical"]
    fig, axes = plt.subplots(1, len(targets), figsize=(2.6 * len(targets) + 1.2, 4.2), squeeze=False)
    cmap = _seq_cmap()
    for ax, t in zip(axes[0], targets):
        sub = g[g["target"] == t]
        models = list(dict.fromkeys(sub["model"]))
        M = np.full((len(models), len(fsets)), np.nan)
        for i, m in enumerate(models):
            for j, f in enumerate(fsets):
                v = sub[(sub["model"] == m) & (sub["feature_set"] == f)]["oof_metric_pooled"]
                if len(v):
                    M[i, j] = float(v.iloc[0])
        vmin, vmax = np.nanmin(M), np.nanmax(M)
        ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.grid(False)
        for i in range(len(models)):
            for j in range(len(fsets)):
                if np.isfinite(M[i, j]):
                    frac = (M[i, j] - vmin) / (vmax - vmin + 1e-9)
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if frac > 0.55 else INK)
        # ring the best proteomic configuration
        pm = M.copy(); pm[:, 0] = np.nan
        if np.isfinite(pm).any():
            bi, bj = np.unravel_index(np.nanargmax(pm), pm.shape)
            ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, lw=1.6, ec=INK))
        ax.set_xticks(range(len(fsets))); ax.set_xticklabels(["clinical", "prot.", "prot.+clin."], fontsize=7)
        ax.set_yticks(range(len(models))); ax.set_yticklabels(models if ax is axes[0][0] else [""] * len(models), fontsize=7)
        task = sub["task"].iloc[0]
        ax.set_title(f"{t}\n(OOF {'AUROC' if task == 'clf' else 'Spearman ρ'}, n={int(sub['n'].iloc[0])})")
    fig.suptitle("Discovery benchmark — nested CV on TRAIN; ring = configuration carried to TEST",
                 fontsize=9, color=INK2, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_discovery_heatmap.png", made)

    b = TAB / "discovery_best.csv"
    if not b.exists():
        return
    bd = pd.read_csv(b)
    fig, ax = plt.subplots(figsize=(6.2, 0.75 * len(bd) + 1.4))
    yv = np.arange(len(bd))
    for i, r in bd.iterrows():
        for k, (col_c, col_b, off, lab) in enumerate((("oof_clinical", "oof_best", -0.18, "OOF (TRAIN CV)"),
                                                      ("test_clinical", "test_best", 0.18, "TEST (PDBP)"))):
            if col_c not in bd or not np.isfinite(r.get(col_c, np.nan)) or not np.isfinite(r.get(col_b, np.nan)):
                continue
            col = C_TRAIN if k == 0 else C_TEST
            ax.plot([r[col_c], r[col_b]], [i + off, i + off], color=col, lw=2, alpha=0.5, zorder=1)
            ax.scatter([r[col_c]], [i + off], s=36, facecolor=SURF, edgecolor=col, lw=1.6, zorder=3,
                       label=f"{lab}: clinical" if i == 0 else None)
            ax.scatter([r[col_b]], [i + off], s=36, color=col, zorder=3,
                       label=f"{lab}: best proteomic" if i == 0 else None)
            lo, hi = r.get(f"{'oof' if k == 0 else 'test'}_best_delta_ci_lo", np.nan), \
                     r.get(f"{'oof' if k == 0 else 'test'}_best_delta_ci_hi", np.nan)
            d = r[col_b] - r[col_c]
            ax.text(max(r[col_c], r[col_b]) + 0.02, i + off,
                    f"Δ={d:+.2f} [{lo:+.2f}, {hi:+.2f}]", va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(yv); ax.set_yticklabels([f"{r['target']}\n{r['best_feature_set']} / {r['best_model']}"
                                           for _, r in bd.iterrows()], fontsize=7)
    ax.invert_yaxis(); ax.set_xlabel("Spearman ρ (continuous targets) or AUROC (fast progressor)")
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_xlim(left=min(-0.05, ax.get_xlim()[0]), right=ax.get_xlim()[1] + 0.35)
    fig.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    ax.set_title("Does the proteome add to baseline clinical scoring?  (hollow = clinical, filled = best proteomic)")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, "fig_discovery_incremental.png", made)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  3. trajectory fans by baseline proteomic-severity tertile               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def fig_trajectories(made):
    plt = _style()
    frames = [(s, pd.read_csv(TAB / f"progression_trajectories_{s}.csv"))
              for s in ("TRAIN", "TEST") if (TAB / f"progression_trajectories_{s}.csv").exists()]
    if not frames:
        return
    fig, axes = plt.subplots(1, len(frames), figsize=(3.8 * len(frames), 3.4), squeeze=False, sharey=True)
    bins = np.arange(0, 61, 6)
    for ax, (split, df) in zip(axes[0], frames):
        df = df[df["months"] <= 60]
        for pid, g in df.groupby("pid"):
            g = g.sort_values("months")
            ax.plot(g["months"], g["y"], color=MUTED, lw=0.5, alpha=0.18, zorder=1)
        for t, col in zip((1, 2, 3), ORD3):
            sub = df[df["pred0_tertile"] == t].copy()
            if sub.empty:
                continue
            sub["bin"] = pd.cut(sub["months"], bins, labels=(bins[:-1] + 3), include_lowest=True)
            agg = sub.groupby("bin", observed=True)["y"].agg(["mean", "sem", "count"])
            agg = agg[agg["count"] >= 5]
            x = agg.index.astype(float)
            ax.fill_between(x, agg["mean"] - agg["sem"], agg["mean"] + agg["sem"], color=col, alpha=0.12, lw=0)
            ax.plot(x, agg["mean"], color=col, lw=2, zorder=3,
                    label=f"T{t} ({'lowest' if t == 1 else 'highest' if t == 3 else 'middle'} predicted severity, n={sub['pid'].nunique()})")
            ax.scatter(x, agg["mean"], s=18, color=col, edgecolor=SURF, lw=1, zorder=4)
        ax.set_title(split)
        ax.set_xlabel("Months from baseline proteomic sample")
        if ax is axes[0][0]:
            ax.set_ylabel("MDS-UPDRS total")
        ax.legend(loc="upper left")
    fig.suptitle("UPDRS trajectories by baseline proteomic-severity tertile "
                 "(thin lines = individual participants; bold = tertile mean ± SE)",
                 fontsize=9, color=INK2, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_trajectories_by_tertile.png", made)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  4. Kaplan–Meier by tertile                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _km(t, e):
    order = np.argsort(t); t, e = t[order], e[order]
    times, surv, s = [0.0], [1.0], 1.0
    for ut in np.unique(t[e == 1]):
        at_risk = (t >= ut).sum(); d = ((t == ut) & (e == 1)).sum()
        s *= 1 - d / at_risk
        times.append(ut); surv.append(s)
    return np.array(times), np.array(surv)


def _logrank(t, e, g):
    from scipy.stats import chi2
    groups = np.unique(g); k = len(groups)
    O = np.zeros(k); E = np.zeros(k); V = np.zeros((k, k))
    for ut in np.unique(t[e == 1]):
        atr = t >= ut; n = atr.sum(); d = ((t == ut) & (e == 1)).sum()
        nj = np.array([(atr & (g == gg)).sum() for gg in groups])
        dj = np.array([((t == ut) & (e == 1) & (g == gg)).sum() for gg in groups])
        O += dj; E += d * nj / n
        if n > 1:
            for a in range(k):
                for b in range(k):
                    V[a, b] += d * (n - d) / (n - 1) * (nj[a] / n) * ((a == b) - nj[b] / n)
    x = (O - E)[:-1]; Vm = V[:-1, :-1]
    try:
        stat = float(x @ np.linalg.solve(Vm, x))
        return stat, float(chi2.sf(stat, k - 1))
    except np.linalg.LinAlgError:
        return np.nan, np.nan


def fig_km(made):
    plt = _style()
    frames = [(s, pd.read_csv(TAB / f"progression_km_{s}.csv"))
              for s in ("TRAIN", "TEST") if (TAB / f"progression_km_{s}.csv").exists()]
    if not frames:
        return
    eps = list(dict.fromkeys(pd.concat([f for _, f in frames])["endpoint"]))
    for ep in eps:
        fig, axes = plt.subplots(1, len(frames), figsize=(3.8 * len(frames), 3.2), squeeze=False, sharey=True)
        for ax, (split, df) in zip(axes[0], frames):
            d = df[(df["endpoint"] == ep)].dropna(subset=["time", "event", "pred0_tertile"])
            d = d[d["time"] > 0]
            if len(d) < 20:
                ax.set_visible(False); continue
            for t, col in zip((1, 2, 3), ORD3):
                s = d[d["pred0_tertile"] == t]
                if len(s) < 5:
                    continue
                x, y = _km(s["time"].values.astype(float), s["event"].values.astype(int))
                ax.step(x, y, where="post", color=col, lw=2,
                        label=f"T{t} (n={len(s)}, events={int(s['event'].sum())})")
            stat, p = _logrank(d["time"].values.astype(float), d["event"].values.astype(int),
                               d["pred0_tertile"].values.astype(int))
            ax.text(0.98, 0.95, f"log-rank p = {p:.3g}", ha="right", va="top", transform=ax.transAxes,
                    fontsize=7, color=INK2)
            ax.set_ylim(0, 1.02); ax.set_xlabel("Months"); ax.set_title(f"{split}: time to {ep}")
            if ax is axes[0][0]:
                ax.set_ylabel("Event-free probability")
            ax.legend(loc="lower left")
        fig.suptitle("Milestone-free survival by baseline proteomic-severity tertile", fontsize=9, color=INK2, y=1.02)
        fig.tight_layout()
        _save(fig, f"fig_km_{ep}.png", made)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  5. trial-enrichment curves                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def fig_enrichment(made):
    plt = _style()
    p = TAB / "discovery_enrichment.csv"
    if not p.exists():
        return
    e = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    ax.plot(e["enrolled_fraction"] * 100, e["fast_progressor_rate"], color=C_TEST, marker="o", ms=5,
            markeredgecolor=SURF, label="enrolled by predicted risk")
    ax.axhline(e["base_rate"].iloc[0], color=AXIS, lw=1, label="unselected (base rate)")
    ax.set_xlabel("% of TEST PD participants enrolled (highest predicted risk first)")
    ax.set_ylabel("Fraction who are fast progressors"); ax.set_ylim(0, 1); ax.legend(loc="upper right")
    ax.set_title("Enrichment of fast progressors")
    ax = axes[1]
    ax.plot(e["enrolled_fraction"] * 100, e["relative_trial_sample_size"], color=C_TEST, marker="o", ms=5,
            markeredgecolor=SURF)
    ax.axhline(1, color=AXIS, lw=1)
    ax.set_xlabel("% enrolled"); ax.set_ylabel("Relative trial sample size\n(same power, slope endpoint)")
    ax.set_title("Implied sample-size ratio vs unselected trial")
    fig.tight_layout()
    _save(fig, "fig_trial_enrichment.png", made)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  6. replication scatter (TRAIN β vs TEST β, treatment-responsive flagged)║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def fig_replication(made):
    plt = _style()
    p = ROB / "confirmatory_severity.csv"
    if not p.exists():
        return
    c = pd.read_csv(p).dropna(subset=["train_beta", "test_beta"])
    if c.empty:
        return
    n_b = len(c)
    levo = c.get("test_levodopa_p", pd.Series(np.nan, index=c.index)) < 0.05 / n_b
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    lim = np.nanmax(np.abs(np.r_[c["train_ci_lo"], c["train_ci_hi"], c["test_ci_lo"], c["test_ci_hi"]])) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color=AXIS, lw=0.8)
    ax.axhline(0, color=AXIS, lw=0.8); ax.axvline(0, color=AXIS, lw=0.8)
    for flag, col, mk, lab in ((False, C_TRAIN, "o", "not medication-associated"),
                               (True, C_CRIT, "^", "levodopa-associated in TEST (Bonferroni)")):
        s = c[levo == flag]
        if s.empty:
            continue
        ax.errorbar(s["train_beta"], s["test_beta"],
                    xerr=[s["train_beta"] - s["train_ci_lo"], s["train_ci_hi"] - s["train_beta"]],
                    yerr=[s["test_beta"] - s["test_ci_lo"], s["test_ci_hi"] - s["test_beta"]],
                    fmt="none", ecolor=col, elinewidth=0.6, alpha=0.5, capsize=0)
        ax.scatter(s["train_beta"], s["test_beta"], s=34, color=col, marker=mk, edgecolor=SURF, lw=1,
                   zorder=3, label=lab)
    lab_df = c.reindex(c["test_beta"].abs().sort_values(ascending=False).index).head(6)
    for k, (_, r) in enumerate(lab_df.iterrows()):
        nm = r["gene"] if isinstance(r.get("gene"), str) and r["gene"] not in ("", "nan") else r["protein"]
        # alternate offsets so neighbouring labels do not collide
        dx, dy = ((6, 6), (6, -9), (-6, 6), (-6, -9))[k % 4]
        ax.annotate(nm, (r["train_beta"], r["test_beta"]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=6.5, color=INK2, ha="left" if dx > 0 else "right",
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.5))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("TRAIN β (UPDRS points per SD protein, 95% CI)")
    ax.set_ylabel("TEST β (95% CI)")
    ax.set_title("Cross-cohort replication of the locked proteins")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, "fig_replication_scatter.png", made)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  7. protein × endpoint correlation map                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def fig_protein_endpoint(made):
    plt = _style()
    p = TAB / "protein_endpoint_correlations.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    splits = [s for s in ("TRAIN", "TEST") if s in set(d["split"])]
    eps = list(dict.fromkeys(d["endpoint"]))
    prots = list(dict.fromkeys(d["protein"]))
    lab = {r["protein"]: (r["gene"] if isinstance(r["gene"], str) and r["gene"] not in ("", "nan") else r["protein"])
           for _, r in d.drop_duplicates("protein").iterrows()}
    fig, axes = plt.subplots(1, len(splits), figsize=(0.55 * len(eps) * len(splits) + 2.4, 0.22 * len(prots) + 1.4),
                             squeeze=False, sharey=True)
    cmap = _div_cmap(); vmax = 0.6
    for ax, s in zip(axes[0], splits):
        M = np.full((len(prots), len(eps)), np.nan)
        sub = d[d["split"] == s]
        for _, r in sub.iterrows():
            M[prots.index(r["protein"]), eps.index(r["endpoint"])] = r["rho"]
        im = ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.grid(False)
        ax.set_xticks(range(len(eps))); ax.set_xticklabels(eps, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(prots))); ax.set_yticklabels([lab[p_] for p_ in prots], fontsize=6)
        ax.set_title(f"{s} (PD, visit-matched samples)")
    cb = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.03, pad=0.02)
    cb.set_label("Spearman ρ (protein z vs endpoint)", fontsize=7)
    fig.suptitle("Which clinical axis does each top protein track?", fontsize=9, color=INK2, y=1.0)
    _save(fig, "fig_protein_endpoint_heatmap.png", made)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  8. within-PD prediction density + 9. model-comparison forest            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def fig_within_pd(made):
    plt = _style()
    p = TAB / "predictions_pd_only.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    df["pid"] = df["participant_id"].astype(str).str.split("-").str[:2].str.join("-")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, split, col in zip(axes, ("TRAIN_OOF", "TEST"), (C_TRAIN, C_TEST)):
        s = df[df["split"] == split].groupby("pid")[["pred", "y"]].mean().dropna()
        if len(s) < 5:
            ax.set_visible(False); continue
        lim = [min(s.min()), max(s.max())]
        ax.hexbin(s["y"], s["pred"], gridsize=18, cmap=_seq_cmap(), mincnt=1, linewidths=0.2, edgecolors=SURF)
        ax.plot(lim, lim, color=AXIS, lw=0.8)
        b = np.polyfit(s["y"], s["pred"], 1)
        xs = np.linspace(lim[0], lim[1], 20); ax.plot(xs, np.polyval(b, xs), color=col, lw=2)
        ax.set_xlabel("Observed MDS-UPDRS (participant mean)"); ax.set_ylabel("Predicted (PD-only model)")
        ax.set_title(f"{split.replace('_', ' ')}: ρ = {spearman_np(s['pred'].values, s['y'].values):.2f}, "
                     f"slope = {b[0]:.2f}, n = {len(s)}")
        ax.grid(False)
    fig.suptitle("Within-PD severity prediction (density of participants)", fontsize=9, color=INK2, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_within_pd_density.png", made)

    q = TAB / "validity_model_comparison.csv"
    if q.exists():
        m = pd.read_csv(q)
        if not m.empty:
            comps = list(dict.fromkeys(m["comparator"]))
            fig, ax = plt.subplots(figsize=(5.2, 0.4 * len(comps) + 1.4))
            for k, (split, col) in enumerate((("OOF", C_TRAIN), ("TEST", C_TEST))):
                s = m[m["split"] == split].set_index("comparator").reindex(comps)
                yv = np.arange(len(comps)) + (0.16 if k else -0.16)
                ax.errorbar(s["delta_rho"], yv, xerr=[s["delta_rho"] - s["delta_ci_lo"], s["delta_ci_hi"] - s["delta_rho"]],
                            fmt="o", ms=5, color=col, ecolor=col, elinewidth=1.2, capsize=2,
                            markeredgecolor=SURF, label=split)
            ax.axvline(0, color=AXIS, lw=0.8)
            ax.set_yticks(range(len(comps))); ax.set_yticklabels(comps, fontsize=7); ax.invert_yaxis()
            ax.set_xlabel(f"Δρ vs primary ({m['primary'].iloc[0]}), participant-level paired bootstrap 95% CI")
            ax.set_title("Candidate models vs the primary (Δρ < 0 = worse than primary)")
            ax.legend(loc="lower right")
            fig.tight_layout()
            _save(fig, "fig_model_comparison_forest.png", made)


def fig_within_person(made):
    """Consecutive-visit Δscore vs ΔUPDRS (the monitoring-biomarker picture)."""
    plt = _style()
    p = TAB / "longitudinal_pairs.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    d = d[(d["target"] == "y") & (d["model"] == "primary")]
    splits = [s for s in ("TRAIN", "TEST") if s in set(d["split"])]
    if not splits:
        return
    fig, axes = plt.subplots(1, len(splits), figsize=(3.7 * len(splits), 3.4), squeeze=False)
    for ax, split, col in zip(axes[0], splits, (C_TRAIN, C_TEST)):
        s = d[d["split"] == split]
        ax.axhline(0, color=AXIS, lw=0.8); ax.axvline(0, color=AXIS, lw=0.8)
        ax.scatter(s["score"], s["tgt"], s=12, alpha=0.55, color=col, edgecolor=SURF, lw=0.5)
        m = np.isfinite(s["score"]) & np.isfinite(s["tgt"])
        if m.sum() > 5:
            b = np.polyfit(s["score"][m], s["tgt"][m], 1)
            xs = np.linspace(s["score"].min(), s["score"].max(), 20)
            ax.plot(xs, np.polyval(b, xs), color=INK, lw=1.2)
        ax.set_xlabel("Δ predicted severity between consecutive samples")
        ax.set_ylabel("Δ MDS-UPDRS between the same visits")
        ax.set_title(f"{split}: ρ = {spearman_np(s['score'].values, s['tgt'].values):.2f}, "
                     f"{len(s)} visit pairs")
    fig.suptitle("Within-person change: does the score move with the clinic?", fontsize=9, color=INK2, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_within_person_coupling.png", made)


def run_advanced_figures() -> List[str]:
    made: List[str] = []
    for fn in (fig_discovery, fig_trajectories, fig_km, fig_enrichment, fig_replication,
               fig_protein_endpoint, fig_within_pd, fig_within_person):
        try:
            fn(made)
        except Exception as e:  # never let a figure kill the run
            print(f"  [figures/advanced] {fn.__name__} skipped: {e}")
    print(f"[Figures/advanced] {len(made)} saved: {made}")
    return made

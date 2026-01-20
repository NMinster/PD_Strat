# %% ========== GSEA (GSEAPreranked) post-processing + publication-quality plots ==========
from __future__ import annotations
import re, math, os
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------- CONFIG: EDIT THESE TWO PATHS --------------------------------
ROOT = Path(r"C:\Users\NM\gsea_home\output\sep19")             # folder that contains the GSEA result folders
OUT  = Path(r"C:\Users\NM\gsea_home\output\sep19\figs")        # output folder for figures and tables
FDR_CUTOFF = 0.25
TOPN_BARS  = 25
TOPN_PER_SUBTYPE_FOR_BUBBLE = 15
# -----------------------------------------------------------------------------------------

sns.set_context("talk", font_scale=0.9)
sns.set_style("whitegrid")
mpl.rcParams["figure.dpi"] = 120
mpl.rcParams["savefig.dpi"] = 300

def _cmap(name: str):
    """Modern Matplotlib colormap accessor (avoids deprecation)."""
    return mpl.colormaps.get_cmap(name)

def _detect_collection(name: str) -> str:
    if name.startswith("GOBP_"): return "GO:BP"
    if name.startswith("GOCC_"): return "GO:CC"
    if name.startswith("GOMF_"): return "GO:MF"
    if name.startswith("HP_"):   return "HPO"
    return "Other"

def _clean_term(name: str) -> str:
    # remove namespace prefix + underscores -> space, title casing
    cleaned = re.sub(r"^(GOBP_|GOCC_|GOMF_|HP_)", "", name)
    cleaned = cleaned.replace("_", " ").strip()
    # Title-case but keep roman numerals/acronyms
    cleaned = " ".join([w if w.isupper() else w.capitalize() for w in cleaned.split()])
    return cleaned

def _subtype_from_path(p: Path) -> str:
    """
    Derive a nice label from the GSEA result folder name.

    Supports:
      - Old style: my_analysis.GseaPreranked.Subtype_0  -> "Subtype 0"
      - New style: K3_cluster2_MIXED_vsHC.GseaPreranked.* -> "Cluster 2"
    """
    s = p.name

    # New pattern: K3_cluster2_MIXED_vsHC.GseaPreranked...
    m_cluster = re.search(r"K\d+_cluster(\d+)_MIXED_vsHC", s, flags=re.IGNORECASE)
    if m_cluster:
        return f"Cluster {m_cluster.group(1)}"

    # Old pattern: Subtype_0, Subtype-1, Subtype 2, etc.
    m_sub = re.search(r"(Subtype[_\s-]*\d+)", s, flags=re.IGNORECASE)
    if m_sub:
        # Normalize: "Subtype_0" -> "Subtype 0"
        label = m_sub.group(1)
        label = label.replace("-", " ").replace("_", " ")
        label = re.sub(r"\s+", " ", label).strip().title()  # "Subtype 0"
        return label

    # Fallback: use folder name directly
    return s

def _load_one_report(tsv: Path, subtype_label: str) -> pd.DataFrame:
    df = pd.read_csv(tsv, sep="\t")
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    # Direction from file name (pos/neg)
    fn = tsv.name.lower()
    direction = "pos" if "_pos_" in fn else ("neg" if "_neg_" in fn else ("pos" if df.get("NES", pd.Series([0])).iloc[0] >= 0 else "neg"))
    # Collection & clean term
    df["collection"] = df["NAME"].map(_detect_collection)
    df["term_clean"] = df["NAME"].map(_clean_term)
    df["direction"] = direction
    df["subtype"] = subtype_label   # already pretty ("Cluster 0", "Subtype 1", etc.)
    # helpful transform
    if "FDR q-val" in df.columns:
        with np.errstate(divide="ignore"):
            df["neglog10_FDR"] = -np.log10(df["FDR q-val"].replace(0, 1e-300))
    if "NES" in df.columns:
        df["NES"] = pd.to_numeric(df["NES"], errors="coerce")
    return df

def load_all(root: Path) -> pd.DataFrame:
    """Find GSEAPreranked result folders (clusters/subtypes) and ingest both pos/neg reports."""
    all_rows = []

    # Look for any *.GseaPreranked* folder under ROOT
    candidates = sorted(
        p for p in root.glob("**/*.GseaPreranked*")
        if p.is_dir()
    )

    # Optionally, restrict to cluster/Subtype patterns so we ignore random GSEA runs
    candidates = [
        p for p in candidates
        if re.search(r"K\d+_cluster\d+_MIXED_vsHC", p.name, re.IGNORECASE)  # K3_cluster*_MIXED_vsHC
        or re.search(r"Subtype[_\s-]*\d+", p.name, re.IGNORECASE)           # Subtype_*
    ]

    if not candidates:
        raise FileNotFoundError(f"No cluster/subtype GSEAPreranked folders found under: {root}")

    for subdir in candidates:
        subtype_label = _subtype_from_path(subdir)
        for tsv in sorted(subdir.glob("gsea_report_for_*_*.tsv")):
            try:
                all_rows.append(_load_one_report(tsv, subtype_label))
            except Exception as e:
                print(f"(!) Skipped {tsv}: {e}")

    if not all_rows:
        raise FileNotFoundError("No GSEA report TSVs were found (pattern gsea_report_for_*_*.tsv).")
    df = pd.concat(all_rows, ignore_index=True)
    # Deduplicate: keep best FDR (then highest NES) per subtype × term × direction
    df = (df
          .sort_values(["subtype","term_clean","direction","FDR q-val","NES"],
                       ascending=[True, True, True, True, False])
          .drop_duplicates(subset=["subtype","term_clean","direction"], keep="first"))
    return df

# ------------------------------ PLOTS ----------------------------------------------------

def plot_top_positive_bars(df: pd.DataFrame, outdir: Path, subtype: str, topn: int, fdr_cut: float):
    sub = df.query("subtype == @subtype and direction == 'pos' and `FDR q-val` <= @fdr_cut and NES.notnull()")
    if sub.empty: return
    top = (sub.sort_values(["NES","FDR q-val"], ascending=[False, True]).head(topn)
             .assign(label=lambda d: d["term_clean"].str.slice(0, 80)))
    fig, ax = plt.subplots(figsize=(6.0, max(3, 0.26*len(top)+1)))
    cmap = _cmap("viridis")
    # Color by FDR (log scale), reverse so most significant is darkest
    vmin = max(min(sub["FDR q-val"].min(), fdr_cut/100), 1e-4)
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=fdr_cut)
    colors = cmap(norm(top["FDR q-val"].values))
    ax.barh(top["label"][::-1], top["NES"][::-1], color=colors[::-1], edgecolor="none")
    ax.set_xlabel("Normalized Enrichment Score (NES)")
    ax.set_ylabel("")
    ax.set_title(f"{subtype} • Top positively enriched (FDR ≤ {fdr_cut})")
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("FDR q-value (log scale)")
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in (".png",".pdf"):
        fig.savefig(outdir / f"{subtype.replace(' ','_')}_topNES_pos_bars{ext}", bbox_inches="tight")
    plt.close(fig)

def plot_rank_at_max(df: pd.DataFrame, outdir: Path, subtype: str, fdr_cut: float):
    sub = df.query("subtype == @subtype and `FDR q-val` <= @fdr_cut and `RANK AT MAX`.notnull()")
    if sub.empty: return
    x = sub["RANK AT MAX"].astype(float)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    if x.nunique(dropna=True) <= 1:
        # variance 0: draw a vertical marker
        val = float(x.iloc[0])
        ax.axvline(val, color="black", lw=2, alpha=0.75)
        ax.text(val, 0.95, f"all at {int(val)}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
    else:
        sns.kdeplot(x, ax=ax, fill=True, lw=1, alpha=0.6, warn_singular=False)
    ax.set_xlabel("Rank at ES max"); ax.set_ylabel("Density")
    ax.set_title(f"{subtype} • Distribution of peak enrichment positions (FDR ≤ {fdr_cut})")
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in (".png",".pdf"):
        fig.savefig(outdir / f"{subtype.replace(' ','_')}_rank_at_max_density{ext}", bbox_inches="tight")
    plt.close(fig)

def plot_cross_subtype_bubble(df: pd.DataFrame, outdir: Path, topn_per_subtype: int, fdr_cut: float):
    # pick top terms per subtype by combined score
    picks = []
    for st, g in df.groupby("subtype"):
        gsig = g.query("`FDR q-val` <= @fdr_cut").copy()
        if gsig.empty: 
            continue
        gsig["score"] = gsig["neglog10_FDR"] * gsig["NES"].abs()
        picks.append(gsig.sort_values("score", ascending=False).head(topn_per_subtype))
    if not picks:
        return
    M = pd.concat(picks, ignore_index=True)
    M = M[["subtype", "term_clean", "NES", "neglog10_FDR"]].drop_duplicates(["subtype","term_clean"])
    # Orders
    term_order = M.groupby("term_clean")["NES"].mean().sort_values(ascending=False).index.tolist()
    try:
        subtype_order = sorted(M["subtype"].unique(), key=lambda s: int(re.search(r"\d+", s).group()))
    except Exception:
        subtype_order = list(M["subtype"].unique())
    # Plot
    fig, ax = plt.subplots(figsize=(7.8, max(3.2, 0.25*len(term_order))))
    vabs = max(2.0, float(M["NES"].abs().max()))
    norm = mpl.colors.TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
    cmap = _cmap("coolwarm")
    x = pd.Categorical(M["subtype"], categories=subtype_order, ordered=True)
    y = pd.Categorical(M["term_clean"], categories=term_order, ordered=True)
    sizes = (M["neglog10_FDR"].clip(0, 20) + 1.0) ** 1.1 * 10
    sc = ax.scatter(x.codes, y.codes, s=sizes, c=M["NES"], cmap=cmap, norm=norm,
                    alpha=0.85, edgecolor="k", linewidths=0.2)
    ax.set_xticks(range(len(subtype_order))); ax.set_xticklabels(subtype_order)
    ax.set_yticks(range(len(term_order)));   ax.set_yticklabels(term_order)
    ax.set_title("Cross-subtype enrichment (size = −log10 FDR; color = NES)")
    ax.set_xlabel(""); ax.set_ylabel("")
    cbar = fig.colorbar(sc, ax=ax, pad=0.01); cbar.set_label("NES (blue=negative, red=positive)")
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in (".png",".pdf"):
        fig.savefig(outdir / f"cross_subtype_bubble_matrix{ext}", bbox_inches="tight")
    plt.close(fig)

# ------------------------------ RUN ------------------------------------------------------

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "tables").mkdir(exist_ok=True)
print(f"Scanning: {ROOT}")

df_all = load_all(ROOT)

# Save a tidy table
tidy_csv = OUT / "tables" / "gsea_combined_tidy.csv"
df_all.to_csv(tidy_csv, index=False)
print(f"✓ Combined tidy table: {tidy_csv}")

# Quick, de-duplicated preview table (best FDR per subtype×term×direction)
preview_cols = ["term_clean","NES","FDR q-val","collection","direction","subtype"]
preview = (df_all
           .query("`FDR q-val` <= @FDR_CUTOFF and NES.notnull()")
           [preview_cols]
           .sort_values(["subtype","direction","FDR q-val","NES"], ascending=[True, True, True, False])
           .groupby(["subtype","direction"])
           .head(12))
preview_path = OUT / "tables" / "quick_preview_top.csv"
preview.to_csv(preview_path, index=False)
print(f"✓ Preview table: {preview_path}")

# Per-subtype (per-cluster) plots
subtypes = sorted(
    df_all["subtype"].unique(),
    key=lambda s: int(re.search(r"\d+", s).group()) if re.search(r"\d+", s) else 999
)
for st in subtypes:
    plot_top_positive_bars(df_all, OUT, st, TOPN_BARS, FDR_CUTOFF)
    plot_rank_at_max(df_all, OUT, st, FDR_CUTOFF)

# Cross-subtype bubble matrix
plot_cross_subtype_bubble(df_all, OUT, TOPN_PER_SUBTYPE_FOR_BUBBLE, FDR_CUTOFF)

print(f"✓ Wrote figures to: {OUT}")
try:
    from IPython.display import display
    display(preview.head(20))
except Exception:
    pass
# =========================================================================================



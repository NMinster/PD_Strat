"""
Data loading, merging, QC diagnostics, and train/test/val splitting.

Functions
---------
load_proteomics(panels_dict)
    Load and pivot Olink proteomics panels to wide format.
load_rnaseq(path)
    Load RNA-seq data and identify feature columns.
load_deg(path, adj_p_thresh, logfc_thresh)
    Load and filter DEG file.
merge_datasets(rnaseq_data, df_proteomics_wide)
    Inner-join RNA-seq and proteomics on PATNO.
run_qc_diagnostics(df_proteomics_wide, rnaseq_data, prot_feat_cols, rna_feature_cols, output_dir)
    Run missingness and distribution diagnostics (no transforms applied).
run_batch_assessment(df_proteomics_wide, rnaseq_data, df_proteomics_long, ...)
    PCA-based batch / site effect assessment.
build_datasets(merged_data, deg_gene_ids, df_proteomics_wide)
    Build Combined, Proteomics-only, and RNA-seq-only dataset dicts.
create_splits(datasets, random_state)
    Create PDBP train/test + PPMI validation splits.
build_visit_level_proteomics(df_proteomics_long)
    Build visit-level proteomics wide table for time-stratified analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.stats import kruskal, f_oneway

from .config import RANDOM_STATE, OUTPUT_DIR


def get_cohort(patno):
    """Derive cohort from PATNO prefix (AMP-PD convention)."""
    s = str(patno)
    if s.startswith("PP-"):
        return "PPMI"
    elif s.startswith("PD-"):
        return "PDBP"
    return "Other"


def load_proteomics(panels_dict):
    """Load and pivot Olink proteomics panels to wide format.

    Returns (df_proteomics_long, df_proteomics_wide).
    """
    panel_dfs = []
    for panel_name, path in panels_dict.items():
        df = pd.read_csv(path)
        df["panel"] = panel_name
        panel_dfs.append(df)
        print(f"  {panel_name}: {df.shape[0]:,} rows, {df['UniProt'].nunique()} proteins")

    df_proteomics = pd.concat(panel_dfs, ignore_index=True)
    print(f"\nTotal proteomics: {df_proteomics.shape[0]:,} rows, "
          f"{df_proteomics['UniProt'].nunique()} unique proteins, "
          f"{df_proteomics['participant_id'].nunique()} subjects")

    df_proteomics_wide = (
        df_proteomics
        .pivot_table(index="participant_id", columns="UniProt", values="NPX", aggfunc="mean")
        .reset_index()
        .rename(columns={"participant_id": "PATNO"})
    )
    print(f"Proteomics wide: {df_proteomics_wide.shape}")
    return df_proteomics, df_proteomics_wide


def load_rnaseq(path):
    """Load RNA-seq data and identify feature columns.

    Returns (rnaseq_data, rna_feature_cols).
    """
    rnaseq_data = pd.read_csv(path)
    if "participant_id" in rnaseq_data.columns:
        rnaseq_data.rename(columns={"participant_id": "PATNO"}, inplace=True)
    print(f"RNA-seq: {rnaseq_data.shape}")

    rna_feature_cols = [c for c in rnaseq_data.columns if c not in ["PATNO", "pd"]]
    print(f"RNA feature columns (first 5): {rna_feature_cols[:5]}")
    print(f"Total RNA features: {len(rna_feature_cols)}")
    return rnaseq_data, rna_feature_cols


def load_deg(path, adj_p_thresh=0.05, logfc_thresh=0.1):
    """Load and filter DEG file.

    Returns (df_deg, deg_gene_ids).
    """
    df_deg = pd.read_csv(path)
    filtered_deg = df_deg[(df_deg["adj.P.Val"] < adj_p_thresh) &
                          (df_deg["logFC"] > logfc_thresh)]
    deg_gene_ids = filtered_deg["gene_id"].tolist()
    print(f"DEGs passing filter (adj.P<{adj_p_thresh}, logFC>{logfc_thresh}): "
          f"{len(deg_gene_ids)}")
    return df_deg, deg_gene_ids


def merge_datasets(rnaseq_data, df_proteomics_wide):
    """Inner-join RNA-seq and proteomics on PATNO."""
    merged_data = pd.merge(rnaseq_data, df_proteomics_wide, on="PATNO", how="inner")
    print(f"Merged: {merged_data.shape} ({merged_data['PATNO'].nunique()} subjects)")
    return merged_data


def run_qc_diagnostics(df_proteomics_wide, rnaseq_data,
                       prot_feat_cols, rna_feature_cols, output_dir=OUTPUT_DIR):
    """Run missingness & distribution diagnostics (no transforms applied).

    Returns (RNA_NEEDS_LOG, rna_expressed).
    """
    print("=" * 60)
    print("QUALITY CONTROL — DIAGNOSTIC ONLY")
    print("=" * 60)

    # ── Missingness ──
    print("\n--- 1. Missingness Assessment ---")
    prot_missing = df_proteomics_wide[prot_feat_cols].isnull().mean()
    print(f"\nProteomics ({len(prot_feat_cols)} features):")
    print(f"  Mean missingness per feature: {prot_missing.mean()*100:.2f}%")
    print(f"  Max missingness:              {prot_missing.max()*100:.2f}%")
    print(f"  Features >10% missing:        {(prot_missing > 0.10).sum()}")
    print(f"  Features >20% missing:        {(prot_missing > 0.20).sum()}")

    rna_missing = rnaseq_data[rna_feature_cols].isnull().mean()
    print(f"\nRNA-seq ({len(rna_feature_cols)} features):")
    print(f"  Mean missingness per feature: {rna_missing.mean()*100:.2f}%")
    print(f"  Max missingness:              {rna_missing.max()*100:.2f}%")
    print(f"  Features >10% missing:        {(rna_missing > 0.10).sum()}")

    prot_sample_missing = df_proteomics_wide[prot_feat_cols].isnull().mean(axis=1)
    rna_sample_missing = rnaseq_data[rna_feature_cols].isnull().mean(axis=1)
    print(f"\nPer-sample missingness:")
    print(f"  Proteomics: median={prot_sample_missing.median()*100:.2f}%, "
          f"max={prot_sample_missing.max()*100:.2f}%")
    print(f"  RNA-seq:    median={rna_sample_missing.median()*100:.2f}%, "
          f"max={rna_sample_missing.max()*100:.2f}%")

    # ── Distribution diagnostics ──
    print("\n--- 2. Distribution Diagnostics ---")
    prot_sample_vals = df_proteomics_wide[prot_feat_cols].values.flatten()
    prot_sample_vals = prot_sample_vals[~np.isnan(prot_sample_vals)]
    print(f"\nProteomics (Olink NPX):")
    print(f"  Range: [{np.min(prot_sample_vals):.2f}, {np.max(prot_sample_vals):.2f}]")
    print(f"  Median: {np.median(prot_sample_vals):.2f}")
    is_npx_log_scale = np.max(prot_sample_vals) < 50
    if is_npx_log_scale:
        print("  Olink NPX is pre-normalized (log2) — no additional log needed")
    else:
        print("  Values look large for NPX — verify data provenance")

    rna_sample_vals = rnaseq_data[rna_feature_cols[:500]].values.flatten()
    rna_sample_vals = rna_sample_vals[~np.isnan(rna_sample_vals)]
    rna_max = np.max(rna_sample_vals)
    rna_median = np.median(rna_sample_vals)
    rna_pct_zero = (rna_sample_vals == 0).mean() * 100

    print(f"\nRNA-seq:")
    print(f"  Range: [{np.min(rna_sample_vals):.2f}, {rna_max:.2f}]")
    print(f"  Median: {rna_median:.2f}")
    print(f"  % zeros: {rna_pct_zero:.1f}%")

    rna_already_log = rna_max < 30 and rna_median < 15
    RNA_NEEDS_LOG = not rna_already_log

    if rna_already_log:
        print("  RNA-seq appears already log-transformed — log2 will NOT be applied")
    else:
        print("  RNA-seq appears to be raw counts/TPM — log2(x+1) WILL be applied")

    # Low-variance features
    prot_var = df_proteomics_wide[prot_feat_cols].var()
    rna_var = rnaseq_data[rna_feature_cols].var()
    rna_expressed = [c for c in rna_feature_cols if rna_var.get(c, 0) >= 1e-6]

    print(f"\n  Low-variance features:")
    print(f"    Proteomics near-zero: {(prot_var < 1e-6).sum()}/{len(prot_feat_cols)}")
    print(f"    RNA near-zero:        {len(rna_feature_cols) - len(rna_expressed)}/{len(rna_feature_cols)}")

    # ── Diagnostic plots ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].hist(prot_sample_vals, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
    axes[0].set_title("Proteomics (NPX) — Raw")
    axes[0].set_xlabel("NPX value")
    axes[0].set_ylabel("Count")

    axes[1].hist(rna_sample_vals[rna_sample_vals > 0], bins=80, color="#E57373",
                 edgecolor="none", alpha=0.8)
    axes[1].set_title("RNA-seq — Raw (non-zero)")
    axes[1].set_xlabel("Expression value")

    axes[2].bar(["Proteomics", "RNA-seq"],
                [prot_missing.mean() * 100, rna_missing.mean() * 100],
                color=["steelblue", "#E57373"], edgecolor="black")
    axes[2].set_ylabel("Mean % missing per feature")
    axes[2].set_title("Feature Missingness")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/qc_distributions_raw.png", dpi=200)
    plt.show()

    print(f"\n  NO TRANSFORMS APPLIED in this step.")
    print(f"  Flags set: RNA_NEEDS_LOG={RNA_NEEDS_LOG}")
    return RNA_NEEDS_LOG, rna_expressed


def needs_correction(pcs, labels, threshold=0.01):
    """Test whether PCs differ significantly by cohort (Kruskal-Wallis)."""
    unique = [c for c in pd.unique(labels) if c != "Other"]
    if len(unique) < 2:
        return False, 1.0
    for pc_idx in range(min(2, pcs.shape[1])):
        groups = [pcs[labels == c, pc_idx] for c in unique]
        groups = [g for g in groups if len(g) >= 5]
        if len(groups) >= 2:
            _, pval = kruskal(*groups)
            if pval < threshold:
                return True, pval
    return False, 1.0


def run_batch_assessment(df_proteomics_wide, rnaseq_data, df_proteomics_long,
                         prot_feat_cols, rna_feature_cols, rna_expressed,
                         output_dir=OUTPUT_DIR):
    """PCA-based batch / site effect assessment.

    Returns NEEDS_BATCH_CORRECTION flag.
    """
    print("=" * 60)
    print("BATCH EFFECT ASSESSMENT")
    print("=" * 60)

    # Proteomics PCA
    prot_for_pca = df_proteomics_wide[prot_feat_cols].dropna()
    prot_patno = df_proteomics_wide.loc[prot_for_pca.index, "PATNO"]
    prot_cohort = prot_patno.apply(get_cohort)

    prot_scaled = StandardScaler().fit_transform(prot_for_pca)
    prot_pca = PCA(n_components=5, random_state=RANDOM_STATE)
    prot_pcs = prot_pca.fit_transform(prot_scaled)
    prot_var_explained = prot_pca.explained_variance_ratio_

    print(f"\n  Proteomics: {len(prot_for_pca)} samples")
    print(f"  Variance explained: PC1={prot_var_explained[0]:.3f}, PC2={prot_var_explained[1]:.3f}")

    # RNA PCA
    rna_for_pca_cols = rna_expressed[:5000] if len(rna_expressed) > 5000 else rna_expressed
    rna_for_pca = rnaseq_data[rna_for_pca_cols].dropna()
    rna_patno = rnaseq_data.loc[rna_for_pca.index, "PATNO"]
    rna_cohort = rna_patno.apply(get_cohort)

    rna_scaled = StandardScaler().fit_transform(rna_for_pca)
    rna_pca = PCA(n_components=5, random_state=RANDOM_STATE)
    rna_pcs = rna_pca.fit_transform(rna_scaled)
    rna_var_explained = rna_pca.explained_variance_ratio_

    # Site info
    prot_has_site = "site" in df_proteomics_long.columns

    # Visualization
    n_panels = 3 if prot_has_site else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))

    for cohort, color in [("PPMI", "#2196F3"), ("PDBP", "#FF9800"), ("Other", "gray")]:
        mask = prot_cohort.values == cohort
        if mask.any():
            axes[0].scatter(prot_pcs[mask, 0], prot_pcs[mask, 1],
                            alpha=0.4, s=15, c=color, label=cohort)
    axes[0].set_xlabel(f"PC1 ({prot_var_explained[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({prot_var_explained[1]:.1%})")
    axes[0].set_title("Proteomics PCA — by Cohort")
    axes[0].legend()

    for cohort, color in [("PPMI", "#2196F3"), ("PDBP", "#FF9800"), ("Other", "gray")]:
        mask = rna_cohort.values == cohort
        if mask.any():
            axes[1].scatter(rna_pcs[mask, 0], rna_pcs[mask, 1],
                            alpha=0.4, s=15, c=color, label=cohort)
    axes[1].set_xlabel(f"PC1 ({rna_var_explained[0]:.1%})")
    axes[1].set_ylabel(f"PC2 ({rna_var_explained[1]:.1%})")
    axes[1].set_title("RNA-seq PCA — by Cohort")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/qc_pca_batch_assessment.png", dpi=200)
    plt.savefig(f"{output_dir}/qc_pca_batch_assessment.eps", format="eps", dpi=300)
    plt.show()

    # Statistical tests
    prot_needs_combat, prot_batch_p = needs_correction(prot_pcs, prot_cohort.values)
    rna_needs_combat, rna_batch_p = needs_correction(rna_pcs, rna_cohort.values)
    NEEDS_BATCH_CORRECTION = prot_needs_combat or rna_needs_combat

    print(f"\n  Proteomics PC1/2 ~ cohort: "
          f"{'SIGNIFICANT' if prot_needs_combat else 'n.s.'} (best p={prot_batch_p:.2e})")
    print(f"  RNA-seq PC1/2 ~ cohort:    "
          f"{'SIGNIFICANT' if rna_needs_combat else 'n.s.'} (best p={rna_batch_p:.2e})")

    if NEEDS_BATCH_CORRECTION:
        print(f"\n  Batch effects detected. Correction will be applied FOLD-AWARE.")
    else:
        print(f"\n  No significant batch effects. StandardScaler suffices.")

    return NEEDS_BATCH_CORRECTION


def make_analysis_copies(rnaseq_data, merged_data, rna_feature_cols, RNA_NEEDS_LOG):
    """Create globally log-transformed copies for descriptive analyses only.

    Returns (rnaseq_analysis, merged_analysis).
    """
    rnaseq_analysis = rnaseq_data.copy()
    merged_analysis = merged_data.copy()

    if RNA_NEEDS_LOG:
        print("  Applying log2(x+1) to RNA analysis copies...")
        rnaseq_analysis[rna_feature_cols] = np.log2(
            rnaseq_analysis[rna_feature_cols].clip(lower=0) + 1)
        for col in rna_feature_cols:
            if col in merged_analysis.columns:
                merged_analysis[col] = np.log2(merged_analysis[col].clip(lower=0) + 1)
    else:
        print("  RNA-seq already log-scale — no transform needed")

    return rnaseq_analysis, merged_analysis


def build_datasets(merged_data, deg_gene_ids, df_proteomics_wide, rna_feature_cols):
    """Build Combined, Proteomics-only, and RNA-seq-only dataset dicts.

    Returns (datasets, columns_rna, proteomics_cols, proteomics_cols_only).
    """
    columns_rna = [c for c in merged_data.columns if c in deg_gene_ids]
    proteomics_cols = [
        c for c in merged_data.columns
        if c not in ["PATNO", "pd"] and c not in deg_gene_ids
    ]
    proteomics_cols_only = [
        c for c in merged_data.columns
        if c in df_proteomics_wide.columns and c not in ["PATNO", "pd"]
    ]

    datasets = {
        "Combined":        merged_data[["PATNO", "pd"] + columns_rna + proteomics_cols].dropna(),
        "Proteomics only": merged_data[["PATNO", "pd"] + proteomics_cols_only].dropna(),
        "RNA-seq only":    merged_data[["PATNO", "pd"] + columns_rna].dropna(),
    }

    for name, df in datasets.items():
        n_pos = df["pd"].sum()
        n_neg = len(df) - n_pos
        print(f"{name:20s}: {df.shape[1]-2:5d} features, "
              f"{len(df):4d} samples (PD={n_pos}, HC={n_neg})")

    return datasets, columns_rna, proteomics_cols, proteomics_cols_only


def create_splits(datasets, random_state=RANDOM_STATE):
    """Create PDBP train/test + PPMI validation splits.

    Returns dict of {dataset_name: split_dict}.
    """
    splits = {}
    for name, df in datasets.items():
        train_test = df[df["PATNO"].str.startswith("PD-")].copy()
        val = df[df["PATNO"].str.startswith("PP-")].copy()

        X_tt = train_test.drop(columns=["PATNO", "pd"])
        y_tt = train_test["pd"]
        groups_tt = train_test["PATNO"]

        X_val = val.drop(columns=["PATNO", "pd"])
        y_val = val["pd"]

        X_train, X_test, y_train, y_test, grp_train, grp_test = train_test_split(
            X_tt, y_tt, groups_tt,
            test_size=0.2, random_state=random_state, stratify=y_tt,
        )

        splits[name] = {
            "X_train": X_train, "y_train": y_train, "groups_train": grp_train,
            "X_test": X_test, "y_test": y_test,
            "X_val": X_val, "y_val": y_val,
            "X_tt": X_tt, "y_tt": y_tt, "groups_tt": groups_tt,
        }
        print(f"{name}: train={len(X_train)}, test={len(X_test)}, "
              f"val={len(X_val)}, features={X_train.shape[1]}")

    return splits


def build_visit_level_proteomics(df_proteomics_long):
    """Build visit-level proteomics wide table.

    Returns df_prot_visit.
    """
    print("Building visit-level proteomics wide table...")
    df_prot_visit = (
        df_proteomics_long
        .dropna(subset=["visit_month", "NPX"])
        .pivot_table(
            index=["participant_id", "visit_month"],
            columns="UniProt", values="NPX", aggfunc="mean",
        )
        .reset_index()
        .rename(columns={"participant_id": "PATNO"})
    )
    print(f"  Visit-level proteomics: {df_prot_visit.shape}")
    print(f"  Subjects: {df_prot_visit['PATNO'].nunique()}")
    return df_prot_visit

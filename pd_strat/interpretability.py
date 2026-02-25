"""
Interpretability and robustness diagnostics (Section 6).

Feature selection stability across CV folds, SHAP rank stability,
SHAP variability plot, and correlation heatmap of top proteins.

Functions
---------
run_feature_selection_stability(splits, preprocessor_configs, top_32_proteins, output_dir)
    CV-fold feature selection + SHAP ranking stability analysis.
plot_selection_frequency(df_freq, top_32_proteins, n_cv_splits, output_dir)
    Two-view selection frequency bar plots.
run_shap_rank_stability(fold_shap_rankings, n_cv_splits, output_dir)
    Pairwise Spearman rank correlations + top-k Jaccard overlap.
plot_shap_variability(fold_shap_rankings, output_dir)
    Mean |SHAP| +/- SD across folds for top 32 features.
plot_protein_correlation_heatmap(splits, top_32_proteins, output_dir)
    Spearman correlation heatmap of top proteomic markers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from itertools import combinations
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.model_selection import GroupKFold
from imblearn.pipeline import Pipeline as IMBPipeline

from .config import RANDOM_STATE, N_CV_SPLITS, OUTPUT_DIR


def run_feature_selection_stability(splits, preprocessor_configs,
                                    top_32_proteins=None,
                                    output_dir=OUTPUT_DIR):
    """Run CV-fold feature selection stability + SHAP ranking analysis.

    Returns (fold_selected_features, fold_shap_rankings, df_freq).
    """
    print("=== Feature Selection Stability (Proteomics-only) ===\n")

    prot_s = splits["Proteomics only"]
    group_kfold = GroupKFold(n_splits=N_CV_SPLITS)

    fold_selected_features = []
    fold_shap_rankings = []

    for fold_idx, (train_idx, test_idx) in enumerate(
            group_kfold.split(prot_s["X_train"], prot_s["y_train"],
                              prot_s["groups_train"])):

        X_fold_train = prot_s["X_train"].iloc[train_idx]
        y_fold_train = prot_s["y_train"].iloc[train_idx]
        X_fold_test = prot_s["X_train"].iloc[test_idx]

        fold_pipe = IMBPipeline([
            ('variance_threshold', VarianceThreshold(threshold=0.0)),
            ('feature_selection', SelectKBest(score_func=f_classif, k=41)),
            ('preprocessor', preprocessor_configs['Proteomics only']),
            ('model', RandomForestClassifier(
                n_estimators=1000, class_weight='balanced',
                random_state=RANDOM_STATE)),
        ])
        fold_pipe.fit(X_fold_train, y_fold_train)

        vt_mask = fold_pipe.named_steps['variance_threshold'].get_support()
        filtered_cols = X_fold_train.columns[vt_mask]
        kb_mask = fold_pipe.named_steps['feature_selection'].get_support()
        selected = filtered_cols[kb_mask].tolist()
        fold_selected_features.append(set(selected))

        X_test_transformed = fold_pipe[:-1].transform(X_fold_test)
        explainer = shap.TreeExplainer(fold_pipe.named_steps['model'])
        shap_vals = explainer.shap_values(X_test_transformed)
        mean_shap = np.abs(shap_vals[1]).mean(axis=0)

        ranking = {selected[i]: mean_shap[i] if i < len(mean_shap) else 0
                   for i in range(len(selected))}
        fold_shap_rankings.append(ranking)
        print(f"  Fold {fold_idx+1}/{N_CV_SPLITS} done")

    print(f"\nCompleted {N_CV_SPLITS}-fold stability analysis")

    # Build frequency table
    all_features = set()
    for fs in fold_selected_features:
        all_features.update(fs)

    selection_freq = {
        feat: sum(1 for fs in fold_selected_features if feat in fs)
        for feat in all_features
    }
    df_freq = pd.DataFrame([
        {"Feature": k, "Selection_Frequency": v, "Fraction": v / N_CV_SPLITS}
        for k, v in selection_freq.items()
    ]).sort_values("Selection_Frequency", ascending=False)

    # Plot and save
    plot_selection_frequency(df_freq, top_32_proteins, N_CV_SPLITS, output_dir)
    run_shap_rank_stability(fold_shap_rankings, N_CV_SPLITS, output_dir)
    plot_shap_variability(fold_shap_rankings, output_dir)
    plot_protein_correlation_heatmap(splits, top_32_proteins, output_dir)

    df_freq.to_csv(f"{output_dir}/feature_selection_frequency.csv", index=False)
    return fold_selected_features, fold_shap_rankings, df_freq


def plot_selection_frequency(df_freq, top_32_proteins, n_cv_splits,
                             output_dir=OUTPUT_DIR):
    """Two-view selection frequency bar plots."""
    # View 1: Top 32 by frequency
    top32_by_freq = df_freq.head(32)

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    ax = axes[0]
    ax.barh(range(len(top32_by_freq)), top32_by_freq["Fraction"],
            color=plt.cm.viridis(top32_by_freq["Fraction"].values),
            edgecolor='black')
    ax.set_yticks(range(len(top32_by_freq)))
    ax.set_yticklabels(top32_by_freq["Feature"], fontsize=7)
    ax.set_xlabel(f"Selection Frequency (out of {n_cv_splits} folds)")
    ax.set_title("View 1: Top 32 Proteins by Selection Frequency\n"
                 "(data-driven ranking)")
    ax.axvline(0.5, color='red', linestyle='--', alpha=0.5,
               label='50% threshold')
    ax.invert_yaxis()
    ax.legend()

    # View 2: Fixed 32-protein panel
    if top_32_proteins is not None:
        selection_freq = dict(zip(df_freq["Feature"], df_freq["Selection_Frequency"]))
        panel_freq = df_freq[df_freq["Feature"].isin(top_32_proteins)].copy()

        never_selected = [p for p in top_32_proteins if p not in selection_freq]
        if never_selected:
            never_df = pd.DataFrame([
                {"Feature": p, "Selection_Frequency": 0, "Fraction": 0.0}
                for p in never_selected
            ])
            panel_freq = pd.concat([panel_freq, never_df], ignore_index=True)

        panel_freq = panel_freq.sort_values("Fraction", ascending=True)

        tier_colors = []
        for frac in panel_freq["Fraction"]:
            if frac >= 0.8:
                tier_colors.append('#4CAF50')
            elif frac >= 0.5:
                tier_colors.append('#FF9800')
            else:
                tier_colors.append('#F44336')

        ax = axes[1]
        ax.barh(range(len(panel_freq)), panel_freq["Fraction"],
                color=tier_colors, edgecolor='black')
        ax.set_yticks(range(len(panel_freq)))
        ax.set_yticklabels(panel_freq["Feature"], fontsize=7)
        ax.set_xlabel(f"Selection Frequency (out of {n_cv_splits} folds)")
        ax.set_title("View 2: Fixed 32-Protein Panel Stability\n"
                     "(predefined panel from SHAP)")
        ax.axvline(0.5, color='red', linestyle='--', alpha=0.5,
                   label='50% threshold')
        ax.axvline(0.8, color='green', linestyle=':', alpha=0.5,
                   label='80% threshold')
        ax.set_xlim(0, 1.05)
        ax.legend(fontsize=8)

        # Summary statistics
        n_stable = (panel_freq["Fraction"] >= 0.8).sum()
        n_moderate = ((panel_freq["Fraction"] >= 0.5) &
                      (panel_freq["Fraction"] < 0.8)).sum()
        n_unstable = (panel_freq["Fraction"] < 0.5).sum()
        median_frac = panel_freq["Fraction"].median()

        print(f"\nFixed 32-protein panel stability ({len(panel_freq)} proteins):")
        print(f"  Highly stable (>=80%): {n_stable}")
        print(f"  Moderate (50-80%):     {n_moderate}")
        print(f"  Unstable (<50%):       {n_unstable}")
        print(f"  Median frequency:      {median_frac:.2f}")

        # Overlap
        top32_set = set(top32_by_freq["Feature"])
        panel_set = set(top_32_proteins)
        overlap = top32_set & panel_set
        print(f"\nOverlap analysis:")
        print(f"  In both:        {len(overlap)}/32")
        print(f"  Panel-only:     {len(panel_set - top32_set)}")
        print(f"  Frequency-only: {len(top32_set - panel_set)}")

        panel_freq.to_csv(f"{output_dir}/fixed_panel_stability.csv",
                          index=False)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_selection_stability.eps",
                format='eps', dpi=300)
    plt.savefig(f"{output_dir}/feature_selection_stability.png", dpi=300)
    plt.show()


def run_shap_rank_stability(fold_shap_rankings, n_cv_splits,
                            output_dir=OUTPUT_DIR):
    """Compute SHAP rank stability via pairwise Spearman correlations."""
    all_shap_features = set()
    for ranking in fold_shap_rankings:
        all_shap_features.update(ranking.keys())

    rank_matrix = pd.DataFrame(index=sorted(all_shap_features))
    for fold_idx, ranking in enumerate(fold_shap_rankings):
        rank_matrix[f"Fold_{fold_idx}"] = pd.Series(ranking)

    rank_matrix_ranks = rank_matrix.rank(ascending=False, method='average')

    rank_corrs = []
    for i, j in combinations(range(n_cv_splits), 2):
        common = rank_matrix_ranks[[f"Fold_{i}", f"Fold_{j}"]].dropna()
        if len(common) > 5:
            r, _ = spearmanr(common.iloc[:, 0], common.iloc[:, 1])
            rank_corrs.append(r)

    print(f"SHAP Rank Stability:")
    print(f"  Mean pairwise Spearman rho: {np.mean(rank_corrs):.4f} "
          f"+/- {np.std(rank_corrs):.4f}")
    print(f"  Range: [{min(rank_corrs):.4f}, {max(rank_corrs):.4f}]")

    for k in [5, 10, 20]:
        print(f"  Top-{k} Jaccard overlap: "
              f"{top_k_overlap(fold_shap_rankings, k=k):.4f}")


def top_k_overlap(rankings, k=10):
    """Compute mean pairwise Jaccard overlap of top-k features."""
    top_sets = []
    for ranking in rankings:
        sorted_feats = sorted(ranking.keys(),
                              key=lambda x: ranking[x], reverse=True)[:k]
        top_sets.append(set(sorted_feats))
    jaccards = [
        len(s1 & s2) / len(s1 | s2)
        for s1, s2 in combinations(top_sets, 2)
        if len(s1 | s2) > 0
    ]
    return np.mean(jaccards) if jaccards else 0


def plot_shap_variability(fold_shap_rankings, output_dir=OUTPUT_DIR):
    """Plot mean |SHAP| +/- SD across folds for top 32 features."""
    all_shap_features = set()
    for ranking in fold_shap_rankings:
        all_shap_features.update(ranking.keys())

    shap_summary = {}
    for feat in all_shap_features:
        vals = [r.get(feat, 0) for r in fold_shap_rankings]
        shap_summary[feat] = {"mean": np.mean(vals), "std": np.std(vals)}

    df = pd.DataFrame(shap_summary).T.sort_values(
        "mean", ascending=False).head(32)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(df)), df["mean"],
            xerr=df["std"], color='steelblue', edgecolor='black',
            capsize=3, alpha=0.85)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index, fontsize=7)
    ax.set_xlabel("Mean |SHAP| +/- SD across folds")
    ax.set_title("SHAP Importance Stability -- Top 32 Features")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_stability.eps", format='eps', dpi=300)
    plt.savefig(f"{output_dir}/shap_stability.png", dpi=300)
    plt.show()


def plot_protein_correlation_heatmap(splits, top_32_proteins,
                                     output_dir=OUTPUT_DIR):
    """Spearman correlation heatmap of top proteomic markers."""
    if top_32_proteins is None:
        return

    prot_s = splits["Proteomics only"]
    available = [p for p in top_32_proteins
                 if p in prot_s["X_train"].columns]

    if len(available) < 5:
        print("Too few proteins available for correlation heatmap")
        return

    corr_matrix = prot_s["X_train"][available].corr(method='spearman')

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix, mask=mask, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
        annot=len(available) <= 20,
        fmt='.2f' if len(available) <= 20 else '',
        square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Spearman rho"}, ax=ax,
    )
    ax.set_title("Correlation Heatmap -- Top 32 Proteomic Markers\n"
                 "(Collinearity context)")
    ax.tick_params(axis='both', labelsize=7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/protein_correlation_heatmap.eps",
                format='eps', dpi=300)
    plt.savefig(f"{output_dir}/protein_correlation_heatmap.png", dpi=300)
    plt.show()

    # Report highly correlated pairs
    high_corr = [
        (corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
        for i in range(len(corr_matrix))
        for j in range(i + 1, len(corr_matrix))
        if abs(corr_matrix.iloc[i, j]) > 0.5
    ]
    if high_corr:
        print(f"Highly correlated pairs (|rho| > 0.5): {len(high_corr)}")
        for p1, p2, r in sorted(high_corr, key=lambda x: abs(x[2]),
                                  reverse=True)[:10]:
            print(f"  {p1} -- {p2}: rho = {r:.3f}")

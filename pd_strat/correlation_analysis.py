"""
mRNA-Protein correlation analysis: baseline, visit-stratified, lagged,
and monotonicity assessment.
"""

import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
from sklearn.isotonic import IsotonicRegression

from .config import RANDOM_STATE, OUTPUT_DIR


def filtered_correlation(prot_vals, rna_vals, min_expr_prot_pct=1, min_expr_rna_pct=1):
    """Compute Spearman correlation after removing unexpressed samples.

    Returns (spearman_r, spearman_p, n_used, n_filtered).
    """
    valid = pd.DataFrame({"prot": prot_vals.values, "rna": rna_vals.values}).dropna()
    n_total = len(valid)
    if n_total < 10:
        return np.nan, np.nan, n_total, 0

    prot_thresh = np.percentile(valid["prot"], min_expr_prot_pct)
    rna_thresh = np.percentile(valid["rna"], min_expr_rna_pct)

    mask = (valid["prot"] > prot_thresh) & (valid["rna"] > rna_thresh) & (valid["rna"] > 0)
    filtered = valid[mask]
    n_filtered = n_total - len(filtered)

    if len(filtered) < 10:
        return np.nan, np.nan, len(filtered), n_filtered

    r, p = spearmanr(filtered["prot"], filtered["rna"])
    return r, p, len(filtered), n_filtered


def run_baseline_correlations(top_32_proteins, merged_analysis,
                              uniprot_to_symbol, uniprot_to_ensg):
    """Compute baseline (cross-sectional) mRNA-protein correlations.

    Returns df_corr_baseline DataFrame.
    """
    print("\n--- Part 1: Baseline (mean across visits) Correlations ---")
    baseline_results = []
    for protein_id in top_32_proteins:
        gene_symbol = uniprot_to_symbol.get(protein_id, "?")
        ensg_matches = uniprot_to_ensg.get(protein_id, [])

        if ensg_matches and protein_id in merged_analysis.columns:
            ensg_col = None
            for e in ensg_matches:
                if e in merged_analysis.columns:
                    ensg_col = e
                    break

            if ensg_col:
                r_sp, p_sp, n_used, n_filt = filtered_correlation(
                    merged_analysis[protein_id], merged_analysis[ensg_col])
                r_pe, p_pe = np.nan, np.nan
                if n_used >= 10:
                    valid = merged_analysis[[protein_id, ensg_col]].dropna()
                    prot_t = np.percentile(valid[protein_id], 1)
                    rna_t = np.percentile(valid[ensg_col], 1)
                    filt = valid[(valid[protein_id] > prot_t) &
                                (valid[ensg_col] > max(rna_t, 0))]
                    if len(filt) >= 10:
                        r_pe, p_pe = pearsonr(filt[protein_id], filt[ensg_col])

                baseline_results.append({
                    "Protein_UniProt": protein_id, "Gene_Symbol": gene_symbol,
                    "RNA_ENSG": ensg_col,
                    "n_total": len(merged_analysis[[protein_id, ensg_col]].dropna()),
                    "n_after_filter": n_used, "n_filtered_out": n_filt,
                    "Spearman_r": r_sp, "Spearman_p": p_sp,
                    "Pearson_r": r_pe, "Pearson_p": p_pe,
                })
                continue

        baseline_results.append({
            "Protein_UniProt": protein_id, "Gene_Symbol": gene_symbol,
            "RNA_ENSG": "NO MATCH",
            "n_total": 0, "n_after_filter": 0, "n_filtered_out": 0,
            "Spearman_r": np.nan, "Spearman_p": np.nan,
            "Pearson_r": np.nan, "Pearson_p": np.nan,
        })

    df_corr_baseline = pd.DataFrame(baseline_results)
    n_matched = df_corr_baseline["Spearman_r"].notna().sum()
    print(f"  Correlations computed: {n_matched}/{len(df_corr_baseline)}")
    return df_corr_baseline


def run_visit_stratified_correlations(top_32_proteins, df_prot_visit,
                                      rnaseq_analysis, uniprot_to_ensg,
                                      uniprot_to_symbol, rna_has_visit):
    """Compute visit-stratified mRNA-protein correlations.

    Returns df_corr_visits DataFrame.
    """
    print("\n--- Part 2: Visit-Stratified Correlations ---")
    gc.collect()

    _rna_ensg_needed = set()
    for _p in top_32_proteins[:10]:
        _rna_ensg_needed.update(uniprot_to_ensg.get(_p, []))
    _rna_merge_cols = ["PATNO"] + [c for c in _rna_ensg_needed if c in rnaseq_analysis.columns]
    if "visit_month_rna" in rnaseq_analysis.columns:
        _rna_merge_cols = (["PATNO", "visit_month_rna"] +
                          [c for c in _rna_ensg_needed if c in rnaseq_analysis.columns])

    visit_corr_results = []
    available_visits = sorted(df_prot_visit["visit_month"].dropna().unique())

    for visit_m in available_visits:
        prot_at_visit = df_prot_visit[df_prot_visit["visit_month"] == visit_m]
        if rna_has_visit:
            rna_visit_months = sorted(rnaseq_analysis["visit_month_rna"].dropna().unique())
            if visit_m in rna_visit_months:
                rna_at_visit = rnaseq_analysis[rnaseq_analysis["visit_month_rna"] == visit_m]
                rna_visit_label = f"M{int(visit_m)}"
            else:
                rna_at_visit = rnaseq_analysis[rnaseq_analysis["visit_month_rna"] == 0]
                rna_visit_label = "M0 (baseline)"
        else:
            rna_at_visit = rnaseq_analysis
            rna_visit_label = "baseline"

        visit_merged = pd.merge(
            prot_at_visit[["PATNO"] + [p for p in top_32_proteins if p in prot_at_visit.columns]],
            rna_at_visit[[c for c in _rna_merge_cols if c in rna_at_visit.columns]],
            on="PATNO", how="inner",
        )
        if len(visit_merged) < 10:
            continue

        for protein_id in top_32_proteins[:10]:
            ensg_matches = uniprot_to_ensg.get(protein_id, [])
            if not ensg_matches or protein_id not in visit_merged.columns:
                continue
            ensg_col = next((e for e in ensg_matches if e in visit_merged.columns), None)
            if not ensg_col:
                continue

            r_sp, p_sp, n_used, _ = filtered_correlation(
                visit_merged[protein_id], visit_merged[ensg_col])
            visit_corr_results.append({
                "Protein_UniProt": protein_id,
                "Gene_Symbol": uniprot_to_symbol.get(protein_id, "?"),
                "Prot_Visit": f"M{int(visit_m)}", "RNA_Visit": rna_visit_label,
                "RNA_ENSG": ensg_col, "n_pairs": n_used,
                "Spearman_r": r_sp, "Spearman_p": p_sp,
                "Correlation_Type": "concurrent",
            })

    df_corr_visits = pd.DataFrame(visit_corr_results)
    if len(df_corr_visits) > 0:
        print(f"  Visit-stratified correlations: {len(df_corr_visits)} pairs")
    return df_corr_visits


def run_lagged_correlations(top_32_proteins, df_prot_visit, rnaseq_analysis,
                            uniprot_to_ensg, uniprot_to_symbol, rna_has_visit):
    """Compute lagged mRNA-protein correlations.

    Returns df_corr_lagged DataFrame.
    """
    print("\n--- Part 3: Lagged Correlations ---")
    _rna_ensg_needed = set()
    for _p in top_32_proteins[:10]:
        _rna_ensg_needed.update(uniprot_to_ensg.get(_p, []))
    _rna_merge_cols = ["PATNO"] + [c for c in _rna_ensg_needed if c in rnaseq_analysis.columns]

    available_visits = sorted(df_prot_visit["visit_month"].dropna().unique())

    lag_pairs = []
    for v1 in available_visits:
        for v2 in available_visits:
            if v1 != v2:
                lag_pairs.append((v1, v2, f"Prot_M{int(v1)}_vs_RNA_M{int(v2)}"))
    if rna_has_visit:
        rna_visits = sorted(rnaseq_analysis["visit_month_rna"].dropna().unique())
        existing = {(p[0], p[1]) for p in lag_pairs}
        for pv in available_visits:
            for rv in rna_visits:
                if pv != rv and (pv, rv) not in existing:
                    lag_pairs.append((pv, rv, f"Prot_M{int(pv)}_vs_RNA_M{int(rv)}"))

    lagged_results = []
    for prot_visit, rna_visit, lag_label in lag_pairs:
        prot_slice = df_prot_visit[df_prot_visit["visit_month"] == prot_visit]
        if rna_has_visit:
            rna_slice = rnaseq_analysis[rnaseq_analysis["visit_month_rna"] == rna_visit]
        else:
            if rna_visit != 0:
                continue
            rna_slice = rnaseq_analysis

        lag_merged = pd.merge(
            prot_slice[["PATNO"] + [p for p in top_32_proteins if p in prot_slice.columns]],
            rna_slice[[c for c in _rna_merge_cols if c in rna_slice.columns]],
            on="PATNO", how="inner",
        )
        if len(lag_merged) < 10:
            continue

        for protein_id in top_32_proteins[:10]:
            ensg_matches = uniprot_to_ensg.get(protein_id, [])
            if not ensg_matches or protein_id not in lag_merged.columns:
                continue
            ensg_col = next((e for e in ensg_matches if e in lag_merged.columns), None)
            if not ensg_col:
                continue

            r_sp, p_sp, n_used, _ = filtered_correlation(
                lag_merged[protein_id], lag_merged[ensg_col])
            if n_used >= 10:
                lag_months = int(rna_visit - prot_visit)
                lagged_results.append({
                    "Protein_UniProt": protein_id,
                    "Gene_Symbol": uniprot_to_symbol.get(protein_id, "?"),
                    "Prot_Visit": f"M{int(prot_visit)}", "RNA_Visit": f"M{int(rna_visit)}",
                    "Lag_Months": lag_months, "RNA_ENSG": ensg_col,
                    "n_pairs": n_used, "Spearman_r": r_sp, "Spearman_p": p_sp,
                    "Direction": "Protein->RNA" if lag_months > 0 else "RNA->Protein",
                })

    df_corr_lagged = pd.DataFrame(lagged_results)
    if len(df_corr_lagged) > 0:
        print(f"  Lagged correlations: {len(df_corr_lagged)} pairs")
    return df_corr_lagged


# ── Monotonicity assessment ──

def _dcov_inner(X, Y):
    """Compute distance covariance components."""
    n = len(X)
    a = np.abs(X[:, None] - X[None, :])
    b = np.abs(Y[:, None] - Y[None, :])
    A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()
    dcov2 = (A * B).mean()
    dvar_x = (A * A).mean()
    dvar_y = (B * B).mean()
    return dcov2, dvar_x, dvar_y


def distance_correlation(x, y):
    """Compute distance correlation (Szekely, 2007). Range [0,1]."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    dcov2, dvx, dvy = _dcov_inner(x, y)
    if dvx <= 0 or dvy <= 0:
        return 0.0
    return np.sqrt(max(dcov2, 0)) / np.sqrt(np.sqrt(max(dvx, 0)) * np.sqrt(max(dvy, 0)))


def runs_test_monotonicity(x, y, n_min=15):
    """Test for non-monotonic departures from a monotone relationship.

    Returns (z_stat, p_value, n_runs, expected_runs).
    """
    from scipy.stats import norm

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < n_min:
        return np.nan, np.nan, np.nan, np.nan

    order = np.argsort(x)
    y_sorted = y[order]
    x_sorted = x[order]

    rho, _ = spearmanr(x, y)
    increasing = rho >= 0

    iso = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
    y_pred = iso.fit_transform(x_sorted, y_sorted)
    residuals = y_sorted - y_pred

    signs = np.sign(residuals)
    signs = signs[signs != 0]

    if len(signs) < 10:
        return np.nan, np.nan, np.nan, np.nan

    n_runs = 1 + np.sum(signs[1:] != signs[:-1])
    n_pos = np.sum(signs > 0)
    n_neg = np.sum(signs < 0)
    n = n_pos + n_neg

    if n_pos == 0 or n_neg == 0:
        return np.nan, np.nan, n_runs, np.nan

    expected = 1 + (2 * n_pos * n_neg) / n
    var_runs = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n)) / (n ** 2 * (n - 1))

    if var_runs <= 0:
        return np.nan, np.nan, n_runs, expected

    z = (n_runs - expected) / np.sqrt(var_runs)
    p = 2 * norm.sf(np.abs(z))
    return z, p, n_runs, expected


def run_monotonicity_assessment(df_corr_baseline, merged_analysis, output_dir=OUTPUT_DIR):
    """Run monotonicity assessment on baseline correlations.

    Returns df_mono DataFrame.
    """
    print("=== Monotonicity Assessment of mRNA-Protein Correlations ===\n")

    mono_results = []
    valid_pairs = df_corr_baseline.dropna(subset=["Spearman_r"])

    for _, row in valid_pairs.iterrows():
        protein_id = row["Protein_UniProt"]
        ensg_col = row["RNA_ENSG"]
        gene = row["Gene_Symbol"]

        if ensg_col == "NO MATCH":
            continue
        if protein_id not in merged_analysis.columns or ensg_col not in merged_analysis.columns:
            continue

        pair = merged_analysis[[protein_id, ensg_col]].dropna()
        if len(pair) < 15:
            continue
        prot_t = np.percentile(pair[protein_id], 1)
        rna_t = np.percentile(pair[ensg_col], 1)
        pair = pair[(pair[protein_id] > prot_t) & (pair[ensg_col] > max(rna_t, 0))]
        if len(pair) < 15:
            continue

        prot_vals = pair[protein_id].values
        rna_vals = pair[ensg_col].values

        rho_sp, p_sp = spearmanr(prot_vals, rna_vals)

        if len(prot_vals) > 500:
            idx = np.random.RandomState(42).choice(len(prot_vals), 500, replace=False)
            dcor = distance_correlation(prot_vals[idx], rna_vals[idx])
        else:
            dcor = distance_correlation(prot_vals, rna_vals)

        z_runs, p_runs, n_runs, exp_runs = runs_test_monotonicity(prot_vals, rna_vals)

        mono_ratio = abs(rho_sp) / dcor if dcor > 0.01 else np.nan

        if dcor < 0.05 and abs(rho_sp) < 0.05:
            mono_class = "No dependence"
        elif mono_ratio is np.nan or np.isnan(mono_ratio):
            mono_class = "Indeterminate"
        elif mono_ratio > 0.85:
            mono_class = "Monotonic"
        elif mono_ratio > 0.60:
            mono_class = "Mostly monotonic"
        else:
            mono_class = "Non-monotonic component"

        if not np.isnan(p_runs) and p_runs < 0.05 and mono_class in ("Monotonic", "Mostly monotonic"):
            mono_class += " (runs test sig.)"

        mono_results.append({
            "Protein_UniProt": protein_id, "Gene_Symbol": gene,
            "RNA_ENSG": ensg_col, "n_pairs": len(pair),
            "Spearman_rho": rho_sp, "Spearman_p": p_sp,
            "Distance_Corr": dcor, "Mono_Ratio": mono_ratio,
            "Runs_Z": z_runs, "Runs_p": p_runs,
            "Monotonicity": mono_class,
        })

    df_mono = pd.DataFrame(mono_results)

    if len(df_mono) > 0:
        print(f"Pairs assessed: {len(df_mono)}")
        for cat, count in df_mono["Monotonicity"].value_counts().items():
            print(f"  {cat}: {count}")

        df_mono.to_csv(f"{output_dir}/mrna_protein_monotonicity_assessment.csv", index=False)

    return df_mono


def plot_correlation_figures(df_corr_baseline, df_corr_visits, df_corr_lagged,
                            df_mono, merged_analysis, output_dir=OUTPUT_DIR):
    """Generate all correlation plots: baseline, visit heatmap, lagged, monotonicity."""
    valid_bl = df_corr_baseline.dropna(subset=["Spearman_r"])

    if len(valid_bl) > 0:
        # Baseline distribution + per-marker bar chart
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(valid_bl["Spearman_r"], bins=20, color="steelblue",
                     edgecolor="black", alpha=0.8)
        axes[0].axvline(valid_bl["Spearman_r"].median(), color="red", linestyle="--",
                        label=f'Median={valid_bl["Spearman_r"].median():.3f}')
        axes[0].set_xlabel("Spearman r")
        axes[0].set_ylabel("Count")
        axes[0].set_title("mRNA-Protein Correlation Distribution\n(Top 32, expression-filtered)")
        axes[0].legend()

        sorted_bl = valid_bl.sort_values("Spearman_r", ascending=True)
        labels = sorted_bl["Gene_Symbol"] + " (" + sorted_bl["Protein_UniProt"] + ")"
        colors = ["green" if abs(r) > 0.3 else "orange" if abs(r) > 0.15 else "red"
                  for r in sorted_bl["Spearman_r"]]
        axes[1].barh(range(len(sorted_bl)), sorted_bl["Spearman_r"],
                     color=colors, edgecolor="black")
        axes[1].set_yticks(range(len(sorted_bl)))
        axes[1].set_yticklabels(labels, fontsize=7)
        axes[1].set_xlabel("Spearman r")
        axes[1].set_title("Per-Marker mRNA-Protein Correlation")
        axes[1].axvline(0, color="gray", linestyle="-", alpha=0.5)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/mrna_protein_correlation_baseline.eps",
                    format="eps", dpi=300)
        plt.savefig(f"{output_dir}/mrna_protein_correlation_baseline.png", dpi=300)
        plt.show()

    # Visit heatmap
    if len(df_corr_visits) > 0:
        pivot_visit = df_corr_visits.pivot_table(
            index="Gene_Symbol", columns="Prot_Visit", values="Spearman_r", aggfunc="mean")
        if pivot_visit.shape[0] > 0 and pivot_visit.shape[1] > 1:
            fig, ax = plt.subplots(figsize=(10, max(6, len(pivot_visit) * 0.4)))
            sns.heatmap(pivot_visit, cmap="RdBu_r", center=0, vmin=-0.6, vmax=0.6,
                        annot=True, fmt=".2f", linewidths=0.5, ax=ax)
            ax.set_title("mRNA-Protein Correlation by Visit")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/mrna_protein_correlation_visit_heatmap.eps",
                        format="eps", dpi=300)
            plt.savefig(f"{output_dir}/mrna_protein_correlation_visit_heatmap.png", dpi=300)
            plt.show()

    # Lagged correlation figures
    if len(df_corr_lagged) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for direction, marker, color in [("Protein->RNA", "o", "#2196F3"),
                                         ("RNA->Protein", "s", "#FF9800")]:
            sub = df_corr_lagged[df_corr_lagged["Direction"] == direction]
            if len(sub) > 0:
                axes[0].scatter(sub["Lag_Months"], sub["Spearman_r"],
                                alpha=0.5, marker=marker, color=color,
                                label=direction, s=40)
        axes[0].axhline(0, color="gray", linestyle="--", alpha=0.5)
        axes[0].set_xlabel("Lag (months)")
        axes[0].set_ylabel("Spearman r")
        axes[0].set_title("Lagged mRNA-Protein Correlations")
        axes[0].legend()

        plt.tight_layout()
        plt.savefig(f"{output_dir}/mrna_protein_correlation_lagged.eps",
                    format="eps", dpi=300)
        plt.savefig(f"{output_dir}/mrna_protein_correlation_lagged.png", dpi=300)
        plt.show()

    # Monotonicity diagnostics
    if df_mono is not None and len(df_mono) > 0:
        df_sorted = df_mono.dropna(subset=["Mono_Ratio"]).sort_values(
            "Mono_Ratio", ascending=True)
        if len(df_sorted) > 0:
            fig, ax = plt.subplots(figsize=(10, max(4, len(df_sorted) * 0.35)))
            colors = ["#4CAF50" if r > 0.85 else "#FF9800" if r > 0.6 else "#F44336"
                      for r in df_sorted["Mono_Ratio"]]
            ax.barh(
                df_sorted["Gene_Symbol"] + " (" + df_sorted["Protein_UniProt"] + ")",
                df_sorted["Mono_Ratio"], color=colors, edgecolor="black", alpha=0.85)
            ax.axvline(0.85, color="green", linestyle="--", alpha=0.5)
            ax.axvline(0.60, color="orange", linestyle="--", alpha=0.5)
            ax.set_xlabel("Monotonicity Ratio (|Spearman rho| / Distance Correlation)")
            ax.set_title("mRNA-Protein Monotonicity Ratio per Marker")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/mrna_protein_monotonicity_ratio.eps",
                        format="eps", dpi=300)
            plt.savefig(f"{output_dir}/mrna_protein_monotonicity_ratio.png", dpi=300)
            plt.show()

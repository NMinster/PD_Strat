"""
PD-Deep Precision Suite — Reviewer Revision Pipeline (v5)

Main orchestration script that runs the full analysis pipeline:
  1. Data loading & QC
  2. Gene/protein ID mapping
  3. ComBat batch correction
  4. RNA-seq pipeline comparison
  5. mRNA-protein correlations
  6. Model training (unimodal + multimodal fusion)
  7. Evaluation (metrics, calibration, ROC/PR, DeLong)
  8. Interpretability (feature stability, SHAP)
  9. Prognostic validation (PSI_resid, Cox PH, KM)
 10. SHAP beeswarm plots
 11. Output summary

Usage
-----
    python -m pd_strat.main
"""

import warnings
warnings.filterwarnings("ignore")

import gc
import os
import glob
import numpy as np
import pandas as pd

from .config import (
    RANDOM_STATE, OUTPUT_DIR, PROTEOMICS_PANELS, RNASEQ_PATH, DEG_PATH,
    PD_GENE_SETS_SYMBOLS, N_CV_SPLITS,
)
from .preprocessing import (
    FoldAwarePreprocessorSimple, FoldAwareCombatPreprocessor, NaNSafeScaler,
)
from .data_loading import (
    load_proteomics, load_rnaseq, load_deg, merge_datasets,
    run_qc_diagnostics, run_batch_assessment, make_analysis_copies,
    build_datasets, create_splits, build_visit_level_proteomics,
)
from .id_mapping import (
    build_ensg_to_symbol, build_uniprot_to_ensg, patch_unmapped_via_mygene,
)
from .combat import (
    apply_combat_to_splits, run_post_combat_diagnostics,
)
from .utils import generate_lhs_params
from .rna_pipelines import (
    make_rna_pipelines, build_gene_sets_ensg, run_rna_pipeline_comparison,
)
from .correlation_analysis import (
    run_baseline_correlations, run_visit_stratified_correlations,
    run_lagged_correlations, run_monotonicity_assessment,
    plot_correlation_figures,
)
from .model_training import run_all_training
from .evaluation import (
    build_metrics_table, run_threshold_optimisation,
    plot_calibration_curves, plot_roc_prc_curves, run_delong_comparisons,
)
from .interpretability import run_feature_selection_stability
from .prognostic import run_prognostic_validation
from .visualization import plot_shap_beeswarms


def main():
    """Run the full PD-Deep Precision Suite pipeline."""
    print("=" * 60)
    print("PD-Deep Precision Suite — Reviewer Revision Pipeline v5")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}/\n")

    # ================================================================
    # 1. Data Loading & QC
    # ================================================================
    print("\n" + "=" * 60)
    print("1. DATA LOADING & PREPROCESSING")
    print("=" * 60)

    # 1A. Load proteomics
    df_proteomics_long, df_proteomics_wide = load_proteomics(PROTEOMICS_PANELS)

    # 1B. Load RNA-seq & DEG
    rnaseq_data = load_rnaseq(RNASEQ_PATH)
    rna_feature_cols = [c for c in rnaseq_data.columns
                        if c not in ["PATNO", "pd"]]
    print(f"RNA feature columns (first 5): {rna_feature_cols[:5]}")
    print(f"Total RNA features: {len(rna_feature_cols)}")

    df_deg, deg_gene_ids = load_deg(DEG_PATH)

    # Merge
    merged_data = merge_datasets(rnaseq_data, df_proteomics_wide)

    # 1B-QC. Quality control
    RNA_NEEDS_LOG = run_qc_diagnostics(
        df_proteomics_wide, rnaseq_data, rna_feature_cols)

    # 1B-QC-3. Batch assessment
    run_batch_assessment(df_proteomics_wide, rnaseq_data,
                         df_proteomics_long, rna_feature_cols)

    # Analysis copies
    rnaseq_analysis, merged_analysis = make_analysis_copies(
        rnaseq_data, merged_data, rna_feature_cols, RNA_NEEDS_LOG)

    # 1F-extra. Visit-level proteomics
    df_prot_visit = build_visit_level_proteomics(df_proteomics_long)

    # 1E. Column groups & datasets
    columns_rna = [c for c in merged_data.columns if c in deg_gene_ids]
    proteomics_cols = [
        c for c in merged_data.columns
        if c not in ["PATNO", "pd"] and c not in deg_gene_ids
    ]
    proteomics_cols_only = [
        c for c in merged_data.columns
        if c in df_proteomics_wide.columns and c not in ["PATNO", "pd"]
    ]

    datasets = build_datasets(merged_data, deg_gene_ids,
                              df_proteomics_wide, rna_feature_cols)

    # 1E-post. Preprocessor configs
    preprocessor_configs = {}
    combat_preprocessors = {}

    for name in datasets:
        if name == "Proteomics only":
            preprocessor_configs[name] = FoldAwarePreprocessorSimple(
                apply_log2=False, rna_col_indices=None)
            combat_preprocessors[name] = FoldAwareCombatPreprocessor(
                apply_log2=False, rna_col_indices=None)
        elif name == "RNA-seq only":
            preprocessor_configs[name] = FoldAwarePreprocessorSimple(
                apply_log2=RNA_NEEDS_LOG, rna_col_indices=None)
            combat_preprocessors[name] = FoldAwareCombatPreprocessor(
                apply_log2=RNA_NEEDS_LOG, rna_col_indices=None)
        elif name == "Combined":
            if RNA_NEEDS_LOG:
                all_cols = datasets[name].drop(
                    columns=["PATNO", "pd"]).columns.tolist()
                rna_indices = [i for i, c in enumerate(all_cols)
                               if c in set(rna_feature_cols)]
                preprocessor_configs[name] = FoldAwarePreprocessorSimple(
                    apply_log2=True, rna_col_indices=rna_indices)
            else:
                preprocessor_configs[name] = FoldAwarePreprocessorSimple(
                    apply_log2=False, rna_col_indices=None)

    # Train/test/val splits
    splits = create_splits(datasets)

    # ComBat batch correction
    print("\n--- Applying ComBat batch correction ---")
    splits_uncorrected = {name: splits[name].copy() for name in splits}
    for name in datasets:
        print(f"\n  {name}:")
        splits[name] = apply_combat_to_splits(
            splits[name], name, datasets,
            columns_rna=columns_rna,
            rna_feature_cols=rna_feature_cols,
            RNA_NEEDS_LOG=RNA_NEEDS_LOG,
            combat_preprocessors=combat_preprocessors,
        )

    run_post_combat_diagnostics(splits, splits_uncorrected, datasets)

    # Update preprocessors with NaNSafeScaler post-ComBat
    for name in preprocessor_configs:
        preprocessor_configs[name] = NaNSafeScaler()

    gc.collect()

    # ================================================================
    # 1C-1D. Gene/Protein ID Mapping
    # ================================================================
    print("\n" + "=" * 60)
    print("1C-D. GENE / PROTEIN ID MAPPING")
    print("=" * 60)

    ensg_to_symbol, symbol_to_ensg, all_mappings = build_ensg_to_symbol(
        df_deg, rna_feature_cols)

    prot_feat_cols = [c for c in df_proteomics_wide.columns
                      if c not in ["PATNO"]]
    uniprot_ids = prot_feat_cols

    uniprot_to_symbol, uniprot_to_ensg = build_uniprot_to_ensg(
        uniprot_ids, symbol_to_ensg, rna_feature_cols,
        all_mappings=all_mappings)

    uniprot_to_symbol, uniprot_to_ensg = patch_unmapped_via_mygene(
        uniprot_ids, uniprot_to_symbol, uniprot_to_ensg,
        symbol_to_ensg, rna_feature_cols)

    # ================================================================
    # 2. Utility Setup (LHS hyperparameters)
    # ================================================================
    print("\n" + "=" * 60)
    print("2. HYPERPARAMETER SETUP")
    print("=" * 60)

    lhs_param_list = generate_lhs_params()
    lhs_param_list_combined = generate_lhs_params(
        max_estimators=800, d=6, seed=RANDOM_STATE + 1)

    # ================================================================
    # 3. RNA-seq Pipeline Comparison
    # ================================================================
    print("\n" + "=" * 60)
    print("3. RNA-SEQ PIPELINE COMPARISON")
    print("=" * 60)

    rna_preprocessor = preprocessor_configs["RNA-seq only"]
    rna_pipelines = make_rna_pipelines(rna_preprocessor, n_components=50)

    gene_sets_ensg = build_gene_sets_ensg(
        PD_GENE_SETS_SYMBOLS, symbol_to_ensg, rna_feature_cols)

    rna_results = run_rna_pipeline_comparison(
        rna_pipelines, splits["RNA-seq only"], gene_sets_ensg,
        rna_preprocessor)

    gc.collect()

    # ================================================================
    # 4. mRNA-Protein Correlations
    # ================================================================
    print("\n" + "=" * 60)
    print("4. mRNA-PROTEIN CORRELATIONS")
    print("=" * 60)

    # Determine top 32 proteins from SHAP (placeholder — filled after training)
    top_32_proteins = proteomics_cols_only[:32]  # initial placeholder

    df_corr_baseline = run_baseline_correlations(
        top_32_proteins, merged_analysis, uniprot_to_ensg,
        rna_feature_cols, ensg_to_symbol)

    df_corr_visits = run_visit_stratified_correlations(
        top_32_proteins, df_prot_visit, rnaseq_analysis,
        uniprot_to_ensg, rna_feature_cols, ensg_to_symbol)

    df_corr_lagged = run_lagged_correlations(
        top_32_proteins, df_prot_visit, rnaseq_analysis,
        uniprot_to_ensg, rna_feature_cols, ensg_to_symbol)

    run_monotonicity_assessment(df_corr_baseline, merged_analysis)

    plot_correlation_figures(df_corr_baseline, df_corr_visits,
                            df_corr_lagged)

    gc.collect()

    # ================================================================
    # 5. Model Training & Evaluation
    # ================================================================
    print("\n" + "=" * 60)
    print("5. MODEL TRAINING & EVALUATION")
    print("=" * 60)

    trained_models, all_probas, all_preds = run_all_training(
        splits, datasets, preprocessor_configs,
        lhs_param_list, lhs_param_list_combined,
        proteomics_cols, columns_rna)

    # 5B. Comprehensive metrics
    df_metrics = build_metrics_table(trained_models, splits,
                                      all_probas, all_preds)

    # 5C. Threshold optimisation
    optimal_thresholds = run_threshold_optimisation(
        trained_models, splits, all_probas)

    # 5D. Calibration curves
    plot_calibration_curves(trained_models, splits, all_probas)

    # 5E. ROC + PR curves
    plot_roc_prc_curves(trained_models, splits, all_probas)

    # 5F. DeLong comparisons
    df_delong = run_delong_comparisons(trained_models, splits, all_probas)

    gc.collect()

    # ================================================================
    # 6. Interpretability & Robustness
    # ================================================================
    print("\n" + "=" * 60)
    print("6. INTERPRETABILITY & ROBUSTNESS DIAGNOSTICS")
    print("=" * 60)

    fold_features, fold_shap, df_freq = run_feature_selection_stability(
        splits, preprocessor_configs, top_32_proteins)

    gc.collect()

    # ================================================================
    # 7. Prognostic Validation
    # ================================================================
    print("\n" + "=" * 60)
    print("7. PROGNOSTIC VALIDATION")
    print("=" * 60)

    try:
        df_prog_summary = run_prognostic_validation(
            trained_models, datasets, splits)
    except FileNotFoundError as e:
        print(f"  Skipping prognostic validation: {e}")

    gc.collect()

    # ================================================================
    # 8. SHAP Beeswarm Plots
    # ================================================================
    print("\n" + "=" * 60)
    print("8. SHAP BEESWARM PLOTS")
    print("=" * 60)

    plot_shap_beeswarms(trained_models, splits)

    gc.collect()

    # ================================================================
    # 9. Output Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("REVISION OUTPUTS SUMMARY")
    print("=" * 60)

    output_files = sorted(glob.glob(f"{OUTPUT_DIR}/*"))
    categories = {
        "Tables (CSV)": [f for f in output_files if f.endswith('.csv')],
        "Figures (EPS)": [f for f in output_files if f.endswith('.eps')],
        "Figures (PNG)": [f for f in output_files if f.endswith('.png')],
        "Text": [f for f in output_files if f.endswith('.txt')],
    }
    for cat, files in categories.items():
        if files:
            print(f"\n{cat}:")
            for f in files:
                print(f"  - {os.path.basename(f)}")

    print("\n" + "=" * 60)
    print("CHECKLIST")
    print("=" * 60)
    for rev, desc in [
        ("R1-1/R2-2", "RNA pipeline comparison table + figure"),
        ("R1-2", "mRNA-protein correlations table + plot"),
        ("R1-minor-1/R2-4", "Comprehensive metrics + calibration + ROC/PRC + DeLong"),
        ("R1-minor-2/R1-4/R2-5", "Feature stability + SHAP stability + correlation heatmap"),
        ("R2-6/R2-8", "Adjusted severity + comparator model + mixed model"),
    ]:
        print(f"  [{rev:20s}] {desc}")

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

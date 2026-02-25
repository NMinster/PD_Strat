"""
SHAP beeswarm plots for all eligible models (Section 8).

Handles both tree-based (Proteomics, Combined → TreeExplainer) and
linear (RNA PCA + LogReg → LinearExplainer) pipelines with appropriate
SHAP explainers.

Functions
---------
plot_shap_beeswarms(trained_models, splits, output_dir)
    Generate beeswarm summary plots for each pipeline-based model.
"""

import numpy as np
import matplotlib.pyplot as plt
import shap

from .config import OUTPUT_DIR


def plot_shap_beeswarms(trained_models, splits, output_dir=OUTPUT_DIR):
    """Generate SHAP beeswarm plots for all eligible models.

    Skips wrapper models (Late Fusion, Stacking) that lack named_steps.
    """
    active = [n for n in trained_models
              if hasattr(trained_models[n], 'named_steps')]

    for name in active:
        s = splits[name]
        best_pipe = trained_models[name]
        model = best_pipe.named_steps["model"]

        print(f"\n--- SHAP: {name} ({type(model).__name__}) ---")

        # Tree-based pipelines (SelectKBest -> RF)
        if hasattr(model, "feature_importances_"):
            X_val_transformed = best_pipe[:-1].transform(s["X_val"])

            vt_mask = best_pipe.named_steps[
                'variance_threshold'].get_support()
            filtered_names = s["X_val"].columns[vt_mask]

            if 'feature_selection' in best_pipe.named_steps:
                kb_mask = best_pipe.named_steps[
                    'feature_selection'].get_support()
                final_names = filtered_names[kb_mask]
            else:
                final_names = filtered_names

            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_val_transformed)

            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_vals[1], X_val_transformed,
                              feature_names=final_names, show=False,
                              max_display=20)
            for collection in plt.gca().collections:
                collection.set_alpha(1.0)

        # Linear pipelines (PCA -> LogReg)
        elif hasattr(model, "coef_"):
            X_val_transformed = best_pipe[:-1].transform(s["X_val"])

            n_components = X_val_transformed.shape[1]
            if 'pca' in best_pipe.named_steps:
                feature_names = [f"PC{i+1}" for i in range(n_components)]
            else:
                feature_names = [f"Feature_{i+1}"
                                 for i in range(n_components)]

            try:
                explainer = shap.LinearExplainer(model, X_val_transformed)
                shap_vals = explainer.shap_values(X_val_transformed)
            except Exception:
                bg = shap.sample(X_val_transformed,
                                 min(100, len(X_val_transformed)))
                explainer = shap.KernelExplainer(model.predict_proba, bg)
                shap_vals = explainer.shap_values(
                    X_val_transformed[:200])[1]

            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_vals, X_val_transformed,
                              feature_names=feature_names, show=False,
                              max_display=20)
            for collection in plt.gca().collections:
                collection.set_alpha(1.0)
        else:
            print(f"  Skipping -- unsupported model type: "
                  f"{type(model).__name__}")
            continue

        plt.title(f"{name} -- SHAP Summary (Beeswarm)")
        plt.tight_layout()
        safe_name = name.replace(' ', '_')
        plt.savefig(f"{output_dir}/{safe_name}_shap_beeswarm.eps",
                    format='eps', dpi=300)
        plt.savefig(f"{output_dir}/{safe_name}_shap_beeswarm.png", dpi=300)
        plt.show()
        print(f"  Saved")

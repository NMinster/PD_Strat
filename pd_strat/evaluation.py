"""
Model evaluation: metrics table, threshold optimisation, calibration, ROC/PR, DeLong.

Sections 5B–5F of the revision pipeline.

Functions
---------
compute_full_metrics(y_true, y_proba, y_pred)
    Compute ROC-AUC, AUPRC, Brier, confusion-matrix metrics, F1.
build_metrics_table(trained_models, splits, all_probas, all_preds, output_dir)
    Comprehensive internal + external metrics for every model.
run_threshold_optimisation(trained_models, splits, all_probas, output_dir)
    Youden's J on PDBP internal test set, applied to external val.
plot_calibration_curves(trained_models, splits, all_probas, output_dir)
    Per-model calibration curves (Brier score annotated).
plot_roc_prc_curves(trained_models, splits, all_probas, output_dir)
    Overlaid ROC + Precision-Recall curves.
run_delong_comparisons(trained_models, splits, all_probas, output_dir)
    Pairwise DeLong ROC-AUC comparisons.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, roc_curve, precision_recall_curve,
    calibration_curve,
)

from .config import OUTPUT_DIR, MODEL_COLORS
from .utils import bootstrap_ci, bootstrap_paired_aucs, delong_test


def compute_full_metrics(y_true, y_proba, y_pred):
    """Compute comprehensive binary classification metrics.

    Returns dict of metric name -> value.
    """
    metrics = {}
    metrics["ROC-AUC"], metrics["AUC_CI_lo"], metrics["AUC_CI_hi"] = \
        bootstrap_ci(y_true, y_proba, roc_auc_score)
    metrics["AUPRC"], metrics["AUPRC_CI_lo"], metrics["AUPRC_CI_hi"] = \
        bootstrap_ci(y_true, y_proba, average_precision_score)
    metrics["Brier"] = brier_score_loss(y_true, y_proba)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics["Sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics["Specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics["PPV"] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics["NPV"] = tn / (tn + fn) if (tn + fn) > 0 else 0
    metrics["F1"] = f1_score(y_true, y_pred)
    return metrics


def build_metrics_table(trained_models, splits, all_probas, all_preds,
                        output_dir=OUTPUT_DIR):
    """Build and save comprehensive metrics table (internal + external).

    Returns DataFrame.
    """
    metrics_rows = []
    for name in trained_models:
        s = splits[name]
        best = trained_models[name]

        # External validation
        ext = compute_full_metrics(s["y_val"], all_probas[name], all_preds[name])
        ext["Model"] = name
        ext["Evaluation"] = "External (PP-/PPMI)"
        metrics_rows.append(ext)

        # Internal test
        y_test_proba = best.predict_proba(s["X_test"])[:, 1]
        y_test_pred = best.predict(s["X_test"])
        int_m = compute_full_metrics(s["y_test"], y_test_proba, y_test_pred)
        int_m["Model"] = name
        int_m["Evaluation"] = "Internal (PD-/PDBP)"
        metrics_rows.append(int_m)

    df = pd.DataFrame(metrics_rows)
    col_order = [
        "Model", "Evaluation", "ROC-AUC", "AUC_CI_lo", "AUC_CI_hi",
        "AUPRC", "AUPRC_CI_lo", "AUPRC_CI_hi", "Brier",
        "Sensitivity", "Specificity", "PPV", "NPV", "F1",
    ]
    df = df[[c for c in col_order if c in df.columns]].round(4)

    print("\n=== Comprehensive Metrics Table ===")
    print(df.to_string())
    df.to_csv(f"{output_dir}/comprehensive_metrics_table.csv", index=False)
    return df


def run_threshold_optimisation(trained_models, splits, all_probas,
                               output_dir=OUTPUT_DIR):
    """Optimise classification threshold via Youden's J on PDBP internal test.

    Returns dict of {model_name: optimal_threshold}.
    """
    print("=== Threshold Optimization (Youden's J, PDBP internal test) ===\n")
    optimal_thresholds = {}

    for name in ["Combined", "Proteomics only", "RNA-seq only"]:
        if name not in trained_models:
            continue
        s = splits[name]
        best = trained_models[name]
        y_test_proba = best.predict_proba(s["X_test"])[:, 1]

        fpr, tpr, thresholds = roc_curve(s["y_test"], y_test_proba)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        optimal_thresh = thresholds[best_idx]
        optimal_thresholds[name] = optimal_thresh

        print(f"{name}:")
        print(f"  Optimal threshold: {optimal_thresh:.4f}")

        y_val_pred_opt = (all_probas[name] >= optimal_thresh).astype(int)
        cm = confusion_matrix(s["y_val"], y_val_pred_opt)
        tn, fp, fn, tp = cm.ravel()
        print(f"  External @ optimal: Sens={tp/(tp+fn):.4f}, "
              f"Spec={tn/(tn+fp):.4f}, PPV={tp/(tp+fp):.4f}, "
              f"NPV={tn/(tn+fn):.4f}\n")

    return optimal_thresholds


def plot_calibration_curves(trained_models, splits, all_probas,
                            output_dir=OUTPUT_DIR):
    """Plot per-model calibration curves with Brier scores."""
    active_models = [n for n in trained_models if n in all_probas]

    _cmap = plt.cm.tab10
    colors = {
        name: MODEL_COLORS.get(name, _cmap(i / max(len(active_models), 1)))
        for i, name in enumerate(active_models)
    }

    fig, axes = plt.subplots(1, len(active_models),
                             figsize=(6 * len(active_models), 5))
    if len(active_models) == 1:
        axes = [axes]

    for idx, name in enumerate(active_models):
        s = splits[name]
        y_proba = all_probas[name]
        brier = brier_score_loss(s["y_val"], y_proba)

        ax = axes[idx]
        prob_true, prob_pred = calibration_curve(
            s["y_val"], y_proba, n_bins=10, strategy='uniform')
        ax.plot(prob_pred, prob_true, 'o-', color=colors[name],
                label=f'{name}\nBrier={brier:.4f}')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"{name}")
        ax.legend(loc='lower right')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    plt.suptitle("Calibration Curves -- External Validation (PPMI)", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/calibration_curves.eps", format='eps', dpi=300)
    plt.savefig(f"{output_dir}/calibration_curves.png", dpi=300)
    plt.show()


def plot_roc_prc_curves(trained_models, splits, all_probas,
                        output_dir=OUTPUT_DIR):
    """Plot overlaid ROC and Precision-Recall curves."""
    active_models = [n for n in trained_models if n in all_probas]

    _cmap = plt.cm.tab10
    colors = {
        name: MODEL_COLORS.get(name, _cmap(i / max(len(active_models), 1)))
        for i, name in enumerate(active_models)
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC curves
    ax = axes[0]
    for name in active_models:
        s = splits[name]
        fpr, tpr, _ = roc_curve(s["y_val"], all_probas[name])
        auc_val = roc_auc_score(s["y_val"], all_probas[name])
        ax.plot(fpr, tpr, color=colors[name], lw=2,
                label=f'{name} (AUC={auc_val:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves -- External Validation (PPMI)")
    ax.legend(loc='lower right')

    # PR curves
    ax = axes[1]
    for name in active_models:
        s = splits[name]
        prec, rec, _ = precision_recall_curve(s["y_val"], all_probas[name])
        ap = average_precision_score(s["y_val"], all_probas[name])
        ax.plot(rec, prec, color=colors[name], lw=2,
                label=f'{name} (AP={ap:.3f})')
    prevalence = splits[active_models[0]]["y_val"].mean()
    ax.axhline(prevalence, color='gray', linestyle='--', alpha=0.5,
               label=f'Baseline (prev={prevalence:.2f})')
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves -- External Validation (PPMI)")
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/roc_prc_curves.eps", format='eps', dpi=300)
    plt.savefig(f"{output_dir}/roc_prc_curves.png", dpi=300)
    plt.show()


def run_delong_comparisons(trained_models, splits, all_probas,
                           output_dir=OUTPUT_DIR):
    """Run pairwise DeLong ROC-AUC comparisons between all models.

    Returns DataFrame of comparison results.
    """
    print("=== DeLong Test: Pairwise ROC-AUC Comparisons ===\n")

    active_models = [n for n in trained_models if n in all_probas]
    model_pairs = list(combinations(active_models, 2))
    delong_rows = []

    for name_a, name_b in model_pairs:
        s_a, s_b = splits[name_a], splits[name_b]

        if np.array_equal(np.asarray(s_a["y_val"]),
                          np.asarray(s_b["y_val"])):
            auc_a, auc_b, z, p = delong_test(
                np.asarray(s_a["y_val"]),
                all_probas[name_a], all_probas[name_b])
            sig = "Yes" if p < 0.05 else "No"
            print(f"{name_a} vs {name_b}: AUC={auc_a:.4f} vs {auc_b:.4f}, "
                  f"Z={z:.3f}, p={p:.5f} [{sig}]")
            delong_rows.append({
                "Model A": name_a, "Model B": name_b,
                "AUC_A": auc_a, "AUC_B": auc_b,
                "Z_stat": z, "p_value": p, "Significant": sig,
            })
        else:
            print(f"{name_a} vs {name_b}: Different val sets -- bootstrap")
            diffs = bootstrap_paired_aucs(
                s_a["y_val"], all_probas[name_a],
                y_true_b=s_b["y_val"], proba_b=all_probas[name_b])
            p_boot = np.mean(diffs < 0) * 2
            print(f"  Bootstrap p={p_boot:.5f}")

    df = pd.DataFrame(delong_rows).round(5)
    print(df.to_string())
    df.to_csv(f"{output_dir}/delong_roc_comparisons.csv", index=False)
    return df

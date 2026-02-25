"""
RNA-seq pipeline variants and Gene-Set PCA.

Defines multiple RNA-seq classification pipelines for fair comparison:
  1. Baseline RF (VarianceThreshold -> SelectKBest -> Scaler -> RF)
  2. PCA + LogReg (L1)
  3. PCA + RF
  4. ICA + LogReg (L1)
  5. NMF + RF
  6. Lasso FS + LogReg
  7. Lasso FS + RF
  8. Gene-Set PCA + RF (pathway aggregation using ENSG->Symbol mapping)
"""

import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import (
    SelectKBest, f_classif, VarianceThreshold, SelectFromModel,
)
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline as SKPipeline
from imblearn.pipeline import Pipeline as IMBPipeline

from .config import RANDOM_STATE, N_CV_SPLITS, N_JOBS_SAFE, OUTPUT_DIR
from .preprocessing import ClipNonNegative, GeneSetPCA
from .utils import bootstrap_ci


def make_rna_pipelines(preprocessor, n_components=50):
    """Construct dict of RNA-seq pipeline variants.

    Parameters
    ----------
    preprocessor : sklearn transformer
        Fold-aware preprocessor for RNA-seq data.
    n_components : int
        Number of PCA/ICA/NMF components.

    Returns dict of {pipeline_name: pipeline}.
    """
    pipelines = {}

    pipelines["Baseline RF"] = IMBPipeline([
        ("variance_threshold", VarianceThreshold(threshold=0.0)),
        ("feature_selection", SelectKBest(score_func=f_classif, k="all")),
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=1000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])

    pipelines["PCA + LogReg (L1)"] = SKPipeline([
        ("variance_threshold", VarianceThreshold(threshold=0.0)),
        ("preprocessor", preprocessor),
        ("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)),
        ("model", LogisticRegression(
            penalty="l1", solver="saga", C=1.0, max_iter=5000,
            class_weight="balanced", random_state=RANDOM_STATE)),
    ])

    pipelines["PCA + RF"] = SKPipeline([
        ("variance_threshold", VarianceThreshold(threshold=0.0)),
        ("preprocessor", preprocessor),
        ("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)),
        ("model", RandomForestClassifier(
            n_estimators=1000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])

    pipelines["ICA + LogReg (L1)"] = SKPipeline([
        ("variance_threshold", VarianceThreshold(threshold=0.0)),
        ("preprocessor", preprocessor),
        ("ica", FastICA(n_components=min(n_components, 30),
                        random_state=RANDOM_STATE, max_iter=1000)),
        ("model", LogisticRegression(
            penalty="l1", solver="saga", C=1.0, max_iter=5000,
            class_weight="balanced", random_state=RANDOM_STATE)),
    ])

    pipelines["NMF + RF"] = SKPipeline([
        ("variance_threshold", VarianceThreshold(threshold=0.0)),
        ("clip", ClipNonNegative()),
        ("nmf", NMF(n_components=min(n_components, 30), init="nndsvda",
                     random_state=RANDOM_STATE, max_iter=500)),
        ("model", RandomForestClassifier(
            n_estimators=1000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])

    lasso_selector = SelectFromModel(
        LogisticRegression(
            penalty="l1", solver="saga", C=0.1, max_iter=5000,
            class_weight="balanced", random_state=RANDOM_STATE,
        )
    )

    pipelines["Lasso FS + LogReg"] = SKPipeline([
        ("variance_threshold", VarianceThreshold(threshold=0.0)),
        ("preprocessor", preprocessor),
        ("lasso_select", lasso_selector),
        ("model", LogisticRegression(
            penalty="l2", solver="lbfgs", C=1.0, max_iter=5000,
            class_weight="balanced", random_state=RANDOM_STATE)),
    ])

    pipelines["Lasso FS + RF"] = SKPipeline([
        ("variance_threshold", VarianceThreshold(threshold=0.0)),
        ("preprocessor", preprocessor),
        ("lasso_select", SelectFromModel(
            LogisticRegression(
                penalty="l1", solver="saga", C=0.1, max_iter=5000,
                class_weight="balanced", random_state=RANDOM_STATE,
            )
        )),
        ("model", RandomForestClassifier(
            n_estimators=1000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])

    return pipelines


def build_gene_sets_ensg(pd_gene_sets_symbols, symbol_to_ensg, rna_feature_cols):
    """Convert symbol-based gene sets to ENSG-based gene sets.

    Returns PD_GENE_SETS_ENSG dict.
    """
    rna_cols_set = set(rna_feature_cols)
    PD_GENE_SETS_ENSG = {}

    for gs_name, symbols in pd_gene_sets_symbols.items():
        ensg_members = []
        matched_symbols = []
        for sym in symbols:
            if sym in symbol_to_ensg:
                for ensg_id in symbol_to_ensg[sym]:
                    if ensg_id in rna_cols_set:
                        ensg_members.append(ensg_id)
                        matched_symbols.append(sym)
                        break
        PD_GENE_SETS_ENSG[gs_name] = ensg_members
        print(f"  {gs_name}: {len(ensg_members)}/{len(symbols)} genes mapped")

    total_mapped = sum(len(v) for v in PD_GENE_SETS_ENSG.values())
    print(f"\nTotal gene-set members mapped to ENSG: {total_mapped}")
    return PD_GENE_SETS_ENSG


def run_rna_pipeline_comparison(rna_pipelines, rna_split, gene_sets_ensg,
                                preprocessor, output_dir=OUTPUT_DIR):
    """Run all RNA-seq pipeline comparisons and produce summary table + figure.

    Returns rna_results dict.
    """
    group_kfold = GroupKFold(n_splits=N_CV_SPLITS)
    rna_results = {}

    for pipe_name, pipe in rna_pipelines.items():
        print(f"\n--- {pipe_name} ---")
        try:
            cv_aucs = cross_val_predict(
                pipe, rna_split["X_train"], rna_split["y_train"],
                groups=rna_split["groups_train"],
                cv=group_kfold, method="predict_proba", n_jobs=N_JOBS_SAFE,
            )
            pipe.fit(rna_split["X_train"], rna_split["y_train"])

            cv_proba = cv_aucs[:, 1]
            cv_auc = roc_auc_score(rna_split["y_train"], cv_proba)
            val_proba = pipe.predict_proba(rna_split["X_val"])[:, 1]
            val_auc = roc_auc_score(rna_split["y_val"], val_proba)
            val_auprc = average_precision_score(rna_split["y_val"], val_proba)
            _, ci_lo, ci_hi = bootstrap_ci(rna_split["y_val"], val_proba, roc_auc_score)

            rna_results[pipe_name] = {
                "cv_auc": cv_auc, "val_auc": val_auc, "val_auprc": val_auprc,
                "ci_lo": ci_lo, "ci_hi": ci_hi, "val_proba": val_proba,
            }
            print(f"  CV AUC={cv_auc:.4f}  |  Val AUC={val_auc:.4f} "
                  f"[{ci_lo:.3f}, {ci_hi:.3f}]  |  AUPRC={val_auprc:.4f}")
        except Exception as e:
            print(f"  FAILED: {e}")
            rna_results[pipe_name] = None
        gc.collect()

    # Gene-set PCA pipeline
    print("\n--- Gene-Set PCA + RF ---")
    try:
        gs_pipe = SKPipeline([
            ("variance_threshold", VarianceThreshold(threshold=0.0)),
            ("preprocessor", preprocessor),
            ("geneset_pca", GeneSetPCA(gene_sets_ensg, n_components_per_set=3)),
            ("model", RandomForestClassifier(
                n_estimators=1000, class_weight="balanced", random_state=RANDOM_STATE)),
        ])
        gs_pipe.fit(rna_split["X_train"], rna_split["y_train"])
        gs_val_proba = gs_pipe.predict_proba(rna_split["X_val"])[:, 1]
        gs_val_auc = roc_auc_score(rna_split["y_val"], gs_val_proba)
        gs_val_auprc = average_precision_score(rna_split["y_val"], gs_val_proba)
        _, gs_ci_lo, gs_ci_hi = bootstrap_ci(
            rna_split["y_val"], gs_val_proba, roc_auc_score)

        rna_results["Gene-Set PCA + RF"] = {
            "cv_auc": np.nan, "val_auc": gs_val_auc, "val_auprc": gs_val_auprc,
            "ci_lo": gs_ci_lo, "ci_hi": gs_ci_hi, "val_proba": gs_val_proba,
        }
        print(f"  Val AUC={gs_val_auc:.4f} [{gs_ci_lo:.3f}, {gs_ci_hi:.3f}]  "
              f"|  AUPRC={gs_val_auprc:.4f}")
    except Exception as e:
        print(f"  FAILED: {e}")

    # Summary table & figure
    rna_summary = pd.DataFrame({
        name: {k: v for k, v in r.items() if k != "val_proba"}
        for name, r in rna_results.items() if r is not None
    }).T.round(4)

    print("\n=== RNA-seq Pipeline Comparison ===")
    print(rna_summary.to_string())
    rna_summary.to_csv(f"{output_dir}/supp_rna_pipeline_comparison.csv")

    fig, ax = plt.subplots(figsize=(10, 5))
    names = rna_summary.index.tolist()
    aucs = rna_summary["val_auc"].values
    ci_lo = rna_summary["ci_lo"].values
    ci_hi = rna_summary["ci_hi"].values
    yerr = np.array([aucs - ci_lo, ci_hi - aucs])

    ax.barh(names, aucs, xerr=yerr,
            color=sns.color_palette("muted", len(names)),
            edgecolor="black", capsize=4)
    ax.set_xlabel("Validation ROC-AUC")
    ax.set_title("RNA-seq Pipeline Comparison (External Validation)")
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_xlim(0.3, 1.0)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/supp_rna_pipeline_comparison.eps", format="eps", dpi=300)
    plt.savefig(f"{output_dir}/supp_rna_pipeline_comparison.png", dpi=300)
    plt.show()

    return rna_results

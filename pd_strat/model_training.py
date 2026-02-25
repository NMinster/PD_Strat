"""
Model training: unimodal classifiers + multimodal fusion strategies.

Phase 1 — Unimodal:
  - Proteomics only: RF + RandomizedSearchCV (LHS hyperparameters)
  - RNA-seq only: PCA(50) + LogReg(L1) (best from RNA pipeline comparison)

Phase 2 — Multimodal Fusion (on Combined split):
  1. Early Fusion (naive concat → SelectKBest → RF)
  2. Balanced Early Fusion (per-modality PCA → LogReg)
  3. Late Fusion (avg): mean of unimodal probabilities
  4. Late Fusion (weighted): optimised weights on internal test set
  5. Stacking: LogReg meta-learner on OOF unimodal predictions

Functions
---------
train_proteomics_model(...)
train_rna_model(...)
train_early_fusion(...)
train_balanced_early_fusion(...)
train_late_fusion(...)
train_stacking(...)
run_all_training(...)
"""

import gc
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV, GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline as SKPipeline
from imblearn.pipeline import Pipeline as IMBPipeline

from .config import RANDOM_STATE, N_CV_SPLITS, N_JOBS_SAFE
from .preprocessing import (
    ModalityBalancedReducerColNames, LateFusionWrapper, StackingWrapper,
)


def train_proteomics_model(splits, preprocessor_configs, lhs_param_list):
    """Train proteomics-only RF with RandomizedSearchCV.

    Returns (model, probas_val, preds_val).
    """
    s = splits["Proteomics only"]
    n_features = s["X_train"].shape[1]
    k_val = min(41, n_features)

    pipe = IMBPipeline([
        ('preprocessor', preprocessor_configs["Proteomics only"]),
        ('variance_threshold', VarianceThreshold(threshold=0.0)),
        ('feature_selection', SelectKBest(score_func=f_classif, k=k_val)),
        ('model', RandomForestClassifier()),
    ])

    adjusted_params = []
    for p in lhs_param_list:
        p_copy = p.copy()
        if 'feature_selection__k' in p_copy:
            p_copy['feature_selection__k'] = [
                min(p_copy['feature_selection__k'][0], n_features)]
        adjusted_params.append(p_copy)

    try:
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=adjusted_params,
            cv=GroupKFold(n_splits=N_CV_SPLITS),
            scoring='roc_auc',
            n_jobs=N_JOBS_SAFE, refit=True,
            return_train_score=True, random_state=RANDOM_STATE,
        )
        search.fit(s["X_train"], s["y_train"], groups=s["groups_train"])
        model = search.best_estimator_
        print(f"  CV AUC: {search.best_score_:.4f}")
    except MemoryError:
        print("  MemoryError -- falling back to default RF")
        model = IMBPipeline([
            ('preprocessor', preprocessor_configs["Proteomics only"]),
            ('variance_threshold', VarianceThreshold(threshold=0.0)),
            ('feature_selection', SelectKBest(score_func=f_classif, k=k_val)),
            ('model', RandomForestClassifier(
                n_estimators=500, class_weight='balanced',
                random_state=RANDOM_STATE)),
        ])
        model.fit(s["X_train"], s["y_train"])

    probas_val = model.predict_proba(s["X_val"])[:, 1]
    preds_val = model.predict(s["X_val"])
    val_auc = roc_auc_score(s["y_val"], probas_val)
    test_auc = roc_auc_score(
        s["y_test"], model.predict_proba(s["X_test"])[:, 1])
    print(f"  Internal test AUC: {test_auc:.4f}")
    print(f"  External val AUC:  {val_auc:.4f}")
    gc.collect()
    return model, probas_val, preds_val


def train_rna_model(splits, preprocessor_configs):
    """Train RNA-seq PCA(50) + LogReg(L1) pipeline.

    Returns (model, probas_val, preds_val).
    """
    s = splits["RNA-seq only"]

    pipe = SKPipeline([
        ('variance_threshold', VarianceThreshold(threshold=0.0)),
        ('preprocessor', preprocessor_configs['RNA-seq only']),
        ('pca', PCA(n_components=50, random_state=RANDOM_STATE)),
        ('model', LogisticRegression(
            penalty='l1', solver='saga', C=1.0, max_iter=5000,
            class_weight='balanced', random_state=RANDOM_STATE)),
    ])
    pipe.fit(s["X_train"], s["y_train"])

    probas_val = pipe.predict_proba(s["X_val"])[:, 1]
    preds_val = pipe.predict(s["X_val"])
    val_auc = roc_auc_score(s["y_val"], probas_val)
    test_auc = roc_auc_score(
        s["y_test"], pipe.predict_proba(s["X_test"])[:, 1])
    print(f"  Internal test AUC: {test_auc:.4f}")
    print(f"  External val AUC:  {val_auc:.4f}")
    gc.collect()
    return pipe, probas_val, preds_val


def train_early_fusion(splits, preprocessor_configs, lhs_param_list_combined):
    """Train early fusion (naive concat -> SelectKBest -> RF).

    Returns (model, probas_val, preds_val).
    """
    s = splits["Combined"]
    n_features = s["X_train"].shape[1]
    k_val = min(41, n_features)

    pipe = IMBPipeline([
        ('preprocessor', preprocessor_configs["Combined"]),
        ('variance_threshold', VarianceThreshold(threshold=0.0)),
        ('feature_selection', SelectKBest(score_func=f_classif, k=k_val)),
        ('model', RandomForestClassifier()),
    ])

    adjusted_params = []
    for p in lhs_param_list_combined:
        p_copy = p.copy()
        if 'feature_selection__k' in p_copy:
            p_copy['feature_selection__k'] = [
                min(p_copy['feature_selection__k'][0], n_features)]
        adjusted_params.append(p_copy)

    try:
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=adjusted_params,
            cv=GroupKFold(n_splits=N_CV_SPLITS),
            scoring='roc_auc',
            n_jobs=N_JOBS_SAFE, refit=True,
            return_train_score=True, random_state=RANDOM_STATE,
        )
        search.fit(s["X_train"], s["y_train"], groups=s["groups_train"])
        model = search.best_estimator_
        print(f"  CV AUC: {search.best_score_:.4f}")
    except MemoryError:
        print("  MemoryError -- falling back to default RF")
        model = IMBPipeline([
            ('preprocessor', preprocessor_configs["Combined"]),
            ('variance_threshold', VarianceThreshold(threshold=0.0)),
            ('feature_selection', SelectKBest(score_func=f_classif, k=k_val)),
            ('model', RandomForestClassifier(
                n_estimators=500, class_weight='balanced',
                random_state=RANDOM_STATE)),
        ])
        model.fit(s["X_train"], s["y_train"])

    probas_val = model.predict_proba(s["X_val"])[:, 1]
    preds_val = model.predict(s["X_val"])
    val_auc = roc_auc_score(s["y_val"], probas_val)
    print(f"  External val AUC: {val_auc:.4f}")
    gc.collect()
    return model, probas_val, preds_val


def train_balanced_early_fusion(splits, preprocessor_configs,
                                 prot_col_set, rna_col_set):
    """Train balanced early fusion (per-modality PCA -> LogReg).

    Returns (model, probas_val, preds_val).
    """
    s = splits["Combined"]
    all_feature_cols = s["X_train"].columns.tolist()

    pipe = SKPipeline([
        ('preprocessor', preprocessor_configs["Combined"]),
        ('balanced_reducer', ModalityBalancedReducerColNames(
            prot_col_set=prot_col_set, rna_col_set=rna_col_set,
            all_col_names=all_feature_cols,
            n_prot_components=20, n_rna_components=30,
            random_state=RANDOM_STATE)),
        ('model', LogisticRegression(
            penalty='l2', solver='lbfgs', C=1.0, max_iter=5000,
            class_weight='balanced', random_state=RANDOM_STATE)),
    ])
    pipe.fit(s["X_train"], s["y_train"])

    probas_val = pipe.predict_proba(s["X_val"])[:, 1]
    preds_val = pipe.predict(s["X_val"])
    val_auc = roc_auc_score(s["y_val"], probas_val)
    test_auc = roc_auc_score(
        s["y_test"], pipe.predict_proba(s["X_test"])[:, 1])
    print(f"  Components: 20 prot + 30 RNA = 50 total")
    print(f"  Internal test AUC: {test_auc:.4f}")
    print(f"  External val AUC:  {val_auc:.4f}")
    gc.collect()
    return pipe, probas_val, preds_val


def _align_columns(X_source, target_cols):
    """Create a DataFrame aligned to target_cols, filling missing with 0."""
    X_aligned = pd.DataFrame(0, index=X_source.index, columns=target_cols)
    for c in target_cols:
        if c in X_source.columns:
            X_aligned[c] = X_source[c].values
    return X_aligned


def train_late_fusion(splits, prot_model, rna_model, prot_expected_cols,
                      rna_expected_cols):
    """Train late fusion (avg and weighted avg of unimodal probabilities).

    Returns dict with keys 'avg' and 'weighted', each containing
    (wrapper, probas_val, preds_val, best_weight).
    """
    s_comb = splits["Combined"]

    # Align validation features
    X_val_for_prot = _align_columns(s_comb["X_val"], prot_expected_cols)
    X_val_for_rna = _align_columns(s_comb["X_val"], rna_expected_cols)

    proba_prot = prot_model.predict_proba(X_val_for_prot)[:, 1]
    proba_rna = rna_model.predict_proba(X_val_for_rna)[:, 1]

    # Average fusion
    proba_avg = (proba_prot + proba_rna) / 2.0
    pred_avg = (proba_avg >= 0.5).astype(int)
    val_auc_avg = roc_auc_score(s_comb["y_val"], proba_avg)
    print(f"  Late Fusion (avg) -- External val AUC: {val_auc_avg:.4f}")

    # Weighted fusion: optimise on internal test set
    X_test_for_prot = _align_columns(s_comb["X_test"], prot_expected_cols)
    X_test_for_rna = _align_columns(s_comb["X_test"], rna_expected_cols)

    proba_prot_test = prot_model.predict_proba(X_test_for_prot)[:, 1]
    proba_rna_test = rna_model.predict_proba(X_test_for_rna)[:, 1]

    best_w, best_auc_w = 0.5, 0.0
    for w in np.arange(0.0, 1.01, 0.05):
        combo = w * proba_prot_test + (1 - w) * proba_rna_test
        auc_w = roc_auc_score(s_comb["y_test"], combo)
        if auc_w > best_auc_w:
            best_w, best_auc_w = w, auc_w

    print(f"  Optimal weight (on internal test): w_prot={best_w:.2f}, "
          f"w_rna={1-best_w:.2f}")
    print(f"  Internal test AUC at optimal: {best_auc_w:.4f}")

    proba_weighted = best_w * proba_prot + (1 - best_w) * proba_rna
    pred_weighted = (proba_weighted >= 0.5).astype(int)
    val_auc_weighted = roc_auc_score(s_comb["y_val"], proba_weighted)
    print(f"  Late Fusion (weighted) -- External val AUC: {val_auc_weighted:.4f}")

    wrapper_avg = LateFusionWrapper(
        prot_model, rna_model, 0.5, prot_expected_cols, rna_expected_cols)
    wrapper_weighted = LateFusionWrapper(
        prot_model, rna_model, best_w, prot_expected_cols, rna_expected_cols)

    return {
        "avg": (wrapper_avg, proba_avg, pred_avg, 0.5),
        "weighted": (wrapper_weighted, proba_weighted, pred_weighted, best_w),
        # Store test probas for stacking
        "_proba_prot_test": proba_prot_test,
        "_proba_rna_test": proba_rna_test,
        "_X_val_for_prot": X_val_for_prot,
        "_X_val_for_rna": X_val_for_rna,
    }


def train_stacking(splits, prot_model, rna_model, preprocessor_configs,
                   prot_expected_cols, rna_expected_cols,
                   late_fusion_cache=None):
    """Train stacking meta-learner on OOF unimodal predictions.

    Returns (wrapper, probas_val, preds_val).
    """
    s_comb = splits["Combined"]
    k_val_prot = min(41, splits["Proteomics only"]["X_train"].shape[1])

    # Fresh unimodal pipelines for OOF generation
    prot_pipe_for_stack = IMBPipeline([
        ('preprocessor', preprocessor_configs["Proteomics only"]),
        ('feature_selection', SelectKBest(score_func=f_classif, k=k_val_prot)),
        ('model', RandomForestClassifier(
            n_estimators=1000, class_weight='balanced',
            random_state=RANDOM_STATE)),
    ])
    rna_pipe_for_stack = SKPipeline([
        ('variance_threshold', VarianceThreshold(threshold=0.0)),
        ('preprocessor', preprocessor_configs['RNA-seq only']),
        ('pca', PCA(n_components=50, random_state=RANDOM_STATE)),
        ('model', LogisticRegression(
            penalty='l1', solver='saga', C=1.0, max_iter=5000,
            class_weight='balanced', random_state=RANDOM_STATE)),
    ])

    # Extract unimodal features from Combined training set
    X_train_prot = _align_columns(s_comb["X_train"], prot_expected_cols)
    X_train_rna = _align_columns(s_comb["X_train"], rna_expected_cols)

    gkf = GroupKFold(n_splits=5)

    print("  Generating OOF predictions for proteomics...")
    oof_prot = cross_val_predict(
        prot_pipe_for_stack, X_train_prot, s_comb["y_train"],
        groups=s_comb["groups_train"], cv=gkf,
        method='predict_proba', n_jobs=1,
    )[:, 1]

    print("  Generating OOF predictions for RNA-seq...")
    oof_rna = cross_val_predict(
        rna_pipe_for_stack, X_train_rna, s_comb["y_train"],
        groups=s_comb["groups_train"], cv=gkf,
        method='predict_proba', n_jobs=1,
    )[:, 1]

    # Meta-learner
    X_meta_train = np.column_stack([oof_prot, oof_rna])
    meta_learner = LogisticRegressionCV(
        cv=5, scoring='roc_auc', class_weight='balanced',
        random_state=RANDOM_STATE, max_iter=5000,
    )
    meta_learner.fit(X_meta_train, s_comb["y_train"])

    meta_coefs = meta_learner.coef_[0]
    print(f"  Meta-learner coefficients: prot={meta_coefs[0]:.3f}, "
          f"rna={meta_coefs[1]:.3f}")
    print(f"  Meta-learner C: {meta_learner.C_[0]:.4f}")

    # Score on validation set
    if late_fusion_cache is not None:
        X_val_for_prot = late_fusion_cache["_X_val_for_prot"]
        X_val_for_rna = late_fusion_cache["_X_val_for_rna"]
    else:
        X_val_for_prot = _align_columns(s_comb["X_val"], prot_expected_cols)
        X_val_for_rna = _align_columns(s_comb["X_val"], rna_expected_cols)

    proba_prot_val = prot_model.predict_proba(X_val_for_prot)[:, 1]
    proba_rna_val = rna_model.predict_proba(X_val_for_rna)[:, 1]
    X_meta_val = np.column_stack([proba_prot_val, proba_rna_val])

    probas_val = meta_learner.predict_proba(X_meta_val)[:, 1]
    preds_val = meta_learner.predict(X_meta_val)
    val_auc = roc_auc_score(s_comb["y_val"], probas_val)

    # Internal test
    if late_fusion_cache is not None:
        proba_prot_test = late_fusion_cache["_proba_prot_test"]
        proba_rna_test = late_fusion_cache["_proba_rna_test"]
    else:
        X_test_for_prot = _align_columns(s_comb["X_test"], prot_expected_cols)
        X_test_for_rna = _align_columns(s_comb["X_test"], rna_expected_cols)
        proba_prot_test = prot_model.predict_proba(X_test_for_prot)[:, 1]
        proba_rna_test = rna_model.predict_proba(X_test_for_rna)[:, 1]

    X_meta_test = np.column_stack([proba_prot_test, proba_rna_test])
    test_auc = roc_auc_score(
        s_comb["y_test"], meta_learner.predict_proba(X_meta_test)[:, 1])

    print(f"  Internal test AUC: {test_auc:.4f}")
    print(f"  External val AUC:  {val_auc:.4f}")

    wrapper = StackingWrapper(
        prot_model, rna_model, meta_learner,
        prot_expected_cols, rna_expected_cols)
    gc.collect()
    return wrapper, probas_val, preds_val


def run_all_training(splits, datasets, preprocessor_configs,
                     lhs_param_list, lhs_param_list_combined,
                     proteomics_cols, columns_rna):
    """Run complete model training pipeline.

    Returns (trained_models, all_probas, all_preds).
    """
    trained_models = {}
    all_probas = {}
    all_preds = {}

    prot_col_set = set(proteomics_cols)
    rna_col_set = set(columns_rna)

    # Phase 1: Unimodal
    print("=== Training: Proteomics only ===")
    model, probas, preds = train_proteomics_model(
        splits, preprocessor_configs, lhs_param_list)
    trained_models["Proteomics only"] = model
    all_probas["Proteomics only"] = probas
    all_preds["Proteomics only"] = preds
    print("  Done")

    print("\n=== Training: RNA-seq only ===")
    print("  [Using best RNA pipeline: PCA(50) + LogReg(L1)]")
    model, probas, preds = train_rna_model(splits, preprocessor_configs)
    trained_models["RNA-seq only"] = model
    all_probas["RNA-seq only"] = probas
    all_preds["RNA-seq only"] = preds
    print("  Done")

    prot_model = trained_models["Proteomics only"]
    rna_model = trained_models["RNA-seq only"]
    prot_expected_cols = splits["Proteomics only"]["X_train"].columns
    rna_expected_cols = splits["RNA-seq only"]["X_train"].columns

    # Phase 2: Multimodal fusion
    s_comb = splits["Combined"]
    all_feature_cols = s_comb["X_train"].columns.tolist()
    rna_cols_in_comb = [c for c in all_feature_cols if c in rna_col_set]
    prot_cols_in_comb = [c for c in all_feature_cols if c in prot_col_set]
    print(f"\n--- Combined dataset feature split ---")
    print(f"  Proteomics features: {len(prot_cols_in_comb)}")
    print(f"  RNA-seq features:    {len(rna_cols_in_comb)}")
    print(f"  Total:               {len(all_feature_cols)}")

    # Strategy 1: Early Fusion
    print("\n=== Combined: Early Fusion (naive concat -> RF) ===")
    model, probas, preds = train_early_fusion(
        splits, preprocessor_configs, lhs_param_list_combined)
    trained_models["Combined"] = model
    all_probas["Combined"] = probas
    all_preds["Combined"] = preds
    print("  Done")

    # Strategy 2: Balanced Early Fusion
    print("\n=== Combined: Balanced Early Fusion (PCA per modality -> LogReg) ===")
    model, probas, preds = train_balanced_early_fusion(
        splits, preprocessor_configs, prot_col_set, rna_col_set)
    trained_models["Balanced Early Fusion"] = model
    all_probas["Balanced Early Fusion"] = probas
    all_preds["Balanced Early Fusion"] = preds
    print("  Done")

    # Strategies 3 & 4: Late Fusion
    print("\n=== Combined: Late Fusion (avg / weighted avg) ===")
    lf_results = train_late_fusion(
        splits, prot_model, rna_model,
        prot_expected_cols, rna_expected_cols)

    wrapper_avg, probas_avg, preds_avg, _ = lf_results["avg"]
    trained_models["Late Fusion (avg)"] = wrapper_avg
    all_probas["Late Fusion (avg)"] = probas_avg
    all_preds["Late Fusion (avg)"] = preds_avg

    wrapper_w, probas_w, preds_w, _ = lf_results["weighted"]
    trained_models["Late Fusion (weighted)"] = wrapper_w
    all_probas["Late Fusion (weighted)"] = probas_w
    all_preds["Late Fusion (weighted)"] = preds_w
    print("  Done")

    # Strategy 5: Stacking
    print("\n=== Combined: Stacking Meta-Learner ===")
    model, probas, preds = train_stacking(
        splits, prot_model, rna_model, preprocessor_configs,
        prot_expected_cols, rna_expected_cols,
        late_fusion_cache=lf_results)
    trained_models["Stacking"] = model
    all_probas["Stacking"] = probas
    all_preds["Stacking"] = preds
    print("  Done")

    # Summary
    print("\n" + "=" * 70)
    print("MODEL TRAINING SUMMARY")
    print("=" * 70)

    summary_data = []
    for name in ["Proteomics only", "RNA-seq only", "Combined",
                 "Balanced Early Fusion", "Late Fusion (avg)",
                 "Late Fusion (weighted)", "Stacking"]:
        if name in all_probas:
            if name == "Proteomics only":
                y_val_ref = splits["Proteomics only"]["y_val"]
            elif name == "RNA-seq only":
                y_val_ref = splits["RNA-seq only"]["y_val"]
            else:
                y_val_ref = s_comb["y_val"]
            auc_val = roc_auc_score(y_val_ref, all_probas[name])
            summary_data.append({"Model": name, "External_Val_AUC": auc_val})

    df_summary = pd.DataFrame(summary_data).set_index("Model")
    print(df_summary.round(4).to_string())

    # Map fusion model names to Combined split for metrics computation
    fusion_names = ["Combined", "Balanced Early Fusion",
                    "Late Fusion (avg)", "Late Fusion (weighted)", "Stacking"]
    for fname in fusion_names:
        if fname not in splits:
            splits[fname] = splits["Combined"]

    return trained_models, all_probas, all_preds

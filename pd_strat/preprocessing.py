"""
Fold-aware preprocessing transformers for sklearn pipelines.

All transformers follow the BaseEstimator / TransformerMixin protocol
and are fit on TRAINING DATA ONLY within each CV fold, preventing
data leakage from global normalization or batch correction.

Classes
-------
FoldAwarePreprocessor
    Log2 + scaling + optional per-cohort batch centering.
FoldAwarePreprocessorSimple
    Log2 + scaling (pipeline-compatible, no batch params at transform time).
FoldAwareCombatPreprocessor
    Full ComBat empirical Bayes correction (train-only fitting).
NaNSafeScaler
    StandardScaler that handles NaN and zero-variance gracefully.
ClipNonNegative
    Clips values to >= 0 (required for NMF).
CombatCVPipeline
    CV wrapper — within single-cohort folds ComBat is skipped.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA

from combat.pycombat import (
    define_batchmod, check_ref_batch, treat_batches, treat_covariates,
    check_NAs, calculate_mean_var, calculate_stand_mean,
    standardise_data, fit_model, adjust_data,
)

from .config import RANDOM_STATE


# ─────────────────────────────────────────────────────────────
# FoldAwarePreprocessor (full-featured)
# ─────────────────────────────────────────────────────────────
class FoldAwarePreprocessor(BaseEstimator, TransformerMixin):
    """
    Fold-aware preprocessing: log2 + standardisation + optional batch correction.

    Placed at the START of every sklearn pipeline. Within each CV fold:
        .fit(X_train_fold) → learns parameters from training fold ONLY
        .transform(X_test_fold) → applies those parameters to held-out fold

    Steps (all fit on training data only):
        1. Optional log2(x+1) for RNA features
        2. Per-feature centering and scaling (StandardScaler)
        3. Optional per-cohort centering when batch effects are detected

    Parameters
    ----------
    apply_log2 : bool
        Whether to apply log2(x+1) to all features.
    rna_col_indices : list or None
        Column indices that are RNA features (for Combined datasets).
    correct_batch : bool
        If True, estimate per-cohort means on training data and subtract.
        Requires cohort_labels to be passed via .fit(X, cohort_labels=...).
    """

    def __init__(self, apply_log2=False, rna_col_indices=None, correct_batch=False):
        self.apply_log2 = apply_log2
        self.rna_col_indices = rna_col_indices
        self.correct_batch = correct_batch

    def fit(self, X, y=None, **fit_params):
        X_work = np.array(X, dtype=np.float64).copy()

        if self.apply_log2:
            if self.rna_col_indices is not None:
                X_work[:, self.rna_col_indices] = np.log2(
                    np.clip(X_work[:, self.rna_col_indices], 0, None) + 1)
            else:
                X_work = np.log2(np.clip(X_work, 0, None) + 1)

        self.mean_ = np.nanmean(X_work, axis=0)
        self.std_ = np.nanstd(X_work, axis=0)
        self.std_[self.std_ < 1e-10] = 1.0

        self.batch_means_ = None
        if self.correct_batch and "cohort_labels" in fit_params:
            labels = fit_params["cohort_labels"]
            unique_batches = np.unique(labels)
            if len(unique_batches) >= 2:
                self.batch_means_ = {}
                X_centered = (X_work - self.mean_) / self.std_
                for batch in unique_batches:
                    mask = labels == batch
                    if mask.sum() >= 5:
                        self.batch_means_[batch] = np.nanmean(X_centered[mask], axis=0)
        return self

    def transform(self, X, **transform_params):
        X_work = np.array(X, dtype=np.float64).copy()

        if self.apply_log2:
            if self.rna_col_indices is not None:
                X_work[:, self.rna_col_indices] = np.log2(
                    np.clip(X_work[:, self.rna_col_indices], 0, None) + 1)
            else:
                X_work = np.log2(np.clip(X_work, 0, None) + 1)

        X_work = (X_work - self.mean_) / self.std_

        if self.batch_means_ is not None and "cohort_labels" in transform_params:
            labels = transform_params["cohort_labels"]
            for batch, offset in self.batch_means_.items():
                mask = labels == batch
                if mask.any():
                    X_work[mask] -= offset

        X_work = np.nan_to_num(X_work, nan=0.0)
        return X_work

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X, **{k: v for k, v in fit_params.items()
                                     if k == "cohort_labels"})


# ─────────────────────────────────────────────────────────────
# FoldAwarePreprocessorSimple (pipeline-compatible)
# ─────────────────────────────────────────────────────────────
class FoldAwarePreprocessorSimple(BaseEstimator, TransformerMixin):
    """
    Simplified version that works within standard sklearn Pipeline.
    Handles log2 + scaling + NaN imputation. No batch correction.
    """

    def __init__(self, apply_log2=False, rna_col_indices=None):
        self.apply_log2 = apply_log2
        self.rna_col_indices = rna_col_indices

    def fit(self, X, y=None):
        X_work = np.array(X, dtype=np.float64).copy()
        if self.apply_log2:
            if self.rna_col_indices is not None:
                X_work[:, self.rna_col_indices] = np.log2(
                    np.clip(X_work[:, self.rna_col_indices], 0, None) + 1)
            else:
                X_work = np.log2(np.clip(X_work, 0, None) + 1)
        self.mean_ = np.nanmean(X_work, axis=0)
        self.std_ = np.nanstd(X_work, axis=0)
        self.std_[self.std_ < 1e-10] = 1.0
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X):
        X_work = np.array(X, dtype=np.float64).copy()
        if self.apply_log2:
            if self.rna_col_indices is not None:
                X_work[:, self.rna_col_indices] = np.log2(
                    np.clip(X_work[:, self.rna_col_indices], 0, None) + 1)
            else:
                X_work = np.log2(np.clip(X_work, 0, None) + 1)
        X_work = (X_work - self.mean_) / self.std_
        X_work = np.nan_to_num(X_work, nan=0.0)
        return X_work


# ─────────────────────────────────────────────────────────────
# FoldAwareCombatPreprocessor — PROPER train-only ComBat
# ─────────────────────────────────────────────────────────────
class FoldAwareCombatPreprocessor(BaseEstimator, TransformerMixin):
    """
    Fold-aware ComBat batch correction + log2 + scaling.

    Fits ComBat empirical Bayes parameters on training data only,
    then applies the learned batch correction to held-out/test data.

    Parameters
    ----------
    apply_log2 : bool
        Apply log2(x+1) before ComBat.
    rna_col_indices : list or None
        If set, log2 is only applied to these column indices.
    par_prior : bool
        Parametric (True) or non-parametric (False) EB priors.
    mean_only : bool
        If True, only correct additive (location) batch effects.
    """

    def __init__(self, apply_log2=False, rna_col_indices=None,
                 par_prior=True, mean_only=False):
        self.apply_log2 = apply_log2
        self.rna_col_indices = rna_col_indices
        self.par_prior = par_prior
        self.mean_only = mean_only

    @staticmethod
    def _patno_to_batch(patnos):
        """Derive batch labels from PATNO prefix."""
        batches = []
        for p in patnos:
            s = str(p)
            if s.startswith("PD-"):
                batches.append("PDBP")
            elif s.startswith("PP-"):
                batches.append("PPMI")
            else:
                batches.append("Other")
        return batches

    def _apply_log2_transform(self, X):
        if self.apply_log2:
            if self.rna_col_indices is not None:
                X[:, self.rna_col_indices] = np.log2(
                    np.clip(X[:, self.rna_col_indices], 0, None) + 1)
            else:
                X = np.log2(np.clip(X, 0, None) + 1)
        return X

    def fit(self, X, y=None, patnos=None, batch_labels=None):
        X_work = np.array(X, dtype=np.float64).copy()
        X_work = self._apply_log2_transform(X_work)

        if batch_labels is not None:
            batch = list(batch_labels)
        elif patnos is not None:
            batch = self._patno_to_batch(patnos)
        else:
            raise ValueError(
                "FoldAwareCombatPreprocessor.fit() requires either "
                "'patnos' or 'batch_labels'.")

        n_samples, n_features = X_work.shape
        unique_batches = sorted(set(batch))

        if len(unique_batches) < 2:
            print("  [ComBat] Only 1 batch in training data — skipping, "
                  "applying standard scaling only.")
            self.combat_fitted_ = False
            self.mean_ = np.nanmean(X_work, axis=0)
            self.std_ = np.nanstd(X_work, axis=0)
            self.std_[self.std_ < 1e-10] = 1.0
            return self

        dat_df = pd.DataFrame(
            X_work.T,
            index=[f"f{i}" for i in range(n_features)],
            columns=[f"s{i}" for i in range(n_samples)],
        )
        dat = dat_df.values

        batchmod = define_batchmod(batch)
        ref = None
        ref, batchmod = check_ref_batch(None, batch, batchmod)
        n_batch, batches_idx, n_batches, n_array = treat_batches(batch)
        design = treat_covariates(batchmod, [], ref, n_batch)
        NAs = check_NAs(dat)

        if NAs:
            raise ValueError("NaN values detected — impute before ComBat.")

        B_hat, grand_mean, var_pooled = calculate_mean_var(
            design, batches_idx, ref, dat, NAs, n_batches, n_batch, n_array)
        stand_mean = calculate_stand_mean(
            grand_mean, n_array, design, n_batch, B_hat)
        s_data = standardise_data(dat, stand_mean, var_pooled, n_array)

        gamma_star, delta_star, batch_design = fit_model(
            design, n_batch, s_data, batches_idx,
            self.mean_only, self.par_prior, None, ref, NAs)

        self.combat_fitted_ = True
        self.gamma_star_ = gamma_star
        self.delta_star_ = delta_star
        var_pooled = np.asarray(var_pooled, dtype=np.float64).ravel()
        var_pooled[var_pooled < 1e-10] = 1.0
        self.var_pooled_ = var_pooled
        self.grand_mean_ = grand_mean
        self.stand_mean_ = stand_mean
        self.B_hat_ = B_hat
        self.design_ = design
        self.n_batch_ = n_batch
        self.batch_labels_map_ = {b: i for i, b in enumerate(unique_batches)}
        self.unique_batches_ = unique_batches
        self.batches_idx_ = batches_idx
        self.n_features_ = n_features
        self.grand_mean_vec_ = np.squeeze(np.asarray(grand_mean))

        print(f"  [ComBat] Fitted on {n_samples} training samples, "
              f"{n_batch} batches {unique_batches}, {n_features} features")
        return self

    def transform(self, X, patnos=None, batch_labels=None):
        X_work = np.array(X, dtype=np.float64).copy()
        X_work = self._apply_log2_transform(X_work)

        if not self.combat_fitted_:
            X_work = (X_work - self.mean_) / self.std_
            return np.nan_to_num(X_work, nan=0.0)

        n_samples = X_work.shape[0]
        dat = X_work.T

        if batch_labels is not None:
            batch = list(batch_labels)
        elif patnos is not None:
            batch = self._patno_to_batch(patnos)
        else:
            raise ValueError(
                "FoldAwareCombatPreprocessor.transform() requires either "
                "'patnos' or 'batch_labels'.")

        gm = self.grand_mean_vec_
        if gm.ndim == 1:
            stand_mean_new = np.outer(gm, np.ones(n_samples))
        else:
            stand_mean_new = np.tile(gm.reshape(-1, 1), (1, n_samples))

        var_pooled = self.var_pooled_
        sqrt_var = np.sqrt(np.maximum(var_pooled, 1e-10))

        s_data = (dat - stand_mean_new) / sqrt_var[:, np.newaxis]

        corrected = s_data.copy()
        for j in range(n_samples):
            b = batch[j]
            if b in self.batch_labels_map_:
                bi = self.batch_labels_map_[b]
                delta_safe = np.maximum(self.delta_star_[bi, :], 1e-10)
                corrected[:, j] = (
                    (s_data[:, j] - self.gamma_star_[bi, :]) /
                    np.sqrt(delta_safe)
                )

        result = corrected * sqrt_var[:, np.newaxis] + stand_mean_new
        result = np.nan_to_num(result.T, nan=0.0)
        return result

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X, **{k: v for k, v in fit_params.items()
                                     if k in ("patnos", "batch_labels")})


# ─────────────────────────────────────────────────────────────
# NaNSafeScaler — post-ComBat simple scaler
# ─────────────────────────────────────────────────────────────
class NaNSafeScaler(BaseEstimator, TransformerMixin):
    """StandardScaler that handles NaN and zero-variance gracefully."""

    def fit(self, X, y=None):
        X_arr = np.nan_to_num(np.array(X, dtype=np.float64), nan=0.0)
        self.mean_ = np.mean(X_arr, axis=0)
        self.std_ = np.std(X_arr, axis=0)
        self.std_[self.std_ < 1e-10] = 1.0
        return self

    def transform(self, X):
        X_arr = np.nan_to_num(np.array(X, dtype=np.float64), nan=0.0)
        return (X_arr - self.mean_) / self.std_

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X)


# ─────────────────────────────────────────────────────────────
# ClipNonNegative — for NMF input
# ─────────────────────────────────────────────────────────────
class ClipNonNegative(BaseEstimator, TransformerMixin):
    """Clips values to >= 0 (required for NMF)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.clip(X, 0, None)


# ─────────────────────────────────────────────────────────────
# GeneSetPCA — pathway-level PCA aggregation
# ─────────────────────────────────────────────────────────────
class GeneSetPCA(BaseEstimator, TransformerMixin):
    """Aggregate gene expression into pathway-level PCA scores.

    Gene sets should use the same column names as the input DataFrame.
    """

    def __init__(self, gene_sets, n_components_per_set=3, min_genes=3):
        self.gene_sets = gene_sets
        self.n_components_per_set = n_components_per_set
        self.min_genes = min_genes
        self.pca_models_ = {}
        self.valid_sets_ = {}
        self.fallback_pca_ = None

    def fit(self, X, y=None):
        cols = set(X.columns) if isinstance(X, pd.DataFrame) else set()

        self.pca_models_ = {}
        self.valid_sets_ = {}

        for gs_name, genes in self.gene_sets.items():
            present = [g for g in genes if g in cols] if isinstance(X, pd.DataFrame) else []
            if len(present) >= self.min_genes:
                n_comp = min(self.n_components_per_set, len(present))
                pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
                pca.fit(X[present])
                self.pca_models_[gs_name] = pca
                self.valid_sets_[gs_name] = present

        if not self.pca_models_:
            print("  WARNING: No gene sets had enough members. Using global PCA fallback.")
            self.fallback_pca_ = PCA(n_components=20, random_state=RANDOM_STATE)
            X_arr = X.values if isinstance(X, pd.DataFrame) else X
            self.fallback_pca_.fit(X_arr)

        return self

    def transform(self, X):
        if self.fallback_pca_ is not None:
            X_arr = X.values if isinstance(X, pd.DataFrame) else X
            return self.fallback_pca_.transform(X_arr)

        parts = []
        for gs_name, pca in self.pca_models_.items():
            present = self.valid_sets_[gs_name]
            sub = X[present] if isinstance(X, pd.DataFrame) else X[:, present]
            parts.append(pca.transform(sub))
        return np.hstack(parts)


# ─────────────────────────────────────────────────────────────
# ModalityBalancedReducer — per-modality PCA for early fusion
# ─────────────────────────────────────────────────────────────
class ModalityBalancedReducer(BaseEstimator, TransformerMixin):
    """Apply separate PCA to proteomics and RNA columns, then concatenate."""

    def __init__(self, prot_indices, rna_indices, n_prot_components=20,
                 n_rna_components=30, random_state=RANDOM_STATE):
        self.prot_indices = prot_indices
        self.rna_indices = rna_indices
        self.n_prot_components = n_prot_components
        self.n_rna_components = n_rna_components
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X)
        X_prot = X[:, self.prot_indices]
        X_rna = X[:, self.rna_indices]
        n_pc_prot = min(self.n_prot_components, X_prot.shape[1], X_prot.shape[0])
        n_pc_rna = min(self.n_rna_components, X_rna.shape[1], X_rna.shape[0])
        self.pca_prot_ = PCA(n_components=n_pc_prot, random_state=self.random_state)
        self.pca_rna_ = PCA(n_components=n_pc_rna, random_state=self.random_state)
        self.pca_prot_.fit(X_prot)
        self.pca_rna_.fit(X_rna)
        return self

    def transform(self, X):
        X = np.asarray(X)
        Z_prot = self.pca_prot_.transform(X[:, self.prot_indices])
        Z_rna = self.pca_rna_.transform(X[:, self.rna_indices])
        return np.hstack([Z_prot, Z_rna])


class ModalityBalancedReducerColNames(BaseEstimator, TransformerMixin):
    """PCA per modality using column name sets (robust to VT filtering)."""

    def __init__(self, prot_col_set, rna_col_set, all_col_names,
                 n_prot_components=20, n_rna_components=30,
                 random_state=RANDOM_STATE):
        self.prot_col_set = prot_col_set
        self.rna_col_set = rna_col_set
        self.all_col_names = all_col_names
        self.n_prot_components = n_prot_components
        self.n_rna_components = n_rna_components
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X)
        n_cols = X.shape[1]
        if n_cols == len(self.all_col_names):
            self.prot_idx_ = [i for i, c in enumerate(self.all_col_names)
                              if c in self.prot_col_set]
            self.rna_idx_ = [i for i, c in enumerate(self.all_col_names)
                             if c in self.rna_col_set]
        else:
            n_prot_est = len([c for c in self.all_col_names if c in self.prot_col_set])
            self.prot_idx_ = list(range(min(n_prot_est, n_cols)))
            self.rna_idx_ = list(range(len(self.prot_idx_), n_cols))

        X_prot = X[:, self.prot_idx_] if self.prot_idx_ else np.empty((X.shape[0], 0))
        X_rna = X[:, self.rna_idx_] if self.rna_idx_ else np.empty((X.shape[0], 0))

        n_pc_prot = min(self.n_prot_components, X_prot.shape[1], X_prot.shape[0]) if X_prot.shape[1] > 0 else 0
        n_pc_rna = min(self.n_rna_components, X_rna.shape[1], X_rna.shape[0]) if X_rna.shape[1] > 0 else 0

        self.pca_prot_ = PCA(n_components=n_pc_prot, random_state=self.random_state) if n_pc_prot > 0 else None
        self.pca_rna_ = PCA(n_components=n_pc_rna, random_state=self.random_state) if n_pc_rna > 0 else None

        if self.pca_prot_:
            self.pca_prot_.fit(X_prot)
        if self.pca_rna_:
            self.pca_rna_.fit(X_rna)
        return self

    def transform(self, X):
        X = np.asarray(X)
        parts = []
        if self.pca_prot_:
            parts.append(self.pca_prot_.transform(X[:, self.prot_idx_]))
        if self.pca_rna_:
            parts.append(self.pca_rna_.transform(X[:, self.rna_idx_]))
        return np.hstack(parts) if parts else X


# ─────────────────────────────────────────────────────────────
# CombatCVPipeline — CV wrapper
# ─────────────────────────────────────────────────────────────
class CombatCVPipeline(BaseEstimator):
    """CV wrapper — within single-cohort CV folds, ComBat is skipped."""

    def __init__(self, pipeline, dataset_name, apply_log2=False,
                 rna_col_indices=None, par_prior=True, mean_only=False):
        self.pipeline = pipeline
        self.dataset_name = dataset_name
        self.apply_log2 = apply_log2
        self.rna_col_indices = rna_col_indices
        self.par_prior = par_prior
        self.mean_only = mean_only

    def fit(self, X, y=None, groups=None, **fit_params):
        self.pipeline_ = clone(self.pipeline)
        self.pipeline_.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline_.predict(X)

    def predict_proba(self, X):
        return self.pipeline_.predict_proba(X)

    def get_params(self, deep=True):
        params = {
            "pipeline": self.pipeline, "dataset_name": self.dataset_name,
            "apply_log2": self.apply_log2, "rna_col_indices": self.rna_col_indices,
            "par_prior": self.par_prior, "mean_only": self.mean_only,
        }
        if deep and hasattr(self.pipeline, "get_params"):
            for key, val in self.pipeline.get_params(deep=True).items():
                params[f"pipeline__{key}"] = val
        return params

    def set_params(self, **params):
        pipeline_params = {}
        for key, val in params.items():
            if key.startswith("pipeline__"):
                pipeline_params[key[len("pipeline__"):]] = val
            elif hasattr(self, key):
                setattr(self, key, val)
        if pipeline_params:
            self.pipeline.set_params(**pipeline_params)
        return self

    @property
    def classes_(self):
        return self.pipeline_.classes_


# ─────────────────────────────────────────────────────────────
# Late Fusion / Stacking wrappers
# ─────────────────────────────────────────────────────────────
class LateFusionWrapper:
    """Thin wrapper so late fusion appears in trained_models for metrics."""

    def __init__(self, model_prot, model_rna, weight_prot, prot_cols, rna_cols):
        self.model_prot = model_prot
        self.model_rna = model_rna
        self.w = weight_prot
        self.prot_cols = prot_cols
        self.rna_cols = rna_cols

    def predict_proba(self, X):
        X_p = pd.DataFrame(0, index=range(len(X)), columns=self.prot_cols)
        X_r = pd.DataFrame(0, index=range(len(X)), columns=self.rna_cols)
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        for c in self.prot_cols:
            if c in X_df.columns:
                X_p[c] = X_df[c].values
        for c in self.rna_cols:
            if c in X_df.columns:
                X_r[c] = X_df[c].values
        p1 = self.model_prot.predict_proba(X_p)[:, 1]
        p2 = self.model_rna.predict_proba(X_r)[:, 1]
        pos = self.w * p1 + (1 - self.w) * p2
        return np.column_stack([1 - pos, pos])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class StackingWrapper:
    """Wrapper so stacking can be called like a standard estimator."""

    def __init__(self, model_prot, model_rna, meta, prot_cols, rna_cols):
        self.model_prot = model_prot
        self.model_rna = model_rna
        self.meta = meta
        self.prot_cols = prot_cols
        self.rna_cols = rna_cols

    def predict_proba(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        X_p = pd.DataFrame(0, index=range(len(X_df)), columns=self.prot_cols)
        X_r = pd.DataFrame(0, index=range(len(X_df)), columns=self.rna_cols)
        for c in self.prot_cols:
            if c in X_df.columns:
                X_p[c] = X_df[c].values
        for c in self.rna_cols:
            if c in X_df.columns:
                X_r[c] = X_df[c].values
        p1 = self.model_prot.predict_proba(X_p)[:, 1]
        p2 = self.model_rna.predict_proba(X_r)[:, 1]
        return self.meta.predict_proba(np.column_stack([p1, p2]))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

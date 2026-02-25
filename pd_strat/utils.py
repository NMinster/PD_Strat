"""
Shared utility functions: hyperparameter generation, bootstrap CI, DeLong test.
"""

import numpy as np
from scipy.stats import qmc, norm
from sklearn.metrics import roc_auc_score

from .config import (
    RANDOM_STATE, N_BOOTSTRAPS, ALPHA, LHS_N_ITER,
    MAX_ESTIMATORS_DEFAULT, MAX_ESTIMATORS_COMBINED,
)


def generate_lhs_params(n_iter=LHS_N_ITER, d=6, seed=RANDOM_STATE,
                        max_estimators=MAX_ESTIMATORS_DEFAULT):
    """Generate Latin Hypercube Sampled hyperparameters for RF pipeline."""
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    samples = sampler.random(n_iter)
    params = []
    for row in samples:
        n_estimators = int(row[0] * (max_estimators - 500)) + 500
        max_depth = None if row[1] < 1 / 11 else int((row[1] - 1 / 11) / (1 - 1 / 11) * (10 - 1)) + 1
        min_samples_split = int(row[2] * (10 - 2)) + 2
        min_samples_leaf = int(row[3] * (4 - 1)) + 1
        class_weight = "balanced" if row[4] < 0.5 else "balanced_subsample"
        feature_k = int(row[5] * (51 - 30)) + 30
        params.append({
            "model__n_estimators": n_estimators,
            "model__max_depth": max_depth,
            "model__min_samples_split": min_samples_split,
            "model__min_samples_leaf": min_samples_leaf,
            "model__class_weight": class_weight,
            "model__random_state": RANDOM_STATE,
            "feature_selection__k": feature_k,
        })
    return [{k: [v] for k, v in c.items()} for c in params]


def bootstrap_ci(y_true, y_score, metric_fn, n_bootstraps=N_BOOTSTRAPS,
                 alpha=ALPHA, seed=RANDOM_STATE):
    """Bootstrap resampling confidence interval for any metric function.

    Returns (mean, lo, hi).
    """
    rng = np.random.RandomState(seed)
    scores = []
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y_true_arr), len(y_true_arr))
        if len(np.unique(y_true_arr[idx])) < 2:
            continue
        scores.append(metric_fn(y_true_arr[idx], y_score_arr[idx]))
    scores = np.array(scores)
    lo = np.percentile(scores, 100 * alpha / 2)
    hi = np.percentile(scores, 100 * (1 - alpha / 2))
    return np.mean(scores), lo, hi


def bootstrap_paired_aucs(y_true_a, proba_a, y_true_b=None, proba_b=None,
                          n_bootstraps=N_BOOTSTRAPS, seed=RANDOM_STATE):
    """Bootstrap AUC difference between two models."""
    rng = np.random.RandomState(seed)
    diffs = []
    ya = np.asarray(y_true_a)
    pa = np.asarray(proba_a)
    if y_true_b is None:
        yb, pb = ya, np.asarray(proba_b)
    else:
        yb = np.asarray(y_true_b)
        pb = np.asarray(proba_b)
    for _ in range(n_bootstraps):
        idx_a = rng.randint(0, len(ya), len(ya))
        idx_b = rng.randint(0, len(yb), len(yb))
        if len(np.unique(ya[idx_a])) < 2 or len(np.unique(yb[idx_b])) < 2:
            continue
        auc_a = roc_auc_score(ya[idx_a], pa[idx_a])
        auc_b = roc_auc_score(yb[idx_b], pb[idx_b])
        diffs.append(auc_a - auc_b)
    return np.array(diffs)


# ── DeLong Test for ROC Comparison ──

def compute_midrank(x):
    """Compute midrank values for the DeLong test."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        for k in range(i, j):
            T[k] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N)
    T2[J] = T + 1
    return T2


def delong_roc_variance(ground_truth, predictions):
    """Compute AUC and its variance using the DeLong method."""
    order = (-ground_truth).argsort()
    label_ordered = ground_truth[order]
    predictions_sorted = predictions[order]
    m = np.sum(label_ordered == 1)
    n = np.sum(label_ordered == 0)
    positive_examples = predictions_sorted[label_ordered == 1]
    negative_examples = predictions_sorted[label_ordered == 0]
    midrank = compute_midrank(predictions_sorted)
    tx_pos = midrank[label_ordered == 1]
    aucs = (np.sum(tx_pos) - m * (m + 1) / 2.0) / (m * n)
    v01 = np.zeros(m)
    v10 = np.zeros(n)
    for i in range(m):
        v01[i] = ((1.0 / n) * np.sum(positive_examples[i] > negative_examples) +
                  (1.0 / (2 * n)) * np.sum(positive_examples[i] == negative_examples))
    for i in range(n):
        v10[i] = ((1.0 / m) * np.sum(negative_examples[i] < positive_examples) +
                  (1.0 / (2 * m)) * np.sum(negative_examples[i] == positive_examples))
    sx = np.var(v01, ddof=1)
    sy = np.var(v10, ddof=1)
    var_auc = sx / m + sy / n
    return aucs, var_auc


def delong_test(y_true, pred_a, pred_b):
    """DeLong test for comparing two ROC curves on the same test set.

    Returns (auc_a, auc_b, z_statistic, p_value).
    """
    y = np.asarray(y_true)
    auc_a, var_a = delong_roc_variance(y, np.asarray(pred_a))
    auc_b, var_b = delong_roc_variance(y, np.asarray(pred_b))
    z = (auc_a - auc_b) / np.sqrt(var_a + var_b + 1e-10)
    p = 2 * norm.sf(abs(z))
    return auc_a, auc_b, z, p

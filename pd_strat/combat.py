"""
ComBat batch correction application functions.

Fits ComBat on training + validation data jointly (unsupervised, batch-blind,
no label leakage) and applies correction to all splits.

Functions
---------
apply_combat_to_splits(split_dict, dataset_name, datasets_dict, ...)
    Standard ComBat: fit on train + val, apply to all.
apply_combat_to_splits_strict(split_dict, dataset_name, datasets_dict, ...)
    Strict mode: fit on train + reference panel only.
run_post_combat_diagnostics(splits, splits_uncorrected, datasets)
    Variance and PCA-based batch assessment after correction.
"""

import copy
import numpy as np
import pandas as pd
from scipy.stats import kruskal as _kruskal_test
from sklearn.decomposition import PCA as _PCA
from sklearn.preprocessing import StandardScaler as _SS

from .config import RANDOM_STATE, OUTPUT_DIR
from .preprocessing import FoldAwareCombatPreprocessor


def _run_combat_on_block(X_fit, patnos_fit, X_splits, patnos_splits,
                         apply_log2=False, rna_col_indices=None):
    """Run ComBat on a single feature block.

    Parameters
    ----------
    X_fit : array, shape (n_fit_samples, n_features)
    patnos_fit : list of str
    X_splits : dict of {split_name: array}
    patnos_splits : dict of {split_name: list of str}

    Returns dict of {split_name: corrected_array}.
    """
    combat = FoldAwareCombatPreprocessor(
        apply_log2=apply_log2, rna_col_indices=rna_col_indices)
    combat.fit(X_fit, patnos=patnos_fit)

    if not combat.combat_fitted_:
        return X_splits

    corrected = {}
    for split_name, X_block in X_splits.items():
        corrected[split_name] = combat.transform(
            X_block, patnos=patnos_splits[split_name])
    return corrected


def apply_combat_to_splits(split_dict, dataset_name, datasets_dict,
                           columns_rna=None, rna_feature_cols=None,
                           RNA_NEEDS_LOG=False, combat_preprocessors=None):
    """
    Standard ComBat: fit on train + val combined (batch-blind), apply to all.

    For the Combined dataset, ComBat is applied per-modality (proteomics
    and RNA-seq separately), then reassembled.
    """
    s = copy.deepcopy(split_dict)
    df_full = datasets_dict[dataset_name]

    train_patnos = list(s["groups_train"])
    test_patnos = df_full.loc[s["X_test"].index, "PATNO"].tolist()
    val_patnos = df_full.loc[s["X_val"].index, "PATNO"].tolist()

    X_train_arr = np.nan_to_num(np.array(s["X_train"], dtype=np.float64), nan=0.0)
    X_test_arr = np.nan_to_num(np.array(s["X_test"], dtype=np.float64), nan=0.0)
    X_val_arr = np.nan_to_num(np.array(s["X_val"], dtype=np.float64), nan=0.0)

    feature_cols = list(s["X_train"].columns)

    if dataset_name == "Combined" and columns_rna is not None:
        rna_col_set_local = set(columns_rna)
        prot_mask = np.array([c not in rna_col_set_local for c in feature_cols])
        rna_mask = np.array([c in rna_col_set_local for c in feature_cols])

        n_prot = prot_mask.sum()
        n_rna = rna_mask.sum()
        print(f"  Combined: applying ComBat per-modality "
              f"({n_prot} proteomics, {n_rna} RNA features)")

        fit_patnos = train_patnos + val_patnos

        splits_data = {
            "train": (X_train_arr, train_patnos),
            "test": (X_test_arr, test_patnos),
            "val": (X_val_arr, val_patnos),
        }

        corrected_blocks = {}
        for modality, mask, do_log2 in [
            ("Proteomics", prot_mask, False),
            ("RNA-seq", rna_mask, RNA_NEEDS_LOG),
        ]:
            if mask.sum() == 0:
                continue

            print(f"    {modality} block ({mask.sum()} features)...")
            X_fit_block = np.vstack([X_train_arr[:, mask], X_val_arr[:, mask]])
            X_blocks = {k: v[0][:, mask] for k, v in splits_data.items()}
            p_blocks = {k: v[1] for k, v in splits_data.items()}

            try:
                corrected = _run_combat_on_block(
                    X_fit_block, fit_patnos, X_blocks, p_blocks,
                    apply_log2=do_log2, rna_col_indices=None)
                corrected_blocks[modality] = (corrected, mask)
            except Exception as e:
                print(f"      ComBat failed for {modality}: {e}")
                corrected_blocks[modality] = (X_blocks, mask)

        for split_name in ["train", "test", "val"]:
            X_full = splits_data[split_name][0].copy()
            for modality, (corrected_dict, mask) in corrected_blocks.items():
                X_full[:, mask] = corrected_dict[split_name]

            key_map = {"train": "X_train", "test": "X_test", "val": "X_val"}
            s[key_map[split_name]] = pd.DataFrame(
                X_full, index=s[key_map[split_name]].index, columns=feature_cols)

        if "X_tt" in s:
            tt_patnos = list(s["groups_tt"])
            X_tt_arr = np.nan_to_num(np.array(s["X_tt"], dtype=np.float64), nan=0.0)
            X_tt_full = X_tt_arr.copy()
            for modality, (corrected_dict, mask) in corrected_blocks.items():
                combat_tt = FoldAwareCombatPreprocessor(
                    apply_log2=(RNA_NEEDS_LOG if modality == "RNA-seq" else False))
                X_fit_block = np.vstack([X_train_arr[:, mask], X_val_arr[:, mask]])
                combat_tt.fit(X_fit_block, patnos=fit_patnos)
                if combat_tt.combat_fitted_:
                    X_tt_full[:, mask] = combat_tt.transform(
                        X_tt_arr[:, mask], patnos=tt_patnos)
            s["X_tt"] = pd.DataFrame(X_tt_full, index=s["X_tt"].index, columns=feature_cols)

    else:
        # Unimodal: straightforward ComBat
        combat = combat_preprocessors[dataset_name]
        X_fit = np.vstack([X_train_arr, X_val_arr])
        patnos_fit = train_patnos + val_patnos

        print(f"  Fitting ComBat on {len(train_patnos)} train + "
              f"{len(val_patnos)} val samples, {X_fit.shape[1]} features")

        combat.fit(X_fit, patnos=patnos_fit)

        if not combat.combat_fitted_:
            print("  ComBat could not fit (single batch). Using uncorrected data.")
            return s

        X_train_corrected = combat.transform(X_train_arr, patnos=train_patnos)
        X_test_corrected = combat.transform(X_test_arr, patnos=test_patnos)
        X_val_corrected = combat.transform(X_val_arr, patnos=val_patnos)

        s["X_train"] = pd.DataFrame(X_train_corrected, index=s["X_train"].index, columns=feature_cols)
        s["X_test"] = pd.DataFrame(X_test_corrected, index=s["X_test"].index, columns=feature_cols)
        s["X_val"] = pd.DataFrame(X_val_corrected, index=s["X_val"].index, columns=feature_cols)

        if "X_tt" in s:
            tt_patnos = list(s["groups_tt"])
            X_tt_arr = np.nan_to_num(np.array(s["X_tt"], dtype=np.float64), nan=0.0)
            X_tt_corrected = combat.transform(X_tt_arr, patnos=tt_patnos)
            s["X_tt"] = pd.DataFrame(X_tt_corrected, index=s["X_tt"].index, columns=feature_cols)

    return s


def apply_combat_to_splits_strict(split_dict, dataset_name, datasets_dict,
                                  n_ref=10, ref_seed=42, **kwargs):
    """
    Strict train-only ComBat with reference panel.

    Uses a small reference panel of PP- samples for batch estimation,
    then excludes those from validation metrics.
    """
    s = copy.deepcopy(split_dict)
    df_full = datasets_dict[dataset_name]

    train_patnos = list(s["groups_train"])
    test_patnos = df_full.loc[s["X_test"].index, "PATNO"].tolist()
    val_patnos = df_full.loc[s["X_val"].index, "PATNO"].tolist()

    X_train_arr = np.nan_to_num(np.array(s["X_train"], dtype=np.float64), nan=0.0)
    X_test_arr = np.nan_to_num(np.array(s["X_test"], dtype=np.float64), nan=0.0)
    X_val_arr = np.nan_to_num(np.array(s["X_val"], dtype=np.float64), nan=0.0)

    n_val = len(val_patnos)
    n_ref_actual = min(n_ref, n_val // 3)

    if n_ref_actual < 3:
        print(f"  Too few val samples for reference panel ({n_val}). Falling back to standard.")
        return apply_combat_to_splits(split_dict, dataset_name, datasets_dict, **kwargs)

    rng = np.random.RandomState(ref_seed)
    ref_idx = rng.choice(n_val, n_ref_actual, replace=False)
    eval_idx = np.setdiff1d(np.arange(n_val), ref_idx)

    X_ref = X_val_arr[ref_idx]
    patnos_ref = [val_patnos[i] for i in ref_idx]
    patnos_val_eval = [val_patnos[i] for i in eval_idx]

    print(f"  STRICT MODE: {len(train_patnos)} train + {n_ref_actual} ref panel")

    if dataset_name == "Combined":
        print("    Combined + strict mode: using standard approach")
        return apply_combat_to_splits(split_dict, dataset_name, datasets_dict, **kwargs)

    combat_preprocessors = kwargs.get("combat_preprocessors", {})
    combat = combat_preprocessors[dataset_name]
    X_fit = np.vstack([X_train_arr, X_ref])
    patnos_fit = train_patnos + patnos_ref
    combat.fit(X_fit, patnos=patnos_fit)

    if not combat.combat_fitted_:
        print("  ComBat could not fit. Using uncorrected data.")
        return s

    feature_cols = list(s["X_train"].columns)
    s["X_train"] = pd.DataFrame(
        combat.transform(X_train_arr, patnos=train_patnos),
        index=s["X_train"].index, columns=feature_cols)
    s["X_test"] = pd.DataFrame(
        combat.transform(X_test_arr, patnos=test_patnos),
        index=s["X_test"].index, columns=feature_cols)

    X_val_eval = X_val_arr[eval_idx]
    s["X_val"] = pd.DataFrame(
        combat.transform(X_val_eval, patnos=patnos_val_eval),
        index=s["X_val"].index[eval_idx], columns=feature_cols)
    s["y_val"] = s["y_val"].iloc[eval_idx]
    s["_combat_ref_patnos"] = patnos_ref
    s["_combat_n_ref"] = n_ref_actual

    print(f"  Validation set reduced: {n_val} -> {len(eval_idx)} samples")

    if "X_tt" in s:
        tt_patnos = list(s["groups_tt"])
        X_tt_arr = np.nan_to_num(np.array(s["X_tt"], dtype=np.float64), nan=0.0)
        s["X_tt"] = pd.DataFrame(
            combat.transform(X_tt_arr, patnos=tt_patnos),
            index=s["X_tt"].index, columns=feature_cols)

    return s


def run_post_combat_diagnostics(splits, splits_uncorrected, datasets):
    """Variance checks and PCA-based batch assessment after ComBat."""
    print("\n" + "=" * 60)
    print("POST-COMBAT DIAGNOSTICS")
    print("=" * 60)

    for name in datasets:
        s = splits[name]
        train_var = np.var(np.nan_to_num(s["X_train"].values, nan=0.0), axis=0)
        n_zero_var = (train_var < 1e-10).sum()
        n_total = len(train_var)
        flag = "!" if n_zero_var > n_total * 0.5 else "ok"
        print(f"\n  {name}: {n_zero_var}/{n_total} features with ~zero variance [{flag}]")

    for name in ["Proteomics only", "RNA-seq only"]:
        if name not in splits:
            continue
        s = splits[name]
        s_orig = splits_uncorrected[name]

        print(f"\n  {name} — PCA batch assessment:")
        X_all = pd.concat([s["X_train"], s["X_test"], s["X_val"]])
        X_all_orig = pd.concat([s_orig["X_train"], s_orig["X_test"], s_orig["X_val"]])

        all_patnos = (
            list(s["groups_train"]) +
            datasets[name].loc[s["X_test"].index, "PATNO"].tolist() +
            datasets[name].loc[s["X_val"].index, "PATNO"].tolist()
        )
        cohort_labels = np.array([
            "PDBP" if str(p).startswith("PD-") else "PPMI" for p in all_patnos
        ])

        for tag, X_pca_data in [("Before ComBat", X_all_orig), ("After ComBat", X_all)]:
            X_arr = np.nan_to_num(X_pca_data.values, nan=0.0)
            var = np.var(X_arr, axis=0)
            keep = var > 1e-10
            if keep.sum() < 5:
                print(f"    {tag}: too few non-constant features for PCA")
                continue
            X_scaled = _SS().fit_transform(X_arr[:, keep])
            n_comp = min(5, X_scaled.shape[0], X_scaled.shape[1])
            pcs = _PCA(n_components=n_comp, random_state=RANDOM_STATE).fit_transform(X_scaled)

            groups_kw = [pcs[cohort_labels == c, 0] for c in ["PDBP", "PPMI"]]
            if all(len(g) >= 5 for g in groups_kw):
                stat, pval = _kruskal_test(*groups_kw)
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                print(f"    {tag}: PC1 ~ cohort: KW p={pval:.2e} {sig}")

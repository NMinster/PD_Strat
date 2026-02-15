"""
§12 — Publication-quality figures for the PD-Deep Precision Suite.

Generates scatter plots, K-selection curves, confounder bars,
calibration bars, latent PCA/t-SNE, cluster boxplots, IG bars,
cluster heatmaps, severity-vs-UPSIT, and modality coverage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import FIG, TAB, SEED
from .utils import spearman_np, summary


def run_figures(
    # Core predictions & targets
    oof_pred, y_tr, train_idx_y,
    test_pred, test_idx, test_idx_omics,
    y_all, prot_ok_test,
    # Subtyping results
    K, labs_trpd, Zs_trpd,
    pd_ids_clean, y_pd_clean,
    # Optional data
    upsit_pd_train=None,
    upsit_te=None, rho_test_upsit=np.nan,
    chosen_ser=None, mte=None, best_rho=np.nan,
    # Modality masks
    Mr=None, Mp=None,
    # Z-score DataFrames for cluster heatmaps
    z_rna=None, z_prot=None,
    d_rna=0, d_prot=0,
    SIG_DIR=None,
    has_any_omics=None,
):
    """Generate all publication-style figures. Failures are caught and logged."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"figure.dpi": 300})
    except ImportError:
        print("[Figures] matplotlib not available — skipping all figures")
        return

    try:
        # ── OOF scatter ──────────────────────────────────────────────
        mask_oof = np.isfinite(oof_pred) & np.isfinite(y_tr)
        if mask_oof.any():
            plt.figure(figsize=(4, 4))
            plt.scatter(y_tr[mask_oof], oof_pred[mask_oof], s=6, alpha=0.3)
            plt.xlabel("UPDRS (true)")
            plt.ylabel("Severity (OOF)")
            plt.title(f"OOF \u03c1={spearman_np(oof_pred, y_tr):.3f}")
            m1 = min(y_tr[mask_oof].min(), oof_pred[mask_oof].min())
            m2 = max(y_tr[mask_oof].max(), oof_pred[mask_oof].max())
            plt.plot([m1, m2], [m1, m2], linewidth=1)
            plt.tight_layout()
            plt.savefig(FIG / "oof_scatter.png", dpi=300)
            plt.close()

        # ── TEST raw scatter (omics-only) ────────────────────────────
        if (test_idx_omics.size > 0 and test_pred.size > 0
                and has_any_omics is not None):
            y_te = y_all[test_idx_omics]
            mask_te_full = (np.isfinite(test_pred)
                           & np.isfinite(y_all[test_idx]))
            mask_te = np.zeros_like(mask_te_full, dtype=bool)
            mask_te[has_any_omics[test_idx]] = (
                mask_te_full[has_any_omics[test_idx]])
            if mask_te.any():
                plt.figure(figsize=(4, 4))
                plt.scatter(y_all[test_idx][mask_te],
                            test_pred[mask_te], s=6, alpha=0.3)
                plt.xlabel("UPDRS (true)")
                plt.ylabel("Severity (TEST raw)")
                rho_te = spearman_np(
                    test_pred[has_any_omics[test_idx]], y_te)
                plt.title(f"TEST raw (omics-only) \u03c1={rho_te:.3f}")
                m1 = min(y_all[test_idx][mask_te].min(),
                         test_pred[mask_te].min())
                m2 = max(y_all[test_idx][mask_te].max(),
                         test_pred[mask_te].max())
                plt.plot([m1, m2], [m1, m2], linewidth=1)
                plt.tight_layout()
                plt.savefig(FIG / "test_scatter_raw.png", dpi=300)
                plt.close()

        # ── TEST chosen scatter (omics-only) ─────────────────────────
        if (test_idx.size > 0 and chosen_ser is not None
                and has_any_omics is not None):
            y_te = y_all[test_idx]
            mask_te2 = (np.isfinite(chosen_ser)
                        & np.isfinite(y_te)
                        & has_any_omics[test_idx])
            if mask_te2.any():
                plt.figure(figsize=(4, 4))
                plt.scatter(y_te[mask_te2], chosen_ser[mask_te2],
                            s=6, alpha=0.3)
                plt.xlabel("UPDRS (true)")
                plt.ylabel("Severity (TEST chosen)")
                rho_disp = (float(best_rho)
                            if np.isfinite(best_rho) else np.nan)
                plt.title(
                    f"TEST chosen (omics-only) \u2014 \u03c1={rho_disp:.3f}")
                m1 = min(y_te[mask_te2].min(), chosen_ser[mask_te2].min())
                m2 = max(y_te[mask_te2].max(), chosen_ser[mask_te2].max())
                plt.plot([m1, m2], [m1, m2], linewidth=1)
                plt.tight_layout()
                plt.savefig(FIG / "test_scatter_chosen.png", dpi=300)
                plt.close()

        # ── K-grid composite z-sum ───────────────────────────────────
        grid_csv = TAB / "subtype_grid_TRAINPD.csv"
        if grid_csv.exists():
            gdf = pd.read_csv(grid_csv)
            if "zsum" in gdf.columns:
                plt.figure(figsize=(4.5, 3.2))
                Kvals = gdf["K"].values
                plt.plot(Kvals, gdf["zsum"], marker="o")
                plt.xticks(Kvals)
                plt.xlabel("K (clusters)")
                plt.ylabel("Composite z-score (sil+AMI+\u03b7\u00b2)")
                bestK = int(Kvals[np.nanargmax(gdf["zsum"].values)])
                plt.title(f"K-choice={bestK} via z-sum")
                plt.tight_layout()
                plt.savefig(FIG / "subtype_K_zsum.png", dpi=300)
                plt.close()

        # ── Confounders bar chart ────────────────────────────────────
        conf_csv = TAB / f"subtypes_TRAINPD_K{K}_confounders.csv"
        if conf_csv.exists():
            cdf = pd.read_csv(conf_csv)
            plt.figure(figsize=(4.5, 3.2))
            plt.barh(cdf["metric"], cdf["value"])
            plt.xlabel("Effect size")
            plt.title("TRAIN-PD confounders")
            plt.tight_layout()
            plt.savefig(FIG / "trainpd_confounders.png", dpi=300)
            plt.close()

        # ── Calibration bars ─────────────────────────────────────────
        if ("calibration" in summary
                and isinstance(summary["calibration"], dict)):
            cal = summary["calibration"]
            labels, maes, rmses = [], [], []
            for k in ["raw", "coral_rna", "blend_or_raw"]:
                v = cal.get(k)
                if v is None:
                    continue
                labels.append(k)
                maes.append(v["mae"])
                rmses.append(v["rmse"])
            if labels:
                x = np.arange(len(labels))
                plt.figure(figsize=(5, 3.2))
                plt.bar(x - 0.15, maes, width=0.3, label="MAE")
                plt.bar(x + 0.15, rmses, width=0.3, label="RMSE")
                plt.xticks(x, labels)
                plt.ylabel("Error")
                plt.title("Calibration (TEST)")
                plt.legend(frameon=False)
                plt.tight_layout()
                plt.savefig(FIG / "calibration_bars.png", dpi=300)
                plt.close()

        # ── Latent PCA (TRAIN-PD, colored by cluster) ────────────────
        from sklearn.decomposition import PCA

        Z_for_plot = Zs_trpd
        labs_for_plot = labs_trpd

        if (Z_for_plot is not None and Z_for_plot.shape[1] >= 2
                and Z_for_plot.shape[0] >= 10):
            pca = PCA(n_components=2, random_state=SEED).fit(Z_for_plot)
            P = pca.transform(Z_for_plot)
            plt.figure(figsize=(4.2, 4.0))
            for k_ in range(K):
                mk = labs_for_plot == k_
                plt.scatter(P[mk, 0], P[mk, 1], s=8, alpha=0.6,
                            label=f"C{k_}")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title("Latent PCA (TRAIN-PD)")
            plt.legend(frameon=False, ncol=min(K, 3))
            plt.tight_layout()
            plt.savefig(FIG / f"latent_pca_trainpd_K{K}.png", dpi=300)
            plt.close()

        # ── Latent t-SNE (TRAIN-PD, colored by cluster) ──────────────
        if Z_for_plot is not None and Z_for_plot.shape[0] >= 50:
            from sklearn.manifold import TSNE
            perp = int(np.clip(Z_for_plot.shape[0] // 10, 5, 35))
            T = TSNE(n_components=2, perplexity=perp,
                     learning_rate="auto", init="pca",
                     random_state=SEED)
            Tproj = T.fit_transform(Z_for_plot)
            plt.figure(figsize=(4.2, 4.0))
            for k_ in range(K):
                mk = labs_for_plot == k_
                plt.scatter(Tproj[mk, 0], Tproj[mk, 1], s=8, alpha=0.6,
                            label=f"C{k_}")
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.title("Latent t-SNE (TRAIN-PD)")
            plt.legend(frameon=False, ncol=min(K, 3))
            plt.tight_layout()
            plt.savefig(FIG / f"latent_tsne_trainpd_K{K}.png", dpi=300)
            plt.close()

        # ── Cluster boxplots for UPDRS and UPSIT ────────────────────
        def _box_from_groups(values, labs, title, ylab, fname):
            data = [pd.to_numeric(values[labs == k_], errors="coerce")
                    for k_ in range(K)]
            plt.figure(figsize=(4.6, 3.6))
            plt.boxplot(
                [d[~np.isnan(d)] for d in data],
                labels=[f"C{k_}" for k_ in range(K)],
                showfliers=False,
            )
            plt.ylabel(ylab)
            plt.title(title)
            plt.tight_layout()
            plt.savefig(FIG / fname, dpi=300)
            plt.close()

        if labs_trpd is not None and pd_ids_clean is not None:
            _box_from_groups(
                pd.Series(y_pd_clean, index=pd_ids_clean), labs_trpd,
                "UPDRS by cluster (TRAIN-PD)", "UPDRS",
                f"box_updrs_trainpd_K{K}.png",
            )
            if upsit_pd_train is not None and upsit_pd_train.notna().sum() > 0:
                _box_from_groups(
                    upsit_pd_train, labs_trpd,
                    "UPSIT by cluster (TRAIN-PD)", "UPSIT",
                    f"box_upsit_trainpd_K{K}.png",
                )

        # ── IG bars (top 20) ─────────────────────────────────────────
        DEEP = FIG.parent / "deep"

        def _bar_top(series_csv_path, title, fname, top=20):
            if not Path(series_csv_path).exists():
                return
            s = (pd.read_csv(series_csv_path, index_col=0)
                 .iloc[:, 0].sort_values(ascending=False).head(top))
            plt.figure(figsize=(6, 4))
            plt.barh(range(len(s))[::-1], s.values[::-1])
            plt.yticks(range(len(s))[::-1], s.index[::-1], fontsize=7)
            plt.xlabel("Integrated Gradients (abs, mean)")
            plt.title(title)
            plt.tight_layout()
            plt.savefig(FIG / fname, dpi=300)
            plt.close()

        _bar_top(DEEP / "drivers_rna_IG.csv",
                 "Top RNA drivers (IG)", "ig_top_rna.png")
        _bar_top(DEEP / "drivers_proteins_IG.csv",
                 "Top Protein drivers (IG)", "ig_top_prot.png")

        # ── Cluster heatmaps ─────────────────────────────────────────
        def _cluster_heatmap(Z_df, feature_rank_csv, outname,
                             per_mod="RNA", max_feats_per_cluster=10):
            if not Path(feature_rank_csv).exists() or Z_df.empty:
                return
            feat_sel = []
            for k_ in range(K):
                mod_tag = "RNA" if per_mod == "RNA" else "PROT"
                base = f"K{K}_cluster{k_}_{mod_tag}"
                rnk_path = (SIG_DIR / f"{base}.rnk.csv"
                            if SIG_DIR else None)
                if rnk_path is None or not rnk_path.exists():
                    continue
                rnk = (pd.read_csv(rnk_path, index_col=0)
                       .iloc[:, 0].sort_values(ascending=False))
                take = [f for f in rnk.index
                        if f in Z_df.columns][:max_feats_per_cluster]
                feat_sel.extend(take)
            feat_sel = list(dict.fromkeys(feat_sel))
            if len(feat_sel) == 0:
                return
            M = []
            for k_ in range(K):
                in_k = (Z_df.reindex(pd_ids_clean[labs_trpd == k_])
                        [feat_sel].astype(float))
                with np.errstate(invalid="ignore"):
                    mvals = (np.ma.masked_invalid(in_k.values)
                             .mean(axis=0).filled(np.nan))
                    M.append(mvals)
            plt.figure(figsize=(max(5, len(feat_sel) * 0.22),
                                2.2 + 0.25 * K))
            im = plt.imshow(M, aspect="auto", interpolation="nearest")
            plt.colorbar(im, fraction=0.025, pad=0.02)
            plt.yticks(range(K), [f"C{k_}" for k_ in range(K)])
            plt.xticks(range(len(feat_sel)), feat_sel,
                       rotation=90, fontsize=6)
            plt.title(f"{per_mod} mean z per cluster (selected features)")
            plt.tight_layout()
            plt.savefig(FIG / outname, dpi=300)
            plt.close()

        if d_rna > 0 and z_rna is not None:
            _cluster_heatmap(
                z_rna, DEEP / "drivers_rna_IG.csv",
                f"cluster_heatmap_rna_K{K}.png", per_mod="RNA")
        if d_prot > 0 and z_prot is not None:
            _cluster_heatmap(
                z_prot, DEEP / "drivers_proteins_IG.csv",
                f"cluster_heatmap_prot_K{K}.png", per_mod="PROT")

        # ── Severity vs UPSIT scatter (TEST) ─────────────────────────
        if (np.isfinite(rho_test_upsit) and chosen_ser is not None
                and upsit_te is not None and mte is not None):
            plt.figure(figsize=(4, 4))
            xx = chosen_ser[mte].values.astype(float)
            yy = upsit_te[mte].values.astype(float)
            plt.scatter(xx, yy, s=8, alpha=0.35)
            plt.xlabel("Predicted severity (TEST chosen)")
            plt.ylabel("UPSIT")
            plt.title(
                f"TEST: severity vs UPSIT (\u03c1={rho_test_upsit:.3f})")
            plt.tight_layout()
            plt.savefig(FIG / "test_severity_vs_upsit.png", dpi=300)
            plt.close()

        # ── Modality coverage counts ─────────────────────────────────
        if Mr is not None and Mp is not None:
            r_has = Mr.sum(1) > 0
            p_has = Mp.sum(1) > 0
            cnts = [
                int((r_has & ~p_has).sum()),
                int((~r_has & p_has).sum()),
                int((r_has & p_has).sum()),
            ]
            plt.figure(figsize=(4.3, 3.2))
            plt.bar(["RNA only", "PROT only", "Both"], cnts)
            plt.ylabel("# participants")
            plt.title("Modality coverage")
            plt.tight_layout()
            plt.savefig(FIG / "modality_coverage.png", dpi=300)
            plt.close()

        print(f"[Figures] Saved to {FIG}/")

    except Exception as e:
        print(f"[Figures] WARN: {e}")

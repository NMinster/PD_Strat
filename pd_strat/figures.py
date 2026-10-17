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

from .config import FIG, TAB, ROB, SEED
from .utils import spearman_np, summary


def run_extended_figures():
    """Figures for the validity / progression / panel-reduction packages.

    Reads the CSV tables written by those modules so it can be re-run
    stand-alone after the fact.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 300, "font.size": 8})
    made = []

    # PD-only participant-level scatter (OOF + TEST)
    p = TAB / "predictions_pd_only.csv"
    if p.exists():
        df = pd.read_csv(p)
        df["pid"] = df["participant_id"].astype(str).str.split("-").str[:2].str.join("-")
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
        for ax, split, col in zip(axes, ("TRAIN_OOF", "TEST"), ("#3b6ea5", "#c0504d")):
            s = df[df["split"] == split].groupby("pid")[["pred", "y"]].mean().dropna()
            if len(s) < 5:
                ax.set_visible(False); continue
            ax.scatter(s["y"], s["pred"], s=10, alpha=0.6, color=col, edgecolor="none")
            lim = [min(s["y"].min(), s["pred"].min()), max(s["y"].max(), s["pred"].max())]
            ax.plot(lim, lim, "--", color="grey", lw=0.8)
            ax.set_xlabel("Observed UPDRS (participant mean)")
            ax.set_ylabel("Predicted UPDRS (participant mean)")
            ax.set_title(f"PD only — {split.replace('_', ' ')}  ρ={spearman_np(s['pred'].values, s['y'].values):.3f}  n={len(s)}")
        fig.tight_layout(); fig.savefig(FIG / "pd_only_participant_scatter.png"); plt.close(fig)
        made.append("pd_only_participant_scatter.png")

    # Cross-endpoint forest (PD, participant level)
    p = TAB / "validity_cross_endpoint.csv"
    if p.exists():
        ce = pd.read_csv(p)
        ce = ce[ce["population"] == "PD"]
        if len(ce):
            eps = list(dict.fromkeys(ce["endpoint"]))
            fig, ax = plt.subplots(figsize=(5.2, 0.35 * len(eps) + 1.4))
            for i, split in enumerate(("OOF", "TEST")):
                s = ce[ce["split"] == split].set_index("endpoint").reindex(eps)
                yv = np.arange(len(eps)) + (0.18 if split == "TEST" else -0.18)
                ax.errorbar(s["rho_participant"], yv,
                            xerr=[s["rho_participant"] - s["ci_lo_participant"],
                                  s["ci_hi_participant"] - s["rho_participant"]],
                            fmt="o" if split == "OOF" else "s", ms=4, capsize=2,
                            color="#3b6ea5" if split == "OOF" else "#c0504d", label=split)
            ax.axvline(0, color="grey", lw=0.8, ls="--")
            ax.set_yticks(range(len(eps))); ax.set_yticklabels(eps); ax.invert_yaxis()
            ax.set_xlabel("Spearman ρ with predicted severity (participant level, 95% CI)")
            ax.legend(frameon=False); fig.tight_layout()
            fig.savefig(FIG / "cross_endpoint_forest.png"); plt.close(fig)
            made.append("cross_endpoint_forest.png")

    # Progression: baseline predicted severity vs slope
    frames = [pd.read_csv(TAB / f"progression_baseline_vs_slope_{s}.csv")
              for s in ("TRAIN", "TEST") if (TAB / f"progression_baseline_vs_slope_{s}.csv").exists()]
    if frames:
        fig, axes = plt.subplots(1, len(frames), figsize=(3.6 * len(frames), 3.4), squeeze=False)
        for ax, df in zip(axes[0], frames):
            ax.scatter(df["pred0"], df["slope_per_year"], s=10, alpha=0.6, edgecolor="none")
            m = np.isfinite(df["pred0"]) & np.isfinite(df["slope_per_year"])
            if m.sum() > 5:
                b = np.polyfit(df["pred0"][m], df["slope_per_year"][m], 1)
                xs = np.linspace(df["pred0"].min(), df["pred0"].max(), 20)
                ax.plot(xs, np.polyval(b, xs), color="k", lw=1)
            ax.axhline(0, color="grey", lw=0.6, ls="--")
            ax.set_xlabel("Baseline predicted UPDRS"); ax.set_ylabel("UPDRS slope (points / year)")
            ax.set_title(f"{df['split'].iloc[0]}  ρ={spearman_np(df['pred0'].values, df['slope_per_year'].values):.3f}  n={len(df)}")
        fig.tight_layout(); fig.savefig(FIG / "progression_baseline_vs_slope.png"); plt.close(fig)
        made.append("progression_baseline_vs_slope.png")

    # Nested cumulative-importance curve
    p = ROB / "cumulative_importance.csv"
    if p.exists():
        cu = pd.read_csv(p)
        fig, ax = plt.subplots(figsize=(4.4, 3.2))
        ax.plot(cu["k"], cu["oof_rho"], "o-", ms=3, label="OOF (nested ranking)")
        if "test_rho" in cu:
            ax.plot(cu["k"], cu["test_rho"], "s--", ms=3, label="TEST (descriptive)")
        ks = summary.get("panel_reduction", {}).get("k_star")
        if ks:
            ax.axvline(ks, color="grey", lw=0.8, ls=":", label=f"k* = {ks} (OOF rule)")
        ax.set_xscale("log"); ax.set_xlabel("Number of proteins (k)"); ax.set_ylabel("Spearman ρ")
        ax.legend(frameon=False, fontsize=7); fig.tight_layout()
        fig.savefig(FIG / "cumulative_importance_curve.png"); plt.close(fig)
        made.append("cumulative_importance_curve.png")

    # Stability selection: top 25 inclusion frequency
    p = ROB / "stability_selection.csv"
    if p.exists():
        st = pd.read_csv(p).sort_values("incl_freq_k50", ascending=False).head(25)
        fig, ax = plt.subplots(figsize=(4.6, 0.22 * len(st) + 1))
        ax.barh(range(len(st)), st["incl_freq_k50"], color=np.where(st["w_boot_median"] > 0, "#c0504d", "#3b6ea5"))
        ax.set_yticks(range(len(st))); ax.set_yticklabels(st["protein"], fontsize=6); ax.invert_yaxis()
        ax.axvline(0.8, color="grey", ls="--", lw=0.8)
        ax.set_xlabel("Inclusion frequency in top-50 (bootstrap)  red = higher → worse")
        fig.tight_layout(); fig.savefig(FIG / "stability_selection_top25.png"); plt.close(fig)
        made.append("stability_selection_top25.png")

    # Confirmatory forest: TRAIN vs TEST betas
    p = ROB / "confirmatory_severity.csv"
    if p.exists():
        cf = pd.read_csv(p)
        if "train_beta" in cf and "test_beta" in cf:
            cf = cf.dropna(subset=["train_beta", "test_beta"]).head(40)
            fig, ax = plt.subplots(figsize=(5.0, 0.22 * len(cf) + 1.2))
            yv = np.arange(len(cf))
            ax.errorbar(cf["train_beta"], yv - 0.18, xerr=1.96 * cf["train_se"], fmt="o", ms=3,
                        capsize=1.5, color="#3b6ea5", label="TRAIN (descriptive)")
            ax.errorbar(cf["test_beta"], yv + 0.18, xerr=1.96 * cf["test_se"], fmt="s", ms=3,
                        capsize=1.5, color="#c0504d", label="TEST (replication)")
            ax.axvline(0, color="grey", lw=0.8, ls="--")
            ax.set_yticks(yv); ax.set_yticklabels(cf["protein"], fontsize=6); ax.invert_yaxis()
            ax.set_xlabel("β (UPDRS points per SD protein), 95% CI"); ax.legend(frameon=False, fontsize=7)
            fig.tight_layout(); fig.savefig(FIG / "confirmatory_forest.png"); plt.close(fig)
            made.append("confirmatory_forest.png")

    # Permutation null histogram
    p = ROB / "permutation_null.csv"
    if p.exists():
        pn = pd.read_csv(p)
        fig, ax = plt.subplots(figsize=(3.8, 2.8))
        ax.hist(pn["rho"], bins=20, color="#bbb", edgecolor="white")
        obs = summary.get("oof_spearman")
        if obs is not None:
            ax.axvline(obs, color="#c0504d", lw=1.5, label=f"observed ρ = {obs:.3f}")
            ax.legend(frameon=False, fontsize=7)
        ax.set_xlabel("OOF ρ under label permutation"); ax.set_ylabel("count")
        fig.tight_layout(); fig.savefig(FIG / "permutation_null.png"); plt.close(fig)
        made.append("permutation_null.png")

    print(f"[Figures/extended] {len(made)} saved: {made}")
    return made


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
                and prot_ok_test is not None):
            y_te = y_all[test_idx_omics]
            mask_te = (np.isfinite(test_pred)
                       & np.isfinite(y_all[test_idx])
                       & prot_ok_test)
            if mask_te.any():
                plt.figure(figsize=(4, 4))
                plt.scatter(y_all[test_idx][mask_te],
                            test_pred[mask_te], s=6, alpha=0.3)
                plt.xlabel("UPDRS (true)")
                plt.ylabel("Severity (TEST raw)")
                rho_te = spearman_np(test_pred[prot_ok_test], y_te)
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
                and prot_ok_test is not None):
            y_te = y_all[test_idx]
            mask_te2 = (np.isfinite(chosen_ser)
                        & np.isfinite(y_te)
                        & prot_ok_test)
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
            plt.boxplot([d[~np.isnan(d)] for d in data], showfliers=False)
            plt.xticks(range(1, K + 1), [f"C{k_}" for k_ in range(K)])
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

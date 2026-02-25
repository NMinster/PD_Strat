"""
Prognostic validation: PSI_resid -> 24-month UPDRS & time-to-milestone.

Cross-sectional classification (PD vs HC) does not address whether the
proteomic signature carries prognostic information beyond what baseline
motor severity already provides.  We define:

- PSI (Proteomic Severity Index): predict_proba(X_prot)[:, 1] from the
  trained proteomics-only model.
- PSI_resid: residual after regressing PSI on baseline UPDRS-III,
  isolating proteomic information orthogonal to current motor state.

Tests (PD cases only, PDBP internal cohort):
1. 24-month UPDRS-III regression (OLS)
2. Time-to-milestone Cox proportional hazards
3. Kaplan-Meier by PSI_resid tertiles

Functions
---------
load_clinical_data(clinical_path, output_dir)
build_guid_bridge(clinical_path)
extract_baseline_and_24m(df_clin, target_month, month_window)
compute_time_to_milestone(df_clin, df_bl, milestone_delta)
compute_psi_scores(prot_model, datasets, guid_bridge)
compute_psi_resid(df_psi, df_bl)
run_24m_regression(df_prog, df_outcome, output_dir)
run_cox_analysis(df_prog, df_surv, psi_quantile_split, milestone_delta, output_dir)
run_prognostic_validation(trained_models, datasets, splits, output_dir)
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import (
    OUTPUT_DIR, CLINICAL_PATH, MILESTONE_DELTA, TARGET_MONTH,
    MONTH_WINDOW, PSI_QUANTILE_SPLIT,
)


def _parse_visit_month(v):
    """Extract numeric month from visit name strings like 'M0', 'M12', 'BL'."""
    if pd.isna(v):
        return np.nan
    v = str(v).strip().upper()
    if v in ("BL", "SC", "SCREENING"):
        return 0
    m = re.match(r'[A-Z]*(\d+)', v)
    if m:
        return int(m.group(1))
    return np.nan


def load_clinical_data(clinical_path=CLINICAL_PATH, output_dir=OUTPUT_DIR):
    """Load and standardise longitudinal UPDRS-III data.

    Returns DataFrame with columns [PATNO, visit_month, updrs3_total].
    """
    print("\n--- Loading longitudinal UPDRS-III ---")

    df = None
    paths_to_try = [
        clinical_path,
        "clinical_unified.csv",
        f"{output_dir}/clinical_unified.csv",
    ]
    for p in paths_to_try:
        try:
            df = pd.read_csv(p)
            print(f"  Loaded from: {p}")
            break
        except FileNotFoundError:
            continue
    if df is None:
        raise FileNotFoundError(
            f"Cannot find clinical_unified.csv. Tried:\n  " +
            "\n  ".join(paths_to_try))

    # Standardise column names
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("participant_id", "patno", "subject_id"):
            col_map[c] = "PATNO"
        elif cl in ("visit_month", "event_month", "month",
                     "visit_month_clinical"):
            col_map[c] = "visit_month"
        elif cl in ("part_iii_total", "part3_total", "mds_updrs_part_iii",
                     "updrs3_total", "updrs_iii_total"):
            col_map[c] = "updrs3_total"
        elif "updrs" in cl and ("iii" in cl or "part3" in cl or "3" in cl):
            if "total" in cl or cl.endswith("3") or cl.endswith("iii"):
                col_map[c] = "updrs3_total"
    df.rename(columns=col_map, inplace=True)

    # Strip cohort prefix
    df["PATNO"] = df["PATNO"].astype(str).str.replace(
        r'^[A-Z]+-', '', regex=True)

    # Handle visit_name -> visit_month
    if "visit_month" not in df.columns and "visit_name" in df.columns:
        print("  Converting visit_name -> visit_month")
        df["visit_month"] = df["visit_name"].apply(_parse_visit_month)
        mapped = df["visit_month"].notna().sum()
        print(f"  Parsed {mapped}/{len(df)} visit names successfully")

    # Broader fallback for UPDRS column
    if "updrs3_total" not in df.columns:
        candidates = [c for c in df.columns
                      if any(x in c.lower()
                             for x in ["updrs_3", "updrs_iii",
                                       "mds_updrs_part_iii",
                                       "motor_score", "updrs3",
                                       "part_iii"])]
        if candidates:
            df.rename(columns={candidates[0]: "updrs3_total"}, inplace=True)
            print(f"  Mapped '{candidates[0]}' -> updrs3_total")

    for col in ["PATNO", "visit_month", "updrs3_total"]:
        assert col in df.columns, (
            f"No {col} column found. Available: {list(df.columns)[:20]}")

    df["PATNO"] = df["PATNO"].astype(str)
    df["visit_month"] = pd.to_numeric(df["visit_month"], errors="coerce")
    df["updrs3_total"] = pd.to_numeric(df["updrs3_total"], errors="coerce")
    df = df.dropna(subset=["PATNO", "visit_month", "updrs3_total"])

    print(f"  Clinical data: {len(df)} rows, {df['PATNO'].nunique()} subjects")
    print(f"  Visit months: [{df['visit_month'].min():.0f}, "
          f"{df['visit_month'].max():.0f}]")
    return df


def build_guid_bridge(clinical_path=CLINICAL_PATH):
    """Build GUID -> PATNO bridge from raw clinical file.

    Returns DataFrame with [PATNO_numeric, GUID_clean].
    """
    try:
        raw = pd.read_csv(
            clinical_path,
            usecols=lambda c: c.lower().strip() in (
                "participant_id", "patno", "subject_id", "guid"))
    except (FileNotFoundError, ValueError):
        return pd.DataFrame(columns=["PATNO_numeric", "GUID_clean"])

    id_col = [c for c in raw.columns
              if c.lower().strip() in ("participant_id", "patno",
                                       "subject_id")]
    if not id_col or "GUID" not in [c.upper() for c in raw.columns]:
        return pd.DataFrame(columns=["PATNO_numeric", "GUID_clean"])

    id_col = id_col[0]
    guid_col = [c for c in raw.columns if c.upper() == "GUID"][0]
    raw["PATNO_numeric"] = raw[id_col].astype(str).str.replace(
        r'^[A-Z]+-', '', regex=True)
    raw["GUID_clean"] = raw[guid_col].astype(str).str.strip()
    bridge = raw[["PATNO_numeric", "GUID_clean"]].drop_duplicates()
    print(f"  GUID bridge: {len(bridge)} mappings")
    return bridge


def extract_baseline_and_24m(df_clin, target_month=TARGET_MONTH,
                             month_window=MONTH_WINDOW):
    """Extract baseline and ~24-month UPDRS from longitudinal data.

    Returns (df_bl, df_outcome).
    """
    print("\n--- Extracting baseline & 24-month UPDRS ---")

    # Baseline: earliest visit (month <= 3)
    df_bl = (df_clin[df_clin["visit_month"] <= 3]
             .sort_values("visit_month")
             .groupby("PATNO")
             .first()
             .reset_index()
             .rename(columns={"updrs3_total": "updrs3_baseline"}))

    # 24-month: closest visit to target within window
    df_m24 = df_clin[
        (df_clin["visit_month"] >= target_month - month_window) &
        (df_clin["visit_month"] <= target_month + month_window)
    ].copy()
    df_m24["dist_to_target"] = (df_m24["visit_month"] - target_month).abs()
    df_m24 = (df_m24.sort_values("dist_to_target")
              .groupby("PATNO")
              .first()
              .reset_index()
              .rename(columns={"updrs3_total": "updrs3_m24"}))

    df_outcome = pd.merge(
        df_bl[["PATNO", "updrs3_baseline"]],
        df_m24[["PATNO", "updrs3_m24", "visit_month"]],
        on="PATNO", how="inner",
    )
    df_outcome.rename(columns={"visit_month": "actual_month_m24"},
                      inplace=True)
    df_outcome["updrs3_change"] = (df_outcome["updrs3_m24"] -
                                   df_outcome["updrs3_baseline"])

    print(f"  Subjects with both baseline & ~24m UPDRS: {len(df_outcome)}")
    return df_bl, df_outcome


def compute_time_to_milestone(df_clin, df_bl,
                              milestone_delta=MILESTONE_DELTA):
    """Compute time-to-milestone (first UPDRS increase >= delta).

    Returns df_surv with [PATNO, updrs3_baseline, time_to_milestone, event].
    """
    print("\n--- Computing time-to-milestone ---")

    df_ms = pd.merge(
        df_bl[["PATNO", "updrs3_baseline"]],
        df_clin[["PATNO", "visit_month", "updrs3_total"]],
        on="PATNO", how="inner",
    )
    df_ms = df_ms[df_ms["visit_month"] > 0]
    df_ms["worsened"] = (
        df_ms["updrs3_total"] >= df_ms["updrs3_baseline"] + milestone_delta)

    # First crossing time
    first_event = (df_ms[df_ms["worsened"]]
                   .sort_values("visit_month")
                   .groupby("PATNO")
                   .first()
                   .reset_index()[["PATNO", "visit_month"]]
                   .rename(columns={"visit_month": "time_to_milestone"}))
    first_event["event"] = 1

    # Last observation for censored subjects
    all_last = (df_ms.sort_values("visit_month")
                .groupby("PATNO")
                .last()
                .reset_index()[["PATNO", "visit_month"]]
                .rename(columns={"visit_month": "time_to_milestone"}))
    all_last["event"] = 0

    df_surv = pd.merge(
        df_bl[["PATNO", "updrs3_baseline"]],
        first_event, on="PATNO", how="left",
    )
    df_surv = pd.merge(
        df_surv,
        all_last[["PATNO", "time_to_milestone"]].rename(
            columns={"time_to_milestone": "last_obs_time"}),
        on="PATNO", how="left",
    )

    censored = df_surv["event"].isna()
    df_surv.loc[censored, "time_to_milestone"] = df_surv.loc[
        censored, "last_obs_time"]
    df_surv.loc[censored, "event"] = 0
    df_surv["event"] = df_surv["event"].astype(int)
    df_surv = df_surv.drop(columns=["last_obs_time"])
    df_surv = df_surv.dropna(subset=["time_to_milestone"])

    return df_surv


def compute_psi_scores(prot_model, datasets, guid_bridge=None):
    """Score all subjects with PSI (Proteomic Severity Index).

    Returns DataFrame with [PATNO, pd_label, PSI].
    """
    print("\n--- Computing PSI scores ---")

    df_prot = datasets["Proteomics only"].copy()
    X = df_prot.drop(columns=["PATNO", "pd"])
    psi_scores = prot_model.predict_proba(X)[:, 1]

    df_psi = pd.DataFrame({
        "PATNO": df_prot["PATNO"].values,
        "pd_label": df_prot["pd"].values,
        "PSI": psi_scores,
    })

    # Remap GUID-style IDs
    df_psi["GUID_clean"] = df_psi["PATNO"].astype(str).str.replace(
        r'^[A-Z]+-', '', regex=True)

    if guid_bridge is not None and len(guid_bridge) > 0:
        df_psi = df_psi.merge(guid_bridge, on="GUID_clean", how="left")
        n_mapped = df_psi["PATNO_numeric"].notna().sum()
        print(f"  ID remapping: {n_mapped} matched")
        df_psi["PATNO"] = df_psi["PATNO_numeric"].fillna(df_psi["PATNO"])
        df_psi = df_psi.drop(
            columns=["GUID_clean", "PATNO_numeric"], errors="ignore")
    else:
        df_psi = df_psi.drop(columns=["GUID_clean"])

    print(f"  PSI scored: {len(df_psi)} subjects")
    print(f"  PD mean PSI: "
          f"{df_psi.loc[df_psi['pd_label']==1, 'PSI'].mean():.3f}")
    print(f"  HC mean PSI: "
          f"{df_psi.loc[df_psi['pd_label']==0, 'PSI'].mean():.3f}")
    return df_psi


def compute_psi_resid(df_psi, df_bl):
    """Residualise PSI on baseline UPDRS (PD cases only).

    Returns df_prog with PSI_resid column and OLS model.
    """
    import statsmodels.api as sm

    print("\n--- Residualising PSI on baseline UPDRS ---")

    df_prog = pd.merge(df_psi, df_bl[["PATNO", "updrs3_baseline"]],
                        on="PATNO", how="inner")
    df_prog = df_prog[df_prog["pd_label"] == 1].copy()
    df_prog = df_prog.dropna(subset=["PSI", "updrs3_baseline"])

    print(f"  PD subjects with PSI + baseline UPDRS: {len(df_prog)}")

    X_bl = sm.add_constant(df_prog["updrs3_baseline"])
    model_resid = sm.OLS(df_prog["PSI"], X_bl).fit()
    df_prog["PSI_resid"] = model_resid.resid

    print(f"  PSI ~ baseline UPDRS: R2={model_resid.rsquared:.3f}, "
          f"beta={model_resid.params.iloc[1]:.4f}, "
          f"p={model_resid.pvalues.iloc[1]:.2e}")
    return df_prog, model_resid


def run_24m_regression(df_prog, df_outcome, output_dir=OUTPUT_DIR):
    """Test PSI_resid -> 24-month UPDRS (covariate-adjusted OLS).

    Returns (model_m24, model_change) or None if insufficient data.
    """
    import statsmodels.api as sm
    from scipy.stats import spearmanr

    print("\n--- PSI_resid -> 24-month UPDRS (OLS) ---")

    df = pd.merge(df_prog,
                  df_outcome[["PATNO", "updrs3_m24", "updrs3_change"]],
                  on="PATNO", how="inner")
    n = len(df)
    print(f"  PD subjects with PSI + baseline + 24m UPDRS: {n}")

    if n < 20:
        print(f"  Only {n} subjects -- insufficient for regression (need >=20)")
        return None, None

    # Model 1: updrs3_m24 ~ PSI_resid + updrs3_baseline
    X_ols = sm.add_constant(df[["PSI_resid", "updrs3_baseline"]])
    model_m24 = sm.OLS(df["updrs3_m24"], X_ols).fit()

    print(f"\n  Model: UPDRS-III_24m ~ PSI_resid + UPDRS-III_baseline")
    print(f"  N = {n}, R2 = {model_m24.rsquared:.3f}")
    for var in ["PSI_resid", "updrs3_baseline"]:
        b = model_m24.params[var]
        ci = model_m24.conf_int().loc[var]
        p = model_m24.pvalues[var]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {var:20s}: beta={b:+.3f}  95%CI=[{ci[0]:+.3f}, "
              f"{ci[1]:+.3f}]  p={p:.2e} {sig}")

    # Model 2: UPDRS change
    X_ols2 = sm.add_constant(df[["PSI_resid", "updrs3_baseline"]])
    model_change = sm.OLS(df["updrs3_change"], X_ols2).fit()
    print(f"\n  Model: delta-UPDRS (24m - baseline) ~ PSI_resid + baseline")
    print(f"  N = {n}, R2 = {model_change.rsquared:.3f}")
    for var in ["PSI_resid", "updrs3_baseline"]:
        b = model_change.params[var]
        p = model_change.pvalues[var]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {var:20s}: beta={b:+.3f}  p={p:.2e} {sig}")

    rho, p_rho = spearmanr(df["PSI_resid"], df["updrs3_m24"])
    print(f"\n  Spearman(PSI_resid, UPDRS-III_24m): rho={rho:.3f}, p={p_rho:.2e}")

    # Scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    ax.scatter(df["PSI_resid"], df["updrs3_m24"],
               alpha=0.5, s=25, c="steelblue", edgecolors="white",
               linewidth=0.3)
    x_range = np.linspace(df["PSI_resid"].min(), df["PSI_resid"].max(), 100)
    y_pred = (model_m24.params["const"] +
              model_m24.params["PSI_resid"] * x_range +
              model_m24.params["updrs3_baseline"] *
              df["updrs3_baseline"].mean())
    ax.plot(x_range, y_pred, 'r-', lw=2, alpha=0.8)
    ax.set_xlabel("PSI_resid (proteomic severity | baseline UPDRS)")
    ax.set_ylabel("UPDRS-III at 24 months")
    ax.set_title(f"PSI_resid -> 24m Motor Score\n"
                 f"beta={model_m24.params['PSI_resid']:+.2f}, "
                 f"p={model_m24.pvalues['PSI_resid']:.2e}, rho={rho:.3f}")

    ax = axes[1]
    ax.scatter(df["PSI_resid"], df["updrs3_change"],
               alpha=0.5, s=25, c="darkorange", edgecolors="white",
               linewidth=0.3)
    x_range2 = np.linspace(df["PSI_resid"].min(),
                           df["PSI_resid"].max(), 100)
    y_pred2 = (model_change.params["const"] +
               model_change.params["PSI_resid"] * x_range2 +
               model_change.params["updrs3_baseline"] *
               df["updrs3_baseline"].mean())
    ax.plot(x_range2, y_pred2, 'r-', lw=2, alpha=0.8)
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.set_xlabel("PSI_resid (proteomic severity | baseline UPDRS)")
    ax.set_ylabel("delta-UPDRS-III (24m - baseline)")
    ax.set_title(f"PSI_resid -> 24m Motor Change\n"
                 f"beta={model_change.params['PSI_resid']:+.2f}, "
                 f"p={model_change.pvalues['PSI_resid']:.2e}")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig_psi_resid_24m_updrs.png", dpi=300)
    plt.savefig(f"{output_dir}/fig_psi_resid_24m_updrs.eps",
                format="eps", dpi=300)
    plt.show()
    return model_m24, model_change


def run_cox_analysis(df_prog, df_surv, psi_quantile_split=PSI_QUANTILE_SPLIT,
                     milestone_delta=MILESTONE_DELTA, output_dir=OUTPUT_DIR):
    """Run Cox PH and Kaplan-Meier analysis for time-to-milestone.

    Returns (cph, summary_stats) or None if lifelines unavailable.
    """
    print("\n--- PSI_resid -> Time-to-milestone (Cox PH) ---")

    try:
        from lifelines import CoxPHFitter, KaplanMeierFitter
        from lifelines.statistics import logrank_test
    except ImportError:
        print("  lifelines not installed -- skipping Cox analysis")
        print("  Install with: pip install lifelines")
        return None, None

    df_cox = pd.merge(
        df_prog[["PATNO", "PSI_resid", "updrs3_baseline"]],
        df_surv[["PATNO", "time_to_milestone", "event"]],
        on="PATNO", how="inner",
    )
    df_cox = df_cox.dropna()

    n_cox = len(df_cox)
    n_events = df_cox["event"].sum()
    print(f"  PD subjects for Cox: {n_cox} ({n_events} events, "
          f"{n_cox - n_events} censored)")

    if n_cox < 30 or n_events < 10:
        print(f"  Insufficient data for Cox: {n_cox} subjects, "
              f"{n_events} events")
        return None, None

    cph = CoxPHFitter()
    cox_df = df_cox[["time_to_milestone", "event",
                     "PSI_resid", "updrs3_baseline"]].copy()
    cph.fit(cox_df, duration_col="time_to_milestone", event_col="event")

    print(f"\n  Cox PH: Time-to-motor-milestone ~ PSI_resid + baseline")
    cph.print_summary(
        columns=["coef", "exp(coef)", "se(coef)", "p",
                 "lower 0.95", "upper 0.95"])

    hr_psi = np.exp(cph.params_["PSI_resid"])
    p_psi = cph.summary.loc["PSI_resid", "p"]
    hr_ci_lo = np.exp(cph.confidence_intervals_.loc["PSI_resid"].iloc[0])
    hr_ci_hi = np.exp(cph.confidence_intervals_.loc["PSI_resid"].iloc[1])
    c_index = cph.concordance_index_

    print(f"\n  PSI_resid HR = {hr_psi:.3f} [{hr_ci_lo:.3f}-{hr_ci_hi:.3f}], "
          f"p = {p_psi:.2e}")
    print(f"  Concordance index: {c_index:.3f}")

    # Kaplan-Meier by tertiles
    df_cox["PSI_resid_group"] = pd.qcut(
        df_cox["PSI_resid"], q=psi_quantile_split,
        labels=["Low PSI_resid", "Mid PSI_resid", "High PSI_resid"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    kmf = KaplanMeierFitter()
    colors_km = {"Low PSI_resid": "#4CAF50", "Mid PSI_resid": "#FFC107",
                 "High PSI_resid": "#F44336"}

    for grp in ["Low PSI_resid", "Mid PSI_resid", "High PSI_resid"]:
        mask = df_cox["PSI_resid_group"] == grp
        if mask.sum() >= 3:
            kmf.fit(df_cox.loc[mask, "time_to_milestone"],
                    df_cox.loc[mask, "event"],
                    label=f"{grp} (n={mask.sum()})")
            kmf.plot_survival_function(ax=ax, color=colors_km[grp], lw=2)

    ax.set_xlabel("Months from baseline")
    ax.set_ylabel(f"P(no motor worsening)\n"
                  f"(UPDRS-III < baseline + {milestone_delta})")
    ax.set_title("Kaplan-Meier by PSI_resid Tertile")
    ax.legend(loc="lower left", fontsize=9)

    # Log-rank
    high_mask = df_cox["PSI_resid_group"] == "High PSI_resid"
    low_mask = df_cox["PSI_resid_group"] == "Low PSI_resid"
    if high_mask.sum() >= 3 and low_mask.sum() >= 3:
        lr = logrank_test(
            df_cox.loc[high_mask, "time_to_milestone"],
            df_cox.loc[low_mask, "time_to_milestone"],
            df_cox.loc[high_mask, "event"],
            df_cox.loc[low_mask, "event"])
        ax.text(0.95, 0.95, f"High vs Low:\nlog-rank p={lr.p_value:.2e}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.8))

    # Forest plot
    ax = axes[1]
    summary = cph.summary
    vars_plot = ["PSI_resid", "updrs3_baseline"]
    for i, var in enumerate(vars_plot):
        hr = np.exp(summary.loc[var, "coef"])
        lo = np.exp(summary.loc[var, "coef lower 95%"])
        hi = np.exp(summary.loc[var, "coef upper 95%"])
        p = summary.loc[var, "p"]
        color = "#F44336" if p < 0.05 else "#999999"
        ax.plot([lo, hi], [i, i], color=color, lw=2.5,
                solid_capstyle="round")
        ax.plot(hr, i, 'o', color=color, markersize=8, zorder=5)
        sig_str = f"p={p:.2e}" + (" *" if p < 0.05 else "")
        ax.text(hi + 0.05, i,
                f"HR={hr:.2f} [{lo:.2f}-{hi:.2f}] {sig_str}",
                va="center", fontsize=9)

    ax.axvline(1, color="gray", ls="--", lw=0.8)
    ax.set_yticks(range(len(vars_plot)))
    ax.set_yticklabels(vars_plot)
    ax.set_xlabel("Hazard Ratio (95% CI)")
    ax.set_title("Cox PH: Time to Motor Worsening")
    ax.set_xlim(left=max(0, ax.get_xlim()[0] - 0.1))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig_psi_resid_survival.png", dpi=300)
    plt.savefig(f"{output_dir}/fig_psi_resid_survival.eps",
                format="eps", dpi=300)
    plt.show()

    # PH assumption check
    print("\n  Schoenfeld residuals test (PH assumption):")
    try:
        cph.check_assumptions(cox_df, p_value_threshold=0.05,
                              show_plots=False)
    except Exception as e:
        print(f"    {e}")

    stats = {
        "hr_psi": hr_psi, "p_psi": p_psi,
        "hr_ci_lo": hr_ci_lo, "hr_ci_hi": hr_ci_hi,
        "c_index": c_index, "n_cox": n_cox, "n_events": n_events,
    }
    return cph, stats


def run_prognostic_validation(trained_models, datasets, splits,
                              output_dir=OUTPUT_DIR):
    """Run complete prognostic validation pipeline (Section 7).

    Returns summary DataFrame.
    """
    print("=" * 60)
    print("7. PROGNOSTIC VALIDATION: PSI_resid -> FUTURE MOTOR OUTCOME")
    print("=" * 60)

    # Step 1: Load clinical data
    df_clin = load_clinical_data(output_dir=output_dir)
    guid_bridge = build_guid_bridge()

    # Step 2: Extract baseline & 24-month
    df_bl, df_outcome = extract_baseline_and_24m(df_clin)

    # Step 3: Time-to-milestone
    df_surv = compute_time_to_milestone(df_clin, df_bl)

    # Step 4: PSI scores
    prot_model = trained_models["Proteomics only"]
    df_psi = compute_psi_scores(prot_model, datasets, guid_bridge)

    # Step 5: PSI_resid
    df_prog, model_resid = compute_psi_resid(df_psi, df_bl)

    # Step 6: 24-month regression
    model_m24, model_change = run_24m_regression(df_prog, df_outcome,
                                                  output_dir)

    # Step 7: Cox analysis
    cph, cox_stats = run_cox_analysis(df_prog, df_surv,
                                       output_dir=output_dir)

    # Step 8: Summary table
    print("\n--- Exporting prognostic summary ---")
    rows = []
    rows.append({
        "Analysis": "PSI ~ UPDRS-III_baseline",
        "N": len(df_prog),
        "Statistic": f"R2={model_resid.rsquared:.3f}",
        "beta / HR": f"{model_resid.params.iloc[1]:.4f}",
        "p-value": f"{model_resid.pvalues.iloc[1]:.2e}",
        "Note": "Variance of PSI explained by baseline motor score",
    })

    if model_m24 is not None:
        rows.append({
            "Analysis": "UPDRS-III_24m ~ PSI_resid + baseline",
            "N": int(model_m24.nobs),
            "Statistic": f"R2={model_m24.rsquared:.3f}",
            "beta / HR": f"{model_m24.params['PSI_resid']:+.4f}",
            "p-value": f"{model_m24.pvalues['PSI_resid']:.2e}",
            "Note": "Adjusted for baseline UPDRS-III",
        })
    if model_change is not None:
        rows.append({
            "Analysis": "delta-UPDRS_24m ~ PSI_resid + baseline",
            "N": int(model_change.nobs),
            "Statistic": f"R2={model_change.rsquared:.3f}",
            "beta / HR": f"{model_change.params['PSI_resid']:+.4f}",
            "p-value": f"{model_change.pvalues['PSI_resid']:.2e}",
            "Note": "Change score analysis",
        })
    if cox_stats is not None:
        rows.append({
            "Analysis": f"Cox PH: Time to UPDRS +{MILESTONE_DELTA}",
            "N": f"{cox_stats['n_cox']} ({cox_stats['n_events']} events)",
            "Statistic": f"C-index={cox_stats['c_index']:.3f}",
            "beta / HR": (f"HR={cox_stats['hr_psi']:.3f} "
                          f"[{cox_stats['hr_ci_lo']:.3f}-"
                          f"{cox_stats['hr_ci_hi']:.3f}]"),
            "p-value": f"{cox_stats['p_psi']:.2e}",
            "Note": "PSI_resid adjusted for baseline UPDRS-III",
        })

    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(f"{output_dir}/table_psi_resid_prognostic.csv",
                      index=False)
    print(df_summary.to_string())
    print(f"\nSaved: table_psi_resid_prognostic.csv")
    return df_summary

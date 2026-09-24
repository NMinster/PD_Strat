# PD_Strat — PD-Deep Precision Suite v2.1

Proteomics-based severity prediction and molecular subtyping of Parkinson's
disease using the AMP-PD release (Olink Explore plasma panels + MDS-UPDRS).

The pipeline trains on PPMI (`PP-` participants), evaluates externally on
PDBP (`PD-` participants), and produces:

* a **RidgeSVD severity model** (MDS-UPDRS total) with participant-grouped
  cross-validation, external TEST evaluation, recalibration, bootstrap CIs and
  a participant-level primary estimand;
* an **unsupervised severity index (MSI_U)** and **GMM molecular subtypes**;
* confounding and robustness audits (seed variance, permutation control, panel
  ablation, coefficient stability, missingness diagnostics, confirmatory
  protein models);
* publication figures and a human-readable **`results/SUMMARY_REPORT.md`**.

---

## 1. Quick start (Windows, Anaconda Prompt)

```bat
:: 1. unzip the project, then open "Anaconda Prompt" and go to the folder
cd C:\path\to\PD_Strat

:: 2. create the environment once (≈2 min)
conda env create -f environment.yml
conda activate pd_strat

:: 3. point the pipeline at your data (edit config.yaml, or pass the flag)
python run.py --data_dir S:/AMP-PD

:: first pass without the slow robustness package (~minutes instead of ~1 h)
python run.py --data_dir S:/AMP-PD --skip_robustness

:: regenerate just the summary report from a finished run
python run.py --report_only
```

`config.yaml` in the project folder already contains `data_dir: "S:/AMP-PD"`,
so plain `python run.py` works once the file paths there match your drive.
Use forward slashes (`S:/AMP-PD`) or quote backslashes in YAML.

Everything printed to the console is also appended to `results/run_log.txt`.

If you prefer `pip`: `pip install -r requirements.txt` (Python ≥ 3.10).

---

## 2. Expected input files

All files are read from `data_dir` (default `S:/AMP-PD`). Names below are the
AMP-PD 2023 v4 release names; override any of them under `clinical_files:` /
`proteomics_panels:` in `config.yaml`.

| Purpose | File |
|---|---|
| MDS-UPDRS I–IV | `releases_2023_v4release_1027_clinical_MDS_UPDRS_Part_{I,II,III,IV}.csv` |
| UPSIT (smell) | `releases_2023_v4release_1027_clinical_UPSIT.csv` |
| Demographics | `Demographics.csv` |
| Case / control | `releases_2023_v4release_1027_amp_pd_case_control.csv` |
| Olink plasma panels | `releases_2023_v4release_1027_proteomics-PLA-PPEA-D03_olink-explore_protein-expression_PLA-PPEA-D03_{oncology,neurology,inflammation,cardiometabolic}.csv` |
| RNA (optional) | `releases_2023_v4release_1027_rnaseq-WB-RWTS_salmon_quantification_matrix.genes.tsv` |

Set `proteomics_tissue: "CSF"` in `config.yaml` to use the CSF panels instead
of plasma.

### 2a. Clinical assembly (`pd_strat/assemble_clinical.py`)

On the first run the pipeline builds `results/tables/clinical_unified.csv`
(one row per participant-visit) from the raw files. Column names are matched
by pattern and every resolution is printed, e.g.

```
[UPDRS-III] id='participant_id', score='mds_updrs_part_iii_summary_score'
[UPDRS-III] medication-state col='upd23b_clinical_state_on_medication' (960 OFF rows) -> preferring OFF per visit
```

Assembly rules (all switchable in `config.yaml`):

* `updrs_total = Part I + II + III + IV`; Parts I–III are required at a visit.
* `updrs3_prefer_off_state: true` — when a visit has both ON and OFF Part III
  exams, the OFF exam is used.
* `updrs_part_iv_missing_as_zero: true` — a missing Part IV is treated as 0.
* `upsit_fill_from_baseline: true` — a participant's earliest UPSIT is carried
  forward to later visits (UPSIT is mostly collected at baseline).
* Cohort: `PP-` → TRAIN, `PD-` → TEST; other studies (`BF-`, `HB-`, `LB-`,
  `SU-`, `SY-`) are dropped unless `keep_other_cohorts: true`.
* Case/control comes from `case_control_other_latest` (falls back to baseline);
  healthy controls in TRAIN anchor the proteomics z-scores.

Subsequent runs reuse the file; pass `--force_assembly` to rebuild it.

### 2b. RNA (optional, off by default)

Uncomment `rna_path:` in `config.yaml` to enable the RNA modality and
protein + RNA late fusion. The loader accepts either samples-as-rows CSV or the
AMP-PD genes × samples salmon TSV (it transposes, maps sample IDs to
participants, and applies `log1p` automatically). The 3 GB gene matrix needs
roughly 16 GB of free RAM and several minutes to load.

---

## 3. Outputs

```
results/
├── SUMMARY_REPORT.md          <- human-readable summary of the whole run
├── run_log.txt                <- full console log (appended per run)
├── tables/
│   ├── clinical_unified.csv   <- assembled clinical table
│   ├── z_prot.csv             <- HC-anchored protein z-scores (cached)
│   ├── summary.json           <- every metric in machine-readable form
│   ├── sample_flow_table.csv, all_severity_metrics.csv, bootstrap_ci.csv,
│   │   participant_level_*.csv, severity_bands.csv, decile_analysis_*.csv,
│   │   msi_U_*.csv, subtype_grid_TRAINPD.csv, subtypes_TRAINPD_K*_...
├── figures/                   <- PNGs (OOF/TEST scatter, latent PCA/t-SNE, ...)
└── robustness/                <- per-analysis CSV/JSON from the robustness package
```

`SUMMARY_REPORT.md` is printed at the end of every run and covers: run
configuration, sample flow, OOF/TEST metrics for every candidate model, the
selected primary model, calibration, bootstrap CIs, participant-level
estimand, incremental gain over covariates, severity-band classification,
MSI_U, subtype K-selection and cluster summary, confounding audit, robustness
highlights, and the list of files written.

### Validity, progression and panel-reduction packages

| Report section | Question it answers | Main outputs |
|---|---|---|
| 3h Severity vs diagnosis | Is the score graded severity within PD, or PD-vs-HC discrimination? | `validity_severity_vs_diagnosis.csv`, `predictions_pd_only.csv`, `figures/pd_only_participant_scatter.png` |
| 3i Extended covariates | Does age / sex / site / disease duration / Part-III medication state explain it? | `validity_extended_covariates.csv` |
| 3j Cross-endpoint | UPDRS I–IV, UPSIT, Hoehn & Yahr, DaTSCAN SBR | `validity_cross_endpoint.csv`, `figures/cross_endpoint_forest.png` |
| 3k Error structure | Bias by severity / visit bin, heteroscedasticity, range shift | `validity_error_stratification.csv` |
| 3l Target recalibration | Cross-fitted (participant-grouped) recalibration inside TEST | `validity_target_recalibration.csv` |
| 3m Model comparison | Paired participant-level bootstrap Δρ with CI | `validity_model_comparison.csv` |
| 5b Progression | Baseline proteomic severity vs UPDRS slope, mixed models, Cox | `progression_*.csv`, `figures/progression_baseline_vs_slope.png` |
| 7 Robustness | Permutation null with empirical p (n = `n_permutations`) | `robustness/permutation_null.csv`, `figures/permutation_null.png` |
| 7b Panel reduction | Stability selection; nested cumulative curve with k* chosen on OOF | `robustness/stability_selection.csv`, `robustness/cumulative_importance.csv`, `robustness/reduced_panel_k_star.csv` |
| 7c Confirmatory | Locked 40 proteins: TEST replication of severity and protein×time models | `robustness/confirmatory_severity.csv`, `robustness/confirmatory_progression.csv`, `figures/confirmatory_forest.png` |

| 5c Discovery benchmark | Does the baseline proteome predict *future* change better than baseline clinical scoring? Nested CV over a model zoo × feature sets × progression targets (incl. DaTSCAN putamen SBR at baseline and its annual change); TEST once; trial-enrichment curve | `discovery_grid.csv`, `discovery_best.csv`, `discovery_enrichment.csv`, `figures/fig_discovery_*.png`, `fig_trajectories_by_tertile.png`, `fig_km_*.png`, `fig_trial_enrichment.png` |
| 5d Within-person coupling | Does the proteomic score *track* a person's own change (monitoring biomarker), separately from between-person differences? Mixed model with person-mean-centred score (within β vs between β), consecutive-visit Δ correlation, slope-vs-slope, per-protein replication | `longitudinal_coupling.csv`, `longitudinal_pairs.csv`, `longitudinal_protein_coupling.csv`, `figures/fig_within_person_coupling.png` |

### Discovery benchmark (§14) — how to read it honestly

`discovery.py` fits ~9 regression / 6 classification model families (ridge-on-SVD,
elastic net, PLS, RBF kernel ridge, SVR, random forest, extra trees, gradient
boosting, MLP; logistic / SVC / tree ensembles) on three feature sets —
**clinical only** (age, sex, disease duration, baseline UPDRS total & III,
H&Y, UPSIT, levodopa: what a neurologist knows at baseline), **proteomics
only**, and **both** — for four clinical targets: baseline severity, UPDRS slope,
24-month change, fast-progressor status (plus Cox when time-to-event data
exist), and two objective imaging targets when DaTSCAN is present: baseline
putamen SBR (scan within ±6 months of the baseline sample) and annualised
putamen SBR change (≥2 scans spanning ≥12 months). The imaging targets are
rater-independent, so a proteomic Δ over clinical there cannot be dismissed as
rater noise; in AMP-PD they are PPMI-only (PDBP has no DaTSCAN), so they are
nested-CV evidence without an external test. Everything is repeated nested CV on TRAIN; the single OOF-selected
proteomic configuration per target is evaluated once on TEST with a paired
bootstrap Δ against the clinical model and a permutation p. The number of
configurations tested is printed beside every result. **Only a TEST Δ whose
CI excludes zero supports an "adds to clinical scoring" claim.**

### PPMI-native Olink releases (Projects 9000, 293, 277, 314, 318, 214)

The loader reads PPMI's own layout (`PATNO`, `EVENT_ID`, `UNIPROT`, `ASSAY`,
`QC_WARNING`, `NPX`; CSV or parquet or xlsx) as well as the AMP-PD harmonised
files. PATNOs become `PP-<patno>` so they join the AMP-PD clinical tables;
`EVENT_ID` codes (BL, V04, …) map to months on the PPMI schedule; the `ASSAY`
column supplies gene symbols. A panel may list several files: later files win
for the same (participant, visit, protein), so PDBP can come from AMP-PD and
PPMI from the bridged Project 9000 release:

```yaml
proteomics_panels:
  PLA:
    oncology:        [releases_2023_v4release_1027_proteomics-PLA-PPEA-D03_olink-explore_protein-expression_PLA-PPEA-D03_oncology.csv,
                      PPMI_Project_9000_Plasma_ONC_NPX_23Sep2026.csv]
    neurology:       [..._PLA-PPEA-D03_neurology.csv,       PPMI_Project_9000_Plasma_NEURO_NPX_23Sep2026.csv]
    inflammation:    [..._PLA-PPEA-D03_inflammation.csv,    PPMI_Project_9000_Plasma_INF_NPX_23Sep2026.csv]
    cardiometabolic: [..._PLA-PPEA-D03_cardiometabolic.csv, PPMI_Project_9000_Plasma_Cardio_NPX_23Sep2026.csv]
  CSF:
    oncology:        [..._CSF-PPEA-D03_oncology.csv,        PPMI_Project_9000_CSF_ONC_NPX_23Sep2026.csv]
    neurology:       [..._CSF-PPEA-D03_neurology.csv,       PPMI_Project_9000_CSF_NEU_NPX_23Sep2026.csv]
    inflammation:    [..._CSF-PPEA-D03_inflammation.csv,    PPMI_Project_9000_CSF_INF_NPX_23Sep2026.csv]
    cardiometabolic: [..._CSF-PPEA-D03_cardiometabolic.csv, PPMI_Project_9000_CSF_Cardio_NPX_23Sep2026.csv]
```

`--tissue CSF` / `--tissue PLA+CSF` pick the matching block. A single-file
release with a panel/block column (Olink Explore HT, Project 293 plasma or 277
CSF) is one entry; it is split into panels by that column, and
`prot_feature_cap` should be raised (e.g. 5000):

```yaml
proteomics_panels:
  PLA:
    explore_ht: ppmi_proj293_plasma_screened_extended_npx_20251121.parquet
prot_feature_cap: 5000
```

Explore HT and Explore 1536 are different assays; do not mix them across
cohorts inside one panel unless the overlap is what you want to model.

Comparator biomarkers from a long Olink table (Target 48 Neurodegeneration,
NULISA export) are picked by assay:

```yaml
extra_biomarkers:
  - {name: nfl,  file: PPMI_Project_318_Plasma_23Sep2026.csv, assay: NEFL}
  - {name: gfap, file: PPMI_Project_318_Plasma_23Sep2026.csv, assay: GFAP}
```

Before wiring a new file, look at it:

```bat
python -m pd_strat.inspect_file "S:/AMP-PD/ppmi_proj293_plasma_screened_extended_npx_20251121.parquet"
```

which prints columns, dtypes, examples, and how the pipeline would interpret
the file (layout, participants, proteins, visits, QC values). Parquet needs
`pyarrow`, xlsx needs `openpyxl` (both in `environment.yml`; `conda env update -f environment.yml`).

Participants that are new in Project 222 / PPMI LITE and absent from the AMP-PD
v4 clinical tables have no UPDRS row to join and are reported as unmatched by
`[Align]`; bringing them in requires PPMI's own clinical exports.

### Plasma vs CSF vs combined

```bat
python run.py                      :: plasma      -> results/
python run.py --tissue CSF         :: CSF         -> results_CSF/
python run.py --tissue PLA+CSF     :: both compartments as one feature set (PLA:/CSF: prefixes) -> results_PLA_CSF/
python -m pd_strat.compare_runs results results_CSF results_PLA_CSF --labels plasma csf both
```

`compare_runs` intersects the PD participants present in every run and reports
within-PD ρ on that common set with paired bootstrap Δ (fewer participants
have CSF, so the headline numbers of separate runs are not comparable), plus
each run's discovery-benchmark deltas and the overlap of top proteins
(`results_comparison/run_comparison.md`, `fig_run_comparison.png`).

The whole zoo runs in minutes on CPU at AMP-PD sample sizes (≈100–200 PD
participants per cohort); a GPU is not needed and deep networks would only
overfit. If you scale to the newer PPMI Olink Explore 3072 release (thousands
of samples) or add CSF, revisit that.

See `docs/manuscript_review.md` for how these map onto the manuscript and
`docs/TRIPOD_AI_checklist.md` for the reporting checklist.

---

## 4. Command-line flags

```
python run.py --help

  --config PATH        YAML config (default: config.yaml)
  --data_dir PATH      raw AMP-PD folder (overrides config.yaml)
  --out_dir PATH       where results/ is written
  --skip_robustness    skip the slow robustness package
  --skip_assembly      reuse an existing clinical_unified.csv (never rebuild)
  --force_assembly     rebuild clinical_unified.csv
  --no_rna             disable the RNA modality even if rna_path is set
  --report_only        regenerate SUMMARY_REPORT.md from tables/summary.json
```

---

## 5. Runtime

On the full AMP-PD plasma data (≈4 000 participant-visits, 1 463 proteins):

| Step | Approx. time |
|---|---|
| assembly + proteomics loading (first run) | 2–5 min |
| severity model, calibration, CIs, subtyping | 5–15 min |
| robustness package (`boot_B`, `coef_boot_B`, permutations) | 30–90 min |

Lower `boot_B` / `coef_boot_B` in `config.yaml` for a faster exploratory run.

---

## 6. Troubleshooting

* **`Required AMP-PD file not found`** — check `data_dir` and the file names
  under `clinical_files:` / `proteomics_panels:`; the message shows the exact
  path that was tried.
* **`Could not resolve '...' column`** — a release file uses an unexpected
  header; the message lists the available columns. Add a pattern in
  `assemble_clinical.py` or rename the column.
* **`No proteomics features loaded`** — the panel files were not found (see the
  `[WARN] Panel ... not found` lines above it).
* **Stale cache** — `z_prot.csv`, `feature_manifest.json` and
  `clinical_unified.csv` are cached in `results/tables/`. Delete them (or the
  whole `results/` folder) after changing data or QC settings.
* **Unicode errors when redirecting output** — `run.py` forces UTF-8; if you
  run `python -m pd_strat.main` directly set `set PYTHONIOENCODING=utf-8`.

---

## 7. Project layout

```
run.py                     entry point (logging, UTF-8 console)
config.yaml                run configuration
environment.yml            conda environment   (requirements.txt for pip)
pd_strat/
  config.py                paths, CLI flags, locked analysis plan
  assemble_clinical.py     raw AMP-PD files -> clinical_unified.csv
  data_loading.py          clinical / Olink / RNA loading, HC z-scoring
  features.py              aligned matrices, panel-aware & monolithic SVD
  severity_model.py        CV setup, OOF models, TEST evaluation, covariates
  calibration.py           recalibration, bootstrap CIs, participant-level
  clinical_analysis.py     severity bands, decile / quintile analysis
  rna_pipeline.py          RNA severity + late fusion
  subtyping.py             MSI_U index + GMM subtypes
  confounding.py           sex / site / age audit
  robustness.py            robustness package
  figures.py               publication figures
  report.py                SUMMARY_REPORT.md
  main.py                  orchestrator
Notebook.ipyn              original single-file version (reference only)
```

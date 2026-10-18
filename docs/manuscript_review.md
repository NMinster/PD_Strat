# Reviewer-style critique and revision plan

**Manuscript:** *Externally validated plasma-proteomic severity estimation in Parkinson's disease* (draft `manuscript_stratification_SJ_7-20-26.docx`)
**Purpose of this document:** anticipate what referees at a Nature Communications / Brain / Movement Disorders–tier journal will raise, and connect every fix to a concrete output of the pipeline (`python run.py`). Section numbers in brackets — e.g. **[3h]** — refer to sections of `results/SUMMARY_REPORT.md`.

---

## 1. Verdict in one paragraph

The core design (train in PPMI, lock, test in PDBP; participant-grouped CV; participant-level estimand; HC-anchored normalization done inside TRAIN) is genuinely stronger than most of the PD proteomics literature and is the manuscript's real contribution. However, in its current form the paper is vulnerable on four points that a competent referee will find within an hour: (i) the "severity" model is trained on a TRAIN set that contains healthy controls, so a large fraction of ρ may be diagnosis rather than graded severity; (ii) roughly half of the Results could not be regenerated from the deposited code; (iii) two headline numbers are produced with test-set information (the k=10 panel) or with circular selection (Bonferroni on the training cohort that chose the proteins); and (iv) the draft is incomplete (no abstract, empty sections, truncated sentences). All four are fixable, and the fixes mostly *add* interesting results rather than remove them.

---

## 2. Blocking issues (reject / major revision if unaddressed)

### B1. Severity or diagnosis? Healthy controls are inside the severity model

*Evidence in the manuscript.* Fig. 1a shows a dense cluster of ~80 participants with observed UPDRS 0–10; Methods says HC-anchored z-scoring used ≥30 TRAIN controls, so controls carry proteomics; nowhere is the severity model stated to be PD-only. The pipeline confirms it: `train_idx = all TRAIN rows with UPDRS` (PD + HC).

*Why it matters.* A model that mostly separates HC (UPDRS≈3) from PD (UPDRS≈25) will have ρ≈0.5 with no ability to grade severity *within* PD. That is a diagnostic classifier, which the Introduction explicitly says is *not* the contribution.

*Fix (implemented).* **[3h]** reports, for OOF and TEST: ρ within PD (row- and participant-level), ρ within HC, AUROC of the same score for PD-vs-HC, and a full **PD-only refit** with bootstrap CIs (`validity_severity_vs_diagnosis.csv`, `predictions_pd_only.csv`, `figures/pd_only_participant_scatter.png`).

*Manuscript action.*
- Decide the primary population *before* looking at the real numbers and state it in Methods → "Cohort definition". Recommendation: make **PD-only the primary analysis** (`severity_population: pd_only` in `config.yaml`) and report the mixed population as a sensitivity. If within-PD ρ collapses (< 0.3), the paper's framing must change to "a proteomic axis that separates PD from HC and tracks severity weakly"; better to know now.
- Add one paragraph to Results ("Severity signal is not reducible to diagnosis") quoting within-PD ρ with CI, AUROC PD-vs-HC, and the PD-only refit.
- Table 1 must give n for PD and HC separately in each split.

### B2. Results that cannot be regenerated from the deposited code

Error stratification, Breusch–Pagan, Huber, cross-endpoint correlations, progression slopes and the "incremental prognostic value" model, protein × time mixed models, TEST replication of the confirmatory models, cumulative-importance curve, stability selection (B=200), GSEA/ORA and pQTL anchoring were not in the repository, and the three-stage feature selection described in Methods (MAD filter, |r|>0.95 pruning, cap 1,168) was not what the code did (variance top-1,463). Reviewers increasingly run the code.

*Fix (implemented).* Everything except GSEA/ORA and pQTL is now a stage of the locked pipeline (**[3i–3m, 5b, 7, 7b, 7c]**) and the feature selector matches Methods (`feature_selection_log_PROT.json` gives the exact drop counts to quote). Add the enrichment / pQTL scripts to the repo as `analysis/enrichment.py` (they depend on gseapy and an external evidence table) and cite the commit hash in Code availability. Fix the URL: the manuscript says `github.com/NMinster/pd-stratification`; the repository is `github.com/NMinster/PD_Strat`.

### B3. Test-set peeking in the reduced-panel result

"The top 10 proteins by importance achieved TEST ρ = 0.562 (vs. 0.544 for the full panel)" selects the panel size by looking at TEST ρ across k. A referee will call this optimistic selection, and the *direction* of the claim (a 10-protein panel beats 1,168 proteins) invites disbelief.

*Fix (implemented).* **[7b]** ranks proteins *inside each CV fold* (nested), chooses k\* by a pre-specified OOF rule (smallest k reaching 95 % of the maximal OOF ρ), and reports TEST ρ at k\* as the confirmatory number; the TEST curve is shown but labelled descriptive (`cumulative_importance.csv`, `figures/cumulative_importance_curve.png`). Report the stability-selected set's TEST performance the same way.

### B4. Circular "confirmatory" inference

The 40 proteins were chosen by importance on TRAIN and then tested in TRAIN with Bonferroni correction ("9 of 40 survived Bonferroni"). Selection and testing on the same data inflate significance; "CI overlap for 38/40" is also a weak criterion because wide CIs overlap easily.

*Fix (implemented).* **[7c]** labels TRAIN p-values *descriptive* and reports the confirmatory evidence as TEST replication: same sign, TEST p < 0.05, Bonferroni across the locked list *in TEST*, and CI overlap as a secondary criterion (`confirmatory_severity.csv`, `figures/confirmatory_forest.png`). The protein × time models are fitted in both cohorts with sign replication and the baseline-UPDRS sensitivity (`confirmatory_progression.csv`). Rewrite the Results paragraph and Fig. 3/4 legends accordingly.

### B5. Unaddressed confounders that differ systematically between PPMI and PDBP

PPMI enrolls de novo, untreated patients; PDBP is treated and later-stage. Part III scores depend on medication state, and severity correlates with disease duration. The covariate analysis uses only age, sex, site — a referee will say the cross-cohort ρ could be duration or medication state.

*Fix (implemented).* Assembly now records Part III medication state (prefers OFF exams when both exist), Hoehn & Yahr, age at diagnosis → disease duration, and DaTSCAN SBR. **[3i]** gives covariates-only / proteomics-only / combined models, partial ρ of proteomics given all covariates, and ρ within disease-duration tertiles and medication-state strata. If LEDD exists in your PD_Medical_History extract, add it as a numeric covariate (`_NUMERIC_COVS` in `validity.py`).

### B6. Under-powered negative control

Five permutations cannot support "correlations centred around zero". **[7]** now runs `n_permutations` (default 100) and reports the null mean/SD/max and an empirical p (`permutation_null.csv`, `figures/permutation_null.png`).

### B7. The draft is incomplete

Missing: Abstract, Keywords, "Unsupervised subtyping" (heading only), "Model selection" (heading only), Table 1–5 and figure files. Truncated sentences: "Compounding this, [15]."; "To address this gap, was modeled"; "Performance was unchanged when and restricting"; "Q5ZPR3 (; β = +4.88"; "The model also identified"; "default ,000"; "Missingness diagnostics included:,"; "baseline UPDRS alone, proteomics-predicted severity, and the residual…" (no verb). These must be fixed before any submission.

---

## 3. Major points (expect explicit reviewer comments)

**M1. Range shift and under-prediction of severe disease.** TEST MAE 20.7 and bias −45 for UPDRS > 60 are the natural consequence of training on a de novo cohort whose UPDRS rarely exceeds ~70; the model *cannot* extrapolate. Present this as a property of the design, not an error: **[3k]** reports ρ restricted to the TRAIN-observed range and the number of TEST rows above it; **[3l]** reports cross-fitted recalibration *within* TEST (linear and isotonic), which is the standard "update the model in the new setting" step (Steyerberg). Say explicitly that rank metrics are the primary transportability claim and calibration must be re-estimated per cohort.

**M2. Repeated measures.** "Mean 11.2 visits per participant in TRAIN" is striking; explain that proteomics were assayed at multiple visits, give the visit distribution in Table 1, and keep the participant-level estimand primary (already done). State that row-level metrics are secondary throughout, including in the Abstract.

**M3. Model comparison without inference.** "Monolithic 0.544 vs panel-aware 0.449" has no uncertainty. **[3m]** provides paired participant-level bootstrap Δρ with 95 % CI for every comparator, OOF and TEST. If the CI for panel-aware vs monolithic includes 0 on TEST, soften "between-panel covariance carries meaningful severity information" to "we found no advantage of panel-wise decomposition".

**M4. Fusion weights.** "Fusion weights favored proteomics (0.80) over RNA (0.64)" — these are unstandardized ridge coefficients on predictions with different variances and do not sum to 1; either report standardized weights or remove the sentence. The conclusion (RNA adds little transportable information) stands on the Δρ.

**M5. Huber sensitivity is in-sample.** "Huber ρ = 0.821 vs Ridge ρ = 0.818 in OOF" is an in-sample fit (OOF ρ is 0.54) and is uninformative; drop it or refit inside CV.

**M6. Heteroscedasticity.** Breusch–Pagan p = 9×10⁻⁴ with R² = 0.004 is statistically detectable but trivial; the TEST R² = 0.05 reflects the range shift (M1). Report BP results in one sentence as expected regression-to-the-mean behaviour.

**M7. Attrition and representativeness.** 24 % of TRAIN and 13 % of TEST participants retained is the first thing an epidemiologist will ask about. Provide a CONSORT-style flow (`sample_flow_table.csv` is the backbone) and a Table 1 comparison of retained vs not-retained participants (age, sex, baseline UPDRS, H&Y). *Not yet in the pipeline* — add a `retention_comparison()` to `assemble_clinical.py` if you want it automated.

**M8. Feature-selection counts.** Quote the numbers from `feature_selection_log_PROT.json` (n dropped for low observation, low MAD, redundancy, cap) so Methods and Results agree exactly.

**M9. Cross-endpoint validation.** Methods promises MoCA, GDS, SCOPA-AUT, Epworth, RBD, Schwab–England and H&Y; Results shows only UPDRS parts and UPSIT. Either add the endpoints or trim Methods. **[3j]** now covers UPDRS I–IV, UPSIT, Hoehn & Yahr and — importantly — **DaTSCAN putamen/caudate SBR**, an objective imaging measure. A significant negative ρ with putaminal SBR in *both* cohorts would be the single most persuasive construct-validity result in the paper; put it in the Abstract if it holds. Report the PD-only rows (controls inflate every endpoint correlation).

**M10. Progression analysis.** "n = 2,270 eligible" counts rows; slopes must be per participant. **[5b]** computes participant slopes from *all* clinical visits, the baseline-proteomic-severity → slope association with bootstrap CI in both cohorts, residual association after baseline UPDRS, an OLS with HC3 SEs, a mixed model with a pred₀ × time interaction, and — if `endpoints_time_to_event.csv` pairs correctly — Cox HRs per SD with Harrell's C. Replication of the slope association in PDBP is a stronger claim than anything currently in the Results; consider making "prognostic value beyond baseline UPDRS, replicated externally" the second headline.

**M11. Subtyping.** K = 2 chosen by the fallback rule with silhouette ≈ 0 and AMI ≈ 0.01 is not evidence of subtypes. Either delete the section (cleanest) or report it as a negative result in one paragraph, with **[5b]** "progression by subtype" (Kruskal–Wallis) as the only follow-up. Do not build the GSEA story on cluster labels that fail your own pre-specified stability criteria.

**M12. Over-reach in framing.** "Quantitative anchoring of NSD-ISS", "trial enrichment", "clinical-grade" — soften to "could complement" unless you show a decision-analytic result (net benefit at a stated threshold; the decision-curve analysis mentioned in Methods has no Results).

**M13. Pathway inference on Olink panels.** The four panels are pre-enriched for inflammation/oncology/cardiometabolic targets; using the assayed panel as background (done) is correct but should be stated as a limitation ("enrichment is relative to the panel, not the proteome"). ORA on 21 nominal hits is fragile; prefer the preranked GSEA on all 1,168 proteins as primary and ORA as descriptive.

**M14. pQTL anchoring.** "A curated pQTL-to-PD GWAS evidence table" needs a citation, a supplementary table, and the criteria for inclusion; otherwise it reads as cherry-picking GPNMB and CD38.

**M15. Cohort-discrimination check.** `cohort_disc_auc` (can the latent features tell PPMI from PDBP?) is computed but not reported; report it — a high AUC is expected and frames the harmonization sensitivity analyses.

---

## 4. Minor / editorial

- References cited out of order ([16] before [14]); ref 15 cited with no sentence; refs 3 and 23 are the same Goetz 2008 paper; refs 32, 34, 36 appear uncited; software section lists scikit-learn 1.8.0 — verify against `pip freeze`.
- "AMP-PD IR3" vs "2023 v4 release 1027" — use one label consistently (Ethics section says "PPMI IR3 and PDBP 2023 releases").
- Fig. 1 legend describes calibration plots but the embedded figure is a scatter; Fig. 2 legend matches; Fig. 4 legend ("replication heatmap") matches; re-number after restructuring.
- "Q5ZPR3" is a UniProt accession with no gene name given; map all accessions to gene symbols in text and tables.
- Units: UPDRS "points", slopes "points/year"; state the clamp range only once.
- Ethics: fine. Data availability: state that individual-level data cannot be redistributed and that `results/tables/*.csv` derived tables will be deposited.

---

## 5. Proposed Results structure (what feeds what)

| # | Results subsection | Pipeline source |
|---|---|---|
| 1 | Cohorts and sample flow (CONSORT; PD/HC per split; visits per participant) | `sample_flow_table.csv`, `clinical_unified.csv`, Table 1 |
| 2 | Feature selection and model development (OOF, all candidates, paired Δρ) | §3a, §3m |
| 3 | External validation in PDBP (row & participant, CIs, calibration, range shift, target recalibration) | §3b–3e, §3k, §3l |
| 4 | **The signal is graded severity, not diagnosis** (within-PD ρ, AUROC, PD-only refit) | §3h |
| 5 | Not explained by demographics, disease duration or medication state | §3i |
| 6 | Construct validity across endpoints incl. DaTSCAN | §3j |
| 7 | Prognostic value: baseline proteomic severity predicts progression and milestones beyond baseline UPDRS, replicated in PDBP | §5b |
| 8 | Robustness and negative controls (seeds, permutation p, random features, SVD sweep, fold-contained selection, harmonization) | §7 |
| 9 | Reduced panel (stability selection; nested curve; TEST at k\*) | §7b |
| 10 | Protein-level replication (locked 40; TEST replication; protein × time) | §7c |
| 11 | Pathways and genetic anchoring (external scripts) | `analysis/enrichment.py` |
| — | Subtyping: drop or one-paragraph negative result | §5, §5b |

---

## 6. Text you can adapt

**Abstract skeleton (fill from `SUMMARY_REPORT.md`).**
> Background: MDS-UPDRS is subjective and intermittent; no blood biomarker provides a transportable, graded estimate of PD severity. Methods: Using Olink Explore plasma proteomics (N proteins) from AMP-PD, we trained a regularised latent-factor model in PPMI (n participants / n visits) under participant-grouped cross-validation with all preprocessing locked inside the training cohort, and evaluated it once in PDBP (n / n). Results: Predicted severity correlated with observed MDS-UPDRS total in PDBP (participant-level ρ = X [CI]); within PD cases alone ρ = X [CI], while the same score discriminated PD from controls with AUROC = X, indicating a graded signal beyond diagnosis. The association was unchanged after adjustment for age, sex, site, disease duration and medication state (partial ρ = X), and predicted severity tracked Hoehn & Yahr stage (ρ = X) and putaminal DaT binding (ρ = −X). Baseline predicted severity predicted subsequent UPDRS slope beyond baseline UPDRS (β per SD = X, p = X) and time to H&Y ≥ 3 (HR per SD = X [CI]) in both cohorts. A k\* = N-protein panel selected on training data retained X % of full-panel performance externally; N proteins replicated at Bonferroni significance in PDBP. Conclusions: …

**Methods — population (new paragraph).**
> The severity model was fitted to [PD cases only | all participants with an MDS-UPDRS total]. Because healthy controls occupy the low end of the UPDRS range, we pre-specified two checks that a severity signal is not reducible to case–control status: rank correlation restricted to PD cases, and the area under the ROC curve of the same score for PD versus control. As a sensitivity analysis the model was refitted on [the complementary population].

**Discussion — limitation (replace current attrition sentence).**
> Two properties of the design bound the interpretation. First, PPMI enrols de novo patients, so the training distribution ends near UPDRS 70 and the model cannot extrapolate to advanced disease; external calibration therefore requires re-estimation in the target cohort, which we show restores absolute accuracy (cross-fitted MAE X) without changing rank performance. Second, only X % of PPMI and X % of PDBP participants had qualifying proteomics; retained participants were comparable on age, sex and baseline severity (Table 1), but selection on sample availability cannot be excluded.

---

## 6b. Literature-driven additions (see `docs/literature_review.md`)

- Cite and differentiate **Minster & Jafri, npj Parkinson's Disease 2026** (same data, PDBP→PPMI, PSI); justify the reversed cohort roles and report the reverse direction as a supplement.
- **Medication confound on DDC (P20711)**: plasma DDC rises with levodopa/DDC-inhibitor treatment (2024). Re-run with `prot_exclude: ["P20711"]`, report levodopa-stratified ρ (§3i) and the DDC coefficient by stratum.
- Update the SAA sentence (quantitative SAA now exists), the NSD-ISS framing (5-year staging results), add NfL as the incumbent comparator, and note Olink-platform dependence as a limitation.

## 7. Checklist after running on the real data

1. `python run.py` (full; ~1–2 h with `n_permutations: 100`, `stability_B: 200`).
2. Read §3h first. Decide primary population; if switching, set `severity_population: pd_only`, delete `results/tables/summary.json`, re-run.
3. Copy numbers into the manuscript from `SUMMARY_REPORT.md` only (never from the console), and cite `results/tables/summary.json` commit hash in Code availability.
4. Check `feature_selection_log_PROT.json` against the Methods counts.
5. Check the assembly log lines for column resolution (H&Y, medication state, age at diagnosis, DaTSCAN, time-to-event pairs). If `[TTE] could not pair`, set `tte_endpoints` in `config.yaml`.
6. Re-make Figs. 1–4 from `figures/`: suggested main figures — (1) design/CONSORT, (2) PD-only participant scatter OOF/TEST, (3) cross-endpoint forest, (4) progression scatter + Cox forest, (5) cumulative curve + stability bars, (6) confirmatory forest. Calibration and error-bin plots to Supplement.
7. Run `docs/TRIPOD_AI_checklist.md` line by line and add the completed checklist as a supplement.

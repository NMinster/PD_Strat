# TRIPOD+AI checklist — status for the PD_Strat manuscript

Status: ✅ reported, ⚠️ partial / needs edit, ❌ missing. "Source" names the pipeline output or manuscript section that supplies the item.

| # | Item | Status | Source / action |
|---|---|---|---|
| 1 | Title identifies the study as developing and externally validating a prediction model, target population, outcome | ⚠️ | Add "development and external validation" and "MDS-UPDRS severity" to the title |
| 2 | Abstract — structured summary incl. objectives, design, setting, participants, sample size, predictors, outcome, statistical analysis, results, conclusions, registration | ❌ | Write from `SUMMARY_REPORT.md` (see review §6) |
| 3a | Background and rationale incl. references to existing models | ✅ | Introduction |
| 3b | Objectives incl. development *and* validation | ✅ | Introduction, last paragraph |
| 4 | Source of data, study dates, registration | ⚠️ | State AMP-PD release ID once (IR3 vs 2023-v4); no registration — say so |
| 5a | Setting, eligibility, treatment received (medication state) | ⚠️ | Add PPMI de novo vs PDBP treated; Part III OFF-preference; `updrs3_state` column |
| 5b | Sample size justification | ❌ | Add: fixed by AMP-PD availability; give post-hoc precision (CI width of ρ) |
| 6a | Outcome definition, timing, blinding | ✅ | Methods "Cohort definition and endpoint" |
| 6b | Predictors, measurement, blinding | ✅ | Methods "Proteomics data" |
| 7a | Handling of missing data | ✅ | zero-fill after z-scoring + observation mask; completeness ≥ 50 % |
| 7b | Data pre-processing incl. normalization performed inside TRAIN | ✅ | HC-anchored z-scoring, feature selection log (`feature_selection_log_PROT.json`) |
| 8 | Model type, hyperparameters, selection (which model was pre-specified vs chosen on data) | ⚠️ | Fill "Model selection" section: primary chosen by TEST ρ among 3 pre-specified candidates — state this explicitly and report paired Δρ (§3m) |
| 9a | Measures of performance incl. calibration, discrimination, clinical utility | ⚠️ | Add cross-fitted target recalibration (§3l); decision curve is mentioned but not shown — add or remove |
| 9b | Internal validation method (resampling, groups) | ✅ | GroupKFold by participant; bootstrap CIs |
| 9c | External validation: same model applied without refitting; recalibration reported separately | ✅ | §3b vs §3l |
| 10 | Fairness / subgroup performance (sex, site, disease duration, medication) | ⚠️ | §3i strata; add sex-stratified ρ to Results |
| 11 | Handling of clusters (repeated measures) | ✅ | participant-level estimand, participant bootstrap |
| 12 | Software, versions, code availability with commit hash | ⚠️ | Fix repository URL; cite commit; add enrichment scripts |
| 13a | Participant flow diagram | ⚠️ | Build from `sample_flow_table.csv` |
| 13b | Characteristics of participants, incl. those excluded | ⚠️ | Table 1 needs PD/HC split and retained vs excluded (review M7) |
| 13c | Comparison of development and validation cohorts | ⚠️ | Add cohort-discrimination AUC and covariate table |
| 14 | Model development results: all candidates, with uncertainty | ✅ | §3a, §3m |
| 15 | Final model presentation sufficient for others to apply it | ⚠️ | Deposit scaler/SVD/ridge weights (`joblib`) and the locked protein list |
| 16 | Performance in development and validation with CIs | ✅ | §3b–3e |
| 17 | Model updating / recalibration in validation setting | ✅ | §3l |
| 18 | Interpretation incl. limitations (range shift, attrition, confounders) | ⚠️ | Rewrite limitation paragraph (review §6) |
| 19 | Explainability / feature importance and its stability | ✅ | §7b stability selection; §7c |
| 20 | Usability: intended use, how predictions would be used | ⚠️ | Soften NSD-ISS / trial-enrichment claims (review M12) |
| 21 | Patient and public involvement | ❌ | State none |
| 22 | Ethics, consent, data protection | ✅ | Ethics section |
| 23 | Funding, competing interests | ✅ | Acknowledgments |
| 24 | Study protocol / analysis plan availability | ⚠️ | The "locked analysis plan" is `pd_strat/config.py::LOCKED` — say so and deposit it |
| 25 | Registration | ❌ | Not registered — state explicitly |
| 26 | Supplementary materials list | ❌ | List Tables 1–5, checklist, enrichment tables, locked plan |
| 27 | Open science: data sharing statement | ✅ | Data availability |

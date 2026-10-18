# Literature gap analysis — plasma proteomics for PD prognosis (October 2026)

Purpose: after the seventh real-data run (Explore HT plasma + PPMI-native clinical;
`docs/real_data_readout.md`), identify where the 2024–2026 literature leaves open
questions that (1) the data already on disk can answer, or (2) other public data can
answer, and rank them by novelty, effort and relevance to the claim the manuscript
wants to make (prognostic value beyond MDS-UPDRS / the office assessment).
This extends `docs/literature_review.md` (September 2026); items covered there
(DDC/medication, NSD-ISS framing, NfL as incumbent, platform dependence, subtypes)
are not repeated. Access was limited to abstracts and search snippets; sample sizes
and effect sizes quoted below should be checked against the papers before citation.

---

## 0. Where the current results sit

| Claim | Seventh-run evidence | Status vs literature |
|---|---|---|
| Plasma proteome correlates with motor severity within PD | participant ρ 0.36 TRAIN / 0.44 holdout; proteome > covariates in holdout (0.45 vs 0.36); 27/40 proteins replicate | Novel at HT scale; the closest work (MDPI 2026, n = 211, 28 proteins) has no external test |
| Score tracks a rater-independent anchor | caudate DaT ρ −0.34 / −0.43 (Xing SBR) | Novel for plasma Olink HT |
| Within-person coupling | β 2.61 / 2.94, medication-adjusted 2.49 / 3.10 | Novel; nobody reports within-person plasma-proteome–UPDRS coupling |
| Baseline proteome predicts motor progression beyond clinical scoring | **negative** (126 configurations, nothing excludes zero); milestone Cox HR 1.33 / 1.17 | Consistent with the field: no replicated plasma predictor of motor slope exists |
| Prodromal phenoconversion | HR 1.69 [1.13, 2.53] per SD in held-out prodromal after age, sex, UPSIT, DaT, Part III; AUROC gain over clinical + imaging +0.01 to +0.03 (n.s.) | Comparator-with-DaTscan is harder than anything published; office-tier comparison running now |

The motor-progression route to "better than UPDRS" is closed by the data. The routes
that remain open are those where UPDRS has no information at all: biological status
(α-synuclein seeding), phenoconversion before motor signs, and cognitive outcomes.
The gaps below are ordered with that in mind.

---

## 1. What the field established in 2024–2026 that bears on these routes

| Theme | Key papers | What they show | What they leave open |
|---|---|---|---|
| Pre-diagnostic blood proteomics (population cohorts) | Gan et al., *Nature Aging* 2025 ([PubMed](https://pubmed.ncbi.nlm.nih.gov/39979637/)); *Brain* 2026 ML pre-symptomatic ([journal](https://academic.oup.com/brain/article/149/4/1254/8234010)); Jiao et al., *Parkinsonism Relat Disord* 2025 UKB NEFL/TPK1 ([Olink summary](https://olink.com/publication/plasma-proteomic-profiling-reveals-parkinsons-disease-associated-proteins-a-uk-biobank-study)); multi-cohort 74,000 medRxiv 2025 ([v2](https://www.medrxiv.org/content/10.1101/2025.07.30.25332433v2.full.pdf)) | UKB Olink 2,920 proteins: incident PD AUC 0.887, PPMI case-control validation 0.80; 23-protein subset AUC 0.78 up to 14 y pre-diagnosis; NEFL + TPK1 lift a clinical model from 0.78 to 0.84 (52,851 participants, 811 incident PD); 17 proteins up to 28 y pre-diagnosis replicated in AGES/ARIC/UKB (TPPP2, HPGDS, ALPL …) | None is tested in an **enriched prodromal cohort against the standard work-up** (UPSIT + DaTscan + Part III). UKB "incident PD" is a general-population diagnosis label, not phenoconversion in people already at risk |
| iRBD conversion | Hallqvist et al., *Nat Commun* 2024 ([journal](https://www.nature.com/articles/s41467-024-48961-3)) | Targeted MS, 8-protein panel, identifies converters up to 7 y before motor onset | Small (tens of iRBD), targeted MS not Olink, no DaTscan comparator, no HT-scale panel |
| Dynamic (serial-sample) proteomics for phenoconversion | Ran, Benatar et al., *Nat Med* 2026, ALS ([Olink summary](https://olink.com/publication/longitudinal-plasma-proteomics-predict-phenoconversion-to-clinically-manifest-als)) | 516 serial plasma samples, 33 phenoconverters, Olink Explore; 92 proteins change before conversion; 19-protein core panel AUC 0.80–0.89 across 0.5–5 y horizons, time-to-conversion MAE 1.6 y, beats NEFL alone, partial UKB replication | **No PD equivalent exists.** PPMI HT has serial prodromal samples |
| CSF α-synuclein SAA as the biological anchor | Venuto et al., *Neurology* 2024, predicting SAA status from clinical data ([journal](https://www.neurology.org/doi/10.1212/WN9.0000000000000008), [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11326325/)); quantitative/kinetic SAA: Orrù et al., *Lancet Neurol* 2025 ([UCL](https://discovery.ucl.ac.uk/id/eprint/10210185)), Coughlin et al., *Neurology* 2025 prodromal ([journal](https://www.neurology.org/doi/10.1212/WNL.0000000000210279)), npj PD 2023 kinetics and cognition ([journal](https://www.nature.com/articles/s41531-023-00627-5)); quantitative SAA predicts dementia ([PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12828062/)) | Demographics + clinical predict CSF SAA status in PPMI with AUROC ≈ 0.92; faster seeding tracks GBA1 and cognitive decline | **No plasma-proteome predictor of SAA / NSD status** has been published; NSD-ISS staging has no blood-based continuous anchor |
| Cognitive decline / PD dementia | CSF SomaScan PD proteomic score, *Neurology* 2023 ([journal](https://www.neurology.org/doi/10.1212/WNL.0000000000207725)); CSF IGF and cytokine signalling predict cognitive impairment, 2026 ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0969996126003153)); CSF proteomic pathways + caudate DAT predict cognitive vulnerability, *EJNMMI* 2026 ([PubMed](https://pubmed.ncbi.nlm.nih.gov/42472725/)); iRBD plasma pTau181/amyloid predict dementia ([PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12129744/)); ML reviews ([PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12738896/)) | CSF proteomics predicts PD-MCI/PDD (reported AUROC up to ≈ 0.88); caudate DAT deficit is the imaging correlate of cognitive vulnerability | **No plasma Olink HT predictor of PD-MCI/PDD** in PPMI; the pipeline's caudate-DaT correlation (ρ −0.43) is the bridge |
| Causal anchoring (pQTL MR) | proteogenomic network MR, *Nat Commun* 2024 ([journal](https://www.nature.com/articles/s41467-024-50718-x)); plasma proteins and brain structure MR, medRxiv 2026 ([preprint](https://www.medrxiv.org/content/10.64898/2026.07.13.26358006v1)); UKB-PPP pQTLs, Sun et al., *Nature* 2023 ([UKB page](https://www.ukbiobank.ac.uk/publications/plasma-proteomic-associations-with-genetics-and-health-in-the-uk-biobank/)); GP2 multi-ancestry PD GWAS, Kim et al., *Nat Genet* 2024 ([GP2](https://gp2.org/new-parkinsons-meta-gwas-of-over-1-8m-samples/)) | Cis-pQTL MR supports GPNMB, BST1, CTSB, HIP1R and others as risk proteins | Severity/progression proteins have not been MR-tested; progression GWAS (Tan et al., time to H&Y 3 / dementia, [medRxiv](https://medrxiv.org/content/10.1101/2022.07.07.22277297.full.pdf)) are underpowered but public |
| Genetic subgroups | LRRK2 integrated biofluid proteomics, medRxiv Dec 2025 ([preprint](https://www.medrxiv.org/content/10.64898/2025.12.22.25342856v1.full)); LRRK2 non-manifest carriers longitudinal ([PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9588016/)); L1CAM+ EV proteomics prodromal ([PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12692637/)) | LRRK2 carriers have distinct biofluid signatures | Whether the severity score transports to LRRK2/GBA1 carriers and non-manifest carriers is untested |
| Aging clocks | organ-specific proteomic clocks, *Nature Aging* 2025 ([journal](https://www.nature.com/articles/s43587-025-01016-8)); brain/immune aging, *Nat Med* 2025 ([journal](https://www.nature.com/articles/s41591-025-03798-1)) | Brain-age gap from plasma proteins predicts neurodegeneration | "Is the severity score accelerated brain aging?" is a reviewer question answerable only with clock weights |
| Data | PPMI Explore HT, ≈ 5,500 samples released August 2026 ([GEN](https://www.genengnews.com/topics/omics/thermo-and-michael-j-fox-foundation-collaborate-to-advance-proteomics-enabled-parkinsons-therapy/)); Olink AD/PD 2026 highlights (ITGAM/ITGAV and Lewy pathology, [blog](https://olink.com/blog/ad-pd-2026-blood-based-biomarkers-at-the-core-of-neurodegeneration-research)) | You are among the first to analyse the HT release end-to-end | Priority window is short: the same files are open to every PPMI user |

---

## 2. Gaps targetable with data already on disk

Data in hand: PPMI Explore HT plasma (Project 314) and CSF (277), bridged Explore 1536
(9000), AMP-PD PDBP Olink, Project 318 plasma (NEFL, GFAP), the PPMI-native clinical
exports (`S:/PPMI_clinical`), and the Curated Data Cut (columns include `NSD_Status`,
`NSD_STAGE`, `NSD_Adjudicated`, `Clinical_Stage`, `HIQ_RBD`, `subgroup`, `Death_Status`).

### G1. Plasma proteome → CSF α-synuclein SAA / NSD status (biological diagnosis from blood)

* **Why open.** Venuto et al. show clinical data alone predict SAA status with AUROC ≈ 0.92
  in PPMI; nobody has asked whether the plasma proteome adds to, or replaces, that
  clinical model, and no blood-based continuous anchor for NSD-ISS stages exists. This is
  the one target where MDS-UPDRS contributes nothing by construction.
* **Data.** Curated cut `NSD_Status` / `NSD_STAGE` / `NSD_Adjudicated`; `ASYNSAA` in
  Clinical_Diagnosis; the SAA Biospecimen Analysis Results file (Projects 155/237,
  `SAA_Status`, `SAA_Type`, kinetic columns Fmax/TTT/T50 per replicate) from LONI
  Biospecimen → Biospecimen Analysis. Not yet downloaded: SAA results file.
  **Access caveat (PPMI user guide §8.5):** LONI serves SAA results for PD participants
  only; the HC and prodromal results are sequestered and released on written request to
  ppmi@michaeljfox.org. Since PD cases are ≈ 90 % SAA+, the prodromal results are where
  the discrimination lives, so file that request first. `pd_strat/explore_paths.py` (§17)
  reports the coverage by cohort once the file is on disk.
* **Analyses.** (a) SAA+ vs SAA− in prodromal + HC + PD pooled and within prodromal
  (where status is ≈ 50/50); (b) NSD-ISS stage as ordinal target (2B/3/4/5/6) with the
  severity score as candidate continuous anchor; (c) SAA kinetic parameters (Fmax, TTT)
  as continuous targets; (d) comparator = Venuto-style clinical model (age, sex, UPSIT,
  RBD, DaT); report Δ AUROC with paired CI, TEST once on the holdout.
* **Pipeline.** `ppmi_clinical._FILES`: add `saa` (regex `SAA.*Biospecimen|SAA_Biospecimen`)
  and read `NSD_Status`/`NSD_STAGE` from the curated workbook; new discovery targets
  (binary `saa_positive`, ordinal `nsd_stage`, continuous `saa_fmax`); a prodromal-style
  §5f with tiered comparators. Moderate effort (≈ 1 day).
* **Risk.** PD cases are ≈ 90 % SAA+, so discrimination lives in the prodromal and HC
  strata; SAA− PD (LRRK2-enriched) is small. Pre-specify the prodromal analysis as primary.

### G2. Prodromal conversion vs the office-only tier (running now)

* **Why open.** All published prodromal signatures are compared with demographics or
  nothing; none is compared with hyposmia + Part III + RBD (the office assessment) and
  separately with DaTscan. The current run computes exactly that (§5e tiers).
* **Additions worth making after the run.** (a) Conversion endpoint variants: PD only
  (`ppmi_conversion_codes: [1]`) vs any synucleinopathy (add DLB/MSA codes from
  `PRIMDIAG`); (b) subgroup Cox by enrolment arm (`HIQ_RBD`, hyposmia, LRRK2/GBA1
  carriers) to show where the hazard comes from; (c) competing risk of death
  (`Death_Status`, `Death_Date` are in the curated cut); (d) decision-curve / net
  reclassification at 36 and 60 months, which is the clinical-utility framing referees
  now expect (TRIPOD+AI). Low effort, all data on disk.

### G3. Dynamic prodromal proteomics → time to conversion (the ALS template)

* **Why open.** The *Nat Med* 2026 ALS paper is the design: serial samples, proteins
  that move before conversion, horizon-specific AUCs and a time-to-conversion
  regression. No PD study has done this; PPMI HT has ≈ 1.5 samples per participant and
  longitudinal prodromal draws.
* **Analyses.** Count prodromal participants with ≥ 2 pre-conversion HT samples; model
  the within-person slope of the severity score (and of the top conversion proteins) as
  a time-dependent covariate in a landmark Cox; estimate years-to-conversion by
  regression on the last sample (MAE as in the ALS paper); compare with the baseline-only
  score.
* **Pipeline.** `longitudinal.py` already computes within-person slopes; needs a
  landmark/time-dependent Cox and a regression target `years_to_conversion`. Moderate
  effort. **Risk:** converters with serial pre-conversion samples may number 30–50; do the
  count first (one query on the aligned table).

### G4. Plasma prediction of cognitive decline and PD dementia

* **Why open.** CSF proteomics predicts PD-MCI/PDD; plasma Olink HT has not been tested.
  UPDRS does not predict dementia, so a positive result is a clean "beyond UPDRS" claim.
  Your caudate-DaT correlation links directly to the 2026 *EJNMMI* finding that caudate
  DAT deficit marks cognitive vulnerability.
* **Data (LONI Study Data → Non-motor Assessments).** `Montreal_Cognitive_Assessment`
  (MCATOT), `Cognitive_Categorization` (COGSTATE 1 normal / 2 MCI / 3 dementia, COGDECLN,
  FNCDTCOG, COGDXCL; [schema](https://blackfynn-ppmi-schema.readthedocs-hosted.com/en/stable-documentation-branch/documentation_files/cognitive_assessments/tables/COGCATG.html)),
  the neuropsychological battery (HVLT-R, Letter-Number Sequencing, SDMT, Benton JLO,
  semantic fluency), MDS-UPDRS 1.1 (NP1COG), and APOE from the PPMI genetic-status file.
  None of these is read by `ppmi_clinical.py` yet.
* **Analyses.** Targets: MoCA slope; time to MoCA < 21 (dementia screen) and to
  COGSTATE ≥ 2 / = 3 (Cox with the score vs age, education, baseline MoCA, APOE ε4, GBA1);
  cross-sectional MoCA as an additional cross-endpoint; cognition-weighted panel via the
  discovery benchmark. Comparators: NfL/GFAP (G5), the CSF-derived IGF/cytokine proteins
  if present on HT.
* **Pipeline.** Reader entries (`moca`, `cogcat`, `neuropsych`), `_updrs_tables`-style
  visit keying, two new `time_to_event` endpoints derived from PPMI-native longitudinal
  rows (the current `time_to_event.csv` comes only from the AMP-PD export), new discovery
  targets. Moderate effort (≈ 1–2 days).

### G5. Single-protein and published-panel comparators (NfL, GFAP, TPK1; UKB and multi-cohort signatures)

* **Why open.** The UKB paper shows NEFL + TPK1 alone add 0.06 AUC to a clinical model;
  referees will ask whether 5,000 proteins beat NfL. Separately, no one has validated the
  UKB incident-PD signatures (Gan 2025; *Brain* 2026; Jiao 2025) or the 17 pre-diagnostic
  proteins (TPPP2, HPGDS, ALPL, MFAP5, OGFR, ACAD8, TCL1A, GPC4, GSTA3, LCN2, KRAS, GJA1)
  in an enriched prodromal cohort against the DaTscan work-up. Both are cheap and both
  strengthen the paper whichever way they go.
* **Data.** NEFL, GFAP, TPK1, MERTK, ITGAV, ITGAM, BAG3, CLEC10A, HNMT are on Explore HT;
  Project 318 NEFL/GFAP are already wired as `extra_biomarkers` in `config_9000.yaml`.
* **Analyses.** For severity, milestone Cox, and conversion: NfL alone, NfL + clinical,
  score vs NfL (paired Δ), and "frozen published panels" scored as direction-weighted
  z-sums in PPMI with the same tiered comparators. Report transportability of each
  published panel as a table.
* **Pipeline.** `literature_proteins.yaml` gains the published panels with directions;
  `prodromal.py` / `discovery.py` gain a `published_panel` feature set; `extra_biomarkers`
  block in `config_ht.yaml`. Low effort (≈ half a day).

### G6. Plasma vs CSF Explore HT head-to-head (Project 277)

* **Why open.** The multi-platform paper (medRxiv Sept 2025) compared platforms, not
  tissues, for prognosis. Nobody has reported matched plasma vs CSF HT for severity,
  caudate-DaT correlation or prodromal conversion.
* **Pipeline.** Already supported (`--tissue CSF`, `PLA+CSF`, `compare_runs` on matched
  participants). Low effort; mainly compute time. Clinically it answers "does the blood
  test lose much against the lumbar puncture?", which is the practical question.

### G7. Genetic and phenotypic subgroup signatures

* **Why open.** LRRK2 biofluid proteomics (Dec 2025 preprint) shows carrier-specific
  signatures; whether the severity score transports to LRRK2/GBA1 carriers, and what it
  does in non-manifest carriers relative to their DaT, is untested.
* **Data.** `subgroup`, `lrrk2`, `gba` columns already assembled; non-manifest carriers
  are in the prodromal cohort.
* **Analyses.** Stratified severity ρ and conversion HR by carrier status; carrier-specific
  protein coefficients; score vs DaT in non-manifest carriers. Also sex-stratified
  replication (cheap, routinely requested).
* **Pipeline.** Config-level stratification plus one table; low effort.

---

## 3. Gaps that need other public data

| Resource | Access | What it enables | Lead time |
|---|---|---|---|
| **UK Biobank Pharma Proteomics Project** (Olink Explore 3072, 54,219 participants; ≈ 1,000 prevalent + 859 incident PD; hospital-coded dementia, death) | Application + fee | External replication of the HT severity panel as an incident-PD/dementia predictor in the general population; direct comparison with Gan 2025 and Jiao 2025 in the same data; NfL head-to-head at scale. No UPDRS, so severity itself cannot be replicated | Months |
| **UKB-PPP and deCODE cis-pQTL summary statistics** (Sun 2023; Ferkingstad 2021) + **PD GWAS** (Kim 2024 multi-ancestry on the NDKP portal; GP2 2025 meta-GWAS) + **progression GWAS** (Tan et al. time to H&Y 3 / dementia) | Public download | Two-sample MR + colocalization for the 27 replicated severity proteins and the conversion-weighted proteins: separates causal (GPNMB-like) from reactive (NfL-like) severity proteins. Fully public, no new samples | Weeks; external tooling (TwoSampleMR/coloc in R or an equivalent Python stack), outside the current package |
| **AMP-PD** cohorts beyond PPMI/PDBP (BioFIND, HBS, SURE-PD3, STEADY-PD3 in v3/v4) | Already granted (AMP-PD Tier 2) | Check which carry Olink 1536; any that do are a third test cohort for the 1536-transferred panel and for the medication question (trial cohorts have structured LEDD) | Days |
| **PPMI SomaScan CSF/plasma** (referenced by the multi-platform paper) and **Project 151/177 MS proteomics** | LONI (already have access) | Cross-platform replication of the top severity proteins (DDC, GPNMB, the 26 stable proteins); answers the platform-dependence limitation with data | Days to weeks |
| **Hallqvist 2024 iRBD panel** (supplementary protein list) | Public | Score the 8-protein panel in PPMI prodromal as another frozen panel (G5) | Hours |
| **ARIC / AGES-Reykjavik** (SomaScan, incident PD) | dbGaP / controlled | Replication of pre-diagnostic signals; cross-platform (SomaScan vs Olink) limits value | Months; low priority |
| **Organ-clock weights** (Oh et al.; *Nature Aging* 2025 supplements) | Public if released for Olink 3072 | Brain-age gap as a comparator to the severity score | Days if weights are available |

---

## 4. Priority ranking

| Rank | Gap | Novelty | Data on disk | Effort | Serves the "beyond UPDRS" claim |
|---|---|---|---|---|---|
| 1 | G2 prodromal vs office tier (+ endpoint variants, subgroup, competing risk, decision curve) | High | Yes | Low | Directly: conversion before motor signs |
| 2 | G1 plasma → SAA / NSD status | High | Yes (SAA file to download) | Moderate | Directly: biological status is outside UPDRS |
| 3 | G5 NfL and published-panel comparators | Medium | Yes | Low | Required to survive review either way |
| 4 | G4 cognition / dementia | High | Yes (3 files to download) | Moderate | Directly: dementia is outside UPDRS |
| 5 | G6 plasma vs CSF HT | Medium | Yes | Low | Indirect (practicality of a blood test) |
| 6 | G3 dynamic prodromal proteomics | High | Yes (count serial samples first) | Moderate–high | Directly, if serial converters ≥ 30 |
| 7 | G7 subgroup / sex | Low–medium | Yes | Low | Indirect (robustness) |
| 8 | MR with public pQTL/GWAS | Medium | Public | Moderate (separate tooling) | Indirect (biological plausibility) |
| 9 | UKB-PPP replication | High | Application | High | Indirect (population generalisation) |

Recommendation: run 1, 3 and 5 on the current data immediately (all are config or
small additions); download the SAA results and the three cognition files and build G1
and G4 next, since those are the two outcomes MDS-UPDRS cannot address and the two
that would justify the prognostic claim. Defer G3 until the serial-sample count is
known, and treat MR and UKB-PPP as a follow-up paper.

---

## 5. Concrete pipeline additions, by module

1. `pd_strat/ppmi_clinical.py` — `_FILES` entries for SAA results, MoCA,
   Cognitive_Categorization, neuropsychological tests, genetic status (APOE); curated
   `NSD_Status`/`NSD_STAGE` passthrough; cognitive milestones (MoCA < 21, COGSTATE ≥ 2 / 3)
   emitted as `time_to_event` rows.
2. `pd_strat/discovery.py` — targets `saa_positive`, `nsd_stage` (ordinal), `saa_fmax`,
   `moca_slope`, `time_to_pd_mci`, `time_to_pdd`.
3. `pd_strat/prodromal.py` — endpoint variants (PD vs any synucleinopathy), competing-risk
   (Fine–Gray or cause-specific) option, subgroup Cox, decision curves,
   `published_panel` and `nfl_only` feature sets.
4. `pd_strat/literature.py` / `docs/literature_proteins.yaml` — published panels with
   directions (Gan 2025, *Brain* 2026, Jiao 2025, 17-protein pre-diagnostic set, Hallqvist
   8-protein set) and a transportability table.
5. `pd_strat/longitudinal.py` — landmark Cox with time-dependent score; years-to-conversion
   regression (G3) once the serial-sample count justifies it.
6. `config_ht.yaml` — `extra_biomarkers` for Project 318 NEFL/GFAP; `--tissue CSF` run for
   Project 277.

---

## 6. Sources consulted

[Gan 2025 Nature Aging](https://pubmed.ncbi.nlm.nih.gov/39979637/) ·
[Brain 2026 pre-symptomatic ML](https://academic.oup.com/brain/article/149/4/1254/8234010) ·
[Jiao 2025 UKB NEFL/TPK1](https://olink.com/publication/plasma-proteomic-profiling-reveals-parkinsons-disease-associated-proteins-a-uk-biobank-study) ·
[multi-cohort 74,000 medRxiv v2](https://www.medrxiv.org/content/10.1101/2025.07.30.25332433v2.full.pdf) ·
[Hallqvist 2024 Nat Commun](https://www.nature.com/articles/s41467-024-48961-3) ·
[Ran & Benatar 2026 Nat Med ALS](https://olink.com/publication/longitudinal-plasma-proteomics-predict-phenoconversion-to-clinically-manifest-als) ·
[Venuto 2024 Neurology SAA prediction](https://www.neurology.org/doi/10.1212/WN9.0000000000000008) ·
[Venuto PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11326325/) ·
[Orrù 2025 Lancet Neurol SAA kinetics](https://discovery.ucl.ac.uk/id/eprint/10210185) ·
[Coughlin 2025 Neurology prodromal SAA](https://www.neurology.org/doi/10.1212/WNL.0000000000210279) ·
[SAA kinetics and cognition npj PD 2023](https://www.nature.com/articles/s41531-023-00627-5) ·
[quantitative SAA predicts dementia](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12828062/) ·
[CSF SomaScan PD proteomic score Neurology 2023](https://www.neurology.org/doi/10.1212/WNL.0000000000207725) ·
[CSF IGF/cytokine cognition 2026](https://www.sciencedirect.com/science/article/pii/S0969996126003153) ·
[CSF pathways + caudate DAT EJNMMI 2026](https://pubmed.ncbi.nlm.nih.gov/42472725/) ·
[iRBD plasma pTau181](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12129744/) ·
[ML for cognitive impairment review](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12738896/) ·
[PPMI Cognitive Categorization schema](https://blackfynn-ppmi-schema.readthedocs-hosted.com/en/stable-documentation-branch/documentation_files/cognitive_assessments/tables/COGCATG.html) ·
[proteogenomic network MR Nat Commun 2024](https://www.nature.com/articles/s41467-024-50718-x) ·
[plasma proteins and brain structure MR 2026](https://www.medrxiv.org/content/10.64898/2026.07.13.26358006v1) ·
[Sun 2023 UKB-PPP](https://www.ukbiobank.ac.uk/publications/plasma-proteomic-associations-with-genetics-and-health-in-the-uk-biobank/) ·
[Kim 2024 / GP2 meta-GWAS](https://gp2.org/new-parkinsons-meta-gwas-of-over-1-8m-samples/) ·
[Tan et al. progression GWAS (H&Y 3, death)](https://medrxiv.org/content/10.1101/2022.07.07.22277297.full.pdf) ·
[PRS and progression JAMA Neurol 2018](https://pubmed.ncbi.nlm.nih.gov/29340614/) ·
[LRRK2 biofluid proteomics 2025](https://www.medrxiv.org/content/10.64898/2025.12.22.25342856v1.full) ·
[LRRK2 non-manifest carriers](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9588016/) ·
[L1CAM+ EV proteomics](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12692637/) ·
[organ clocks Nature Aging 2025](https://www.nature.com/articles/s43587-025-01016-8) ·
[brain/immune aging Nat Med 2025](https://www.nature.com/articles/s41591-025-03798-1) ·
[PPMI HT release (GEN)](https://www.genengnews.com/topics/omics/thermo-and-michael-j-fox-foundation-collaborate-to-advance-proteomics-enabled-parkinsons-therapy/) ·
[Olink AD/PD 2026](https://olink.com/blog/ad-pd-2026-blood-based-biomarkers-at-the-core-of-neurodegeneration-research) ·
[multi-platform DDC validation](https://pmc.ncbi.nlm.nih.gov/articles/PMC12485981/) ·
[MDPI 2026 motor progression](https://pmc.ncbi.nlm.nih.gov/articles/PMC13113626/).

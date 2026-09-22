# Read-out of the first complete real-data run (plasma, pd_hc, visit-matched; 2026-09-21)

Numbers from `results/SUMMARY_REPORT.md`; this note records what they mean for the
manuscript so the framing decisions are written down before any text is rewritten.

## Who is in the model
| | PD participants (samples) | HC participants (samples) | PD UPDRS mean ± SD |
|---|---|---|---|
| TRAIN (PPMI) | 97 (431) | 125 (345) | 31 ± 17 |
| TEST (PDBP) | 84 (324) | 54 (193) | 51 ± 22 |

1,552 proteomic samples from 413 participants (≈4 per participant); 63 samples had no
matching clinical visit (mostly M6) and were unused. The manuscript's "3,074 samples"
must become 776 TRAIN / 517 TEST samples.

## Severity
* Mixed population (PD + HC): participant ρ = 0.55 (OOF) / 0.53 [0.39, 0.66] (TEST);
  AUROC PD-vs-HC 0.86 / 0.75 → a large part of the headline ρ is diagnosis.
* Within PD (same score): 0.18 / 0.36. PD-only refit: 0.29 [0.11, 0.45] / 0.45 [0.25, 0.64].
* Within PD, clinical covariates alone (age, sex, disease duration, medication) reach
  ρ = 0.41 / 0.51 — *better than proteomics* (0.25 / 0.34); partial ρ of proteomics
  given covariates = 0.16 / 0.23; combined ≈ covariates alone.
* Endpoints within PD: UPDRS III (0.25 / 0.36), H&Y (0.30 / 0.34), UPSIT (−0.26 / −0.17),
  DaTSCAN caudate −0.33 [−0.52, −0.11] (PPMI only; PDBP has no DaTSCAN).
* Range shift: PDBP mean UPDRS 51 vs PPMI 31; MAE 22, R² ≈ 0; cross-fitted recalibration
  in PDBP restores R² to 0.22–0.25 without changing rank.

**Framing that survives review:** a replicated but modest plasma-proteomic correlate of
*motor* severity in PD (ρ ≈ 0.3–0.45 within PD, external), distinct from case–control
signal, and weaker than — and largely orthogonal to — what disease duration, medication
and demographics already explain. Not a stand-alone severity estimator.

## Progression — the answer is no
* Severity-score → slope: PPMI ρ = 0.21 [0.02, 0.39] (p = 0.01); **PDBP ρ = −0.09
  [−0.29, 0.14]**; mixed-model interaction PPMI p = 0.011, PDBP p = 0.98; Cox HR/SD
  1.26 (PPMI, p = 0.06) vs 0.81 (PDBP). Does not replicate.
* Discovery benchmark (99 configurations, nested CV, TEST once): no target shows a
  TEST Δ vs clinical whose CI excludes 0 (slope Δ = +0.20 [−0.08, +0.47]; 24-month
  change Δ = +0.22 [−0.07, +0.50]; fast-progressor AUROC 0.42 vs clinical 0.48).
  OOF permutation p ≥ 0.02 and uncorrected for 99 configurations. No trial enrichment
  (top decile 0/8 fast progressors).
* Note: *clinical* baseline scoring does not predict progression here either
  (baseline UPDRS vs slope ρ ≈ 0 in PPMI); progression is intrinsically hard to predict
  in these cohorts at n ≈ 100.

**Consequence:** remove the "incremental prognostic value" section and any
progression / trial-enrichment claim. A one-paragraph pre-specified negative result
(with the benchmark table in the Supplement) is defensible and useful.

## Proteins
* Q5ZPR3 and P11215 replicate at Bonferroni in PDBP; DDC (P20711) replicates
  (β 1.96 → 1.94) and is **not** medication-associated in PPMI once samples are
  visit-matched (the earlier 3.4 → 11.6 discrepancy was the broadcast artifact).
  In PDBP nearly every PD sample is medicated, so no medication contrast exists there —
  say so; the visit-level `pd_medicated` flag (PPMI upd23a) now gives an
  untreated→treated contrast within participants.
* 12 stability-selected proteins reproduce full-panel TEST ρ (0.50 vs 0.49) — chosen on
  TRAIN, evaluated once. This is the cleanest "reduced panel" claim available.
* Monolithic > panel-aware on TEST with inference (Δρ −0.07 [−0.14, −0.00]).
* Permutation p = 0.01 (n = 100); seed / SVD / fold-contained selection stable.

## Drop
Subtyping (silhouette 0.007, AMI 0.02, K = 2 fallback), MSI_U (sign flips between
cohorts), all row-level ("sample-level") claims, the top-10 = full-panel claim.

## What would change the picture
1. More PD participants (newer PPMI Olink Explore 3072 release) — the only route to a
   progression claim.
2. CSF (`--tissue CSF`, `--tissue PLA+CSF`, then `compare_runs`).
3. Serum NfL as comparator (`extra_biomarkers`).

# Slide content — Clinical TabReD (P034)

Slide-by-slide content for the final presentation, in the layout of the course sample deck
(`reference/sample_slides/sample-NN.png`, style in `style_guide.md`). Every number comes from
`key_numbers.json`, `tables/` or `../docs/RESULTS.md` (all computed from the saved results).
`[CHECK]` marks items the presenters must confirm or adapt.

## Slide 1 — Title
Layout: sample-01.png
Section label: —
Presenter: 1
Content:
- Course line: CSCI 447 MACHINE LEARNING: THEORY AND PRACTICE · FINAL PROJECT · FALL 2026
- Project ID: P034
- Title: Clinical TabReD
- Subtitle: Evaluating Tabular Machine Learning Models under Temporal Distribution Shift in Healthcare Data
- Submitted by:
- Nurbek Baktygali, 202127847
- Yelzhan Rakhimzhanov, 202197093
- Submitted to: Dr. Hari Mohan Rai, Assistant Professor, Department of Computer Science, SCAI, Nazarbayev University
- Footer: CSCI 447 Machine Learning: Theory and Practice   |   Project ID: P034   |   Clinical TabReD
Visuals: none
Caption: none
Big numbers: none
Speaker notes:
Introduce the team and the project. The notebook is CSCI447P034.ipynb.

## Slide 2 — Presentation Roadmap
Layout: sample-02.png
Section label: OUTLINE
Presenter: 1
Content:
1 Abstract · 2 Introduction · 3 Background and Related Work · 4 Methodology / Project Approach ·
5 Implementation and System Design · 6 Evaluation and Results · 7 Discussion ·
8 Conclusion and Future Work · 9 References
Callout: Presenter 1: Sections 1–4 | Presenter 2: Sections 5–9 [CHECK]
Visuals: none
Caption: none
Big numbers: none
Speaker notes:
Nine sections, same order as the report.

## Slide 3 — Abstract
Layout: sample-03.png
Section label: SECTION 1 · ABSTRACT
Presenter: 1
Content:
- Aim: Build a clinical TabReD — a temporal-shift benchmark from public health data with each drift linked to its clinical cause — and test whether TabReD's conclusions hold in healthcare.
- Data: NHANES 2005–2018 (diabetes, 280 features), BRFSS 2017–2023 (diabetes, 93), NCHS natality 2016–2024 (preterm birth, 73).
- Method: 5 models × 7 TabReD splits (default time split, 3 sliding windows, 3 size-matched random splits), bootstrap CIs, 5-fold CV, drift analysis with documented causes.
- Results: GBDTs are best everywhere (test ROC-AUC 0.900 / 0.826 / 0.750). Random splits overestimate future performance in the diabetes datasets and reorder models in NHANES, but not for preterm birth.
- Conclusion: the type of drift (label shift, coding revisions) matters more than its size.
Visuals: none
Caption: none
Big numbers: none
Speaker notes:
One sentence each for aim, data, method, results, conclusion.

## Slide 4 — Problem, Motivation and Objectives
Layout: sample-04.png
Section label: SECTION 2 · INTRODUCTION
Presenter: 1
Content:
Problem: clinical models are trained on past patients and used on future ones; random train/test splits ignore time. TabReD [1] showed on 8 industrial datasets that this misjudges performance and model rankings, but it has no healthcare data.
Objectives:
1 Test whether random splits misjudge future performance and the model ranking — RQ1
2 Find which models transfer best to later periods — RQ2
3 Measure what drifts and identify its clinical cause — RQ3
4 Relate the type and size of drift to the performance gaps — RQ4
Visuals: none
Caption: none
Big numbers: 10.4% → 16.4% — diagnosed diabetes in NHANES, 2005-06 → 2017-18 (key_numbers: nhanes_prevalence_first_pct / last_pct)
Speaker notes:
The prevalence rise is a first example of temporal shift in clinical data.

## Slide 5 — Three Clinical Datasets
Layout: sample-05.png
Section label: SECTION 2 · INTRODUCTION
Presenter: 1
Content:
Table: source, target, periods, rows × features, positive rate (first → last period) for NHANES, BRFSS and natality.
Takeaway: three different kinds of clinical data (exam + lab, telephone survey, birth certificates) with 7–9 periods each.
Visuals: tables/t01_datasets.md
Caption: Table 1: The three datasets of the benchmark.
Big numbers: none
Speaker notes:
BRFSS and natality are sampled to 50,000 rows per year with a fixed seed (TabReD also subsampled its largest datasets). Rows are different people in every period.

## Slide 6 — Project Timeline
Layout: sample-06.png
Section label: SECTION 2 · INTRODUCTION
Presenter: 1
Content:
Phases: literature review (TabReD) → original 9-feature NHANES prototype → wide NHANES table 2005–2018 (280 features) → BRFSS and natality tables + leakage audits → benchmark (5 models × 7 splits × 3 datasets) → CV and drift analysis → notebook, report, presentation.
Visuals: none (draw as a timeline) [CHECK: add your real dates]
Caption: Fig. 1: Project timeline.
Big numbers: none
Speaker notes:
The project grew from one small NHANES table to three wide clinical tables.

## Slide 7 — Related Work and Research Gap
Layout: sample-07.png
Section label: SECTION 3 · BACKGROUND AND RELATED WORK
Presenter: 1
Content:
- TabReD [1]: time-based splits on industrial data; random splits misjudge performance; GBDTs and MLP-PLR strongest.
- TableShift [2]: distribution-shift tasks incl. BRFSS diabetes, but shifts by population group, not time.
- Grinsztajn et al. [3]: tree models outperform deep learning on medium-sized tabular data.
- Rabanser et al. [4]: domain classifiers detect dataset shift.
Research gap: no TabReD-style temporal benchmark on clinical data, and drift is rarely linked to its cause.
Visuals: none
Caption: none
Big numbers: none
Speaker notes:
Our contribution: a clinical TabReD with documented drift causes, and a same-period comparison of random vs time splits.

## Slide 8 — Proposed Methodology
Layout: sample-08.png
Section label: SECTION 4 · METHODOLOGY / PROJECT APPROACH
Presenter: 1
Content:
7 steps: 1 Public data → 2 Build tables (harmonise codes, missing codes, every-period rule) → 3 Leakage audit → 4 TabReD splits → 5 Train 5 models (tuned on validation) → 6 Evaluate (metrics, bootstrap CIs, random vs time, same-period, CV) → 7 Drift by cause
Visuals: diagrams/d01_methodology.svg
Caption: Fig. 2: Methodology.
Big numbers: none
Speaker notes:
Test data are used once; preprocessing, model choice and threshold use training/validation only.

## Slide 9 — Data Preparation and Leakage Control
Layout: sample-09.png
Section label: SECTION 4 · METHODOLOGY / PROJECT APPROACH
Presenter: 1
Content:
- Special codes ("don't know", "refused") → missing; renamed BRFSS questions harmonised; only features present in every period; IDs, weights, dates and >95%-empty columns removed; rows never dropped.
- Leakage removed by rule: diabetes/insulin/glucose questions (asked only of diabetics), consequences of the diagnosis (e.g. pneumococcal vaccine), and for natality everything recorded at or after delivery.
- Audit: no remaining feature predicts the label on its own with ROC-AUC > 0.80, and no missingness pattern does.
Visuals: none (optional: results_*/leakage_audit.csv)
Caption: none
Big numbers: 280 | 93 | 73 — features (NHANES | BRFSS | natality)
Speaker notes:
Strongest single predictors are plausible risk factors (age, BMI, general health, plurality), not restatements of the label.

## Slide 10 — Split Design (TabReD)
Layout: sample-10.png
Section label: SECTION 4 · METHODOLOGY / PROJECT APPROACH
Presenter: 1
Content:
- Default time split: train early periods → validate next → test latest.
- Sliding windows (×3): train 3 periods → validate 1 → test 1.
- Random splits (×3): the rows of each window shuffled and re-split into the same sizes.
Visuals: diagrams/d03_splits.svg | tables/t02_splits.md
Caption: Fig. 3: Time-based splits per dataset. Table 2: Split periods.
Big numbers: none
Speaker notes:
Random-k differs from sliding-window-k only by ignoring time order, so their difference measures the cost of ignoring time.

## Slide 11 — Mathematical Formulation
Layout: sample-11.png
Section label: SECTION 4 · METHODOLOGY / PROJECT APPROACH
Presenter: 1
Content:
Equations (1)–(8) from equations.md: logistic model, F1, ROC-AUC, PSI, TVD, bootstrap CI, random-minus-time gap, relative improvement over the MLP.
Visuals: equations.md
Caption: Equations (1)–(8).
Big numbers: none
Speaker notes:
(7) and (8) are the two quantities TabReD reports; we add bootstrap CIs to them.

## Slide 12 — Technology Stack and Experimental Setup
Layout: sample-12.png
Section label: SECTION 5 · IMPLEMENTATION AND SYSTEM DESIGN
Presenter: 2
Content:
Tools and versions; CPU-only training (16 cores, one job at a time); seed 42; small predefined grids; 1,000 bootstrap resamples.
Visuals: tables/t14_tech_stack.md
Caption: Table 3: Technology stack.
Big numbers: 105 — trained model selections (5 models × 7 splits × 3 datasets)
Speaker notes:
The full benchmark takes about 3 hours on a 16-core CPU; running two jobs in parallel slowed both 10–50× (thread oversubscription).

## Slide 13 — System Architecture
Layout: sample-13.png
Section label: SECTION 5 · IMPLEMENTATION AND SYSTEM DESIGN
Presenter: 2
Content:
Wide table → ColumnTransformer (fit on train only) → 5 models → selection by validation PR-AUC → threshold max F1 on validation → test scored once with bootstrap CIs.
Visuals: diagrams/d02_architecture.svg
Caption: Fig. 4: Model pipeline.
Big numbers: none
Speaker notes:
Class imbalance: balanced class weights / scale_pos_weight; the MLP uses balanced sample weights and early stopping.

## Slide 14 — Core Algorithm
Layout: sample-14.png
Section label: SECTION 5 · IMPLEMENTATION AND SYSTEM DESIGN
Presenter: 2
Content:
Algorithm 1 pseudocode (algorithm.md) with grouped explanations on the right.
Visuals: algorithm.md
Caption: Algorithm 1: Clinical TabReD benchmark for one dataset.
Big numbers: none
Speaker notes:
Explain the groups: splits, tuning loop, threshold, single test evaluation, random-vs-time comparison.

## Slide 15 — Experimental Design: Evaluation and Cross-Validation
Layout: sample-15.png
Section label: SECTION 6 · EVALUATION AND RESULTS
Presenter: 2
Content:
- Main metric ROC-AUC (independent of threshold and of prevalence, which drifts); PR-AUC, accuracy, precision, recall, F1, confusion matrix also reported.
- 95% bootstrap CIs (1,000 resamples); paired bootstrap for model differences; unpaired for random vs time.
- 5-fold stratified CV on the development periods only (in-period performance and stability).
- New: random vs time scored on the same test period.
Visuals: none
Caption: none
Big numbers: none
Speaker notes:
CV measures in-period performance; the time split measures transfer to the future. Their difference is the same question as random vs time.

## Slide 16 — Results on the Default Time Split
Layout: sample-16.png
Section label: SECTION 6 · EVALUATION AND RESULTS
Presenter: 2
Content:
All metrics of the 5 models on each dataset's test periods (validation-tuned threshold).
Takeaway: GBDTs best on all three datasets; NHANES separates models most (0.876–0.900), BRFSS and natality only about 0.005.
Visuals: tables/t03_default_test_metrics.md
Caption: Table 4: Test performance on the default time split.
Big numbers: 0.900 — NHANES XGBoost | 0.826 — BRFSS LightGBM | 0.750 — natality XGBoost (test ROC-AUC)
Speaker notes:
In NHANES the model chosen on validation (MLP, 0.887) is significantly worse than XGBoost on the later test years.

## Slide 17 — Model Performance Comparison
Layout: sample-17.png
Section label: SECTION 6 · EVALUATION AND RESULTS
Presenter: 2
Content:
Test ROC-AUC per model with 95% bootstrap CIs, one panel per dataset; paired bootstrap differences in tables/t04_paired_comparisons.md.
Visuals: figures/fig02_default_auc_ci.png | charts_data/c02_default_auc_ci.csv
Caption: Fig. 5: Default-split test ROC-AUC with 95% bootstrap CIs.
Big numbers: none
Speaker notes:
The x-axes do not start at zero. Small absolute differences are significant because the test sets are large (BRFSS 100,000, natality 100,000 rows).

## Slide 18 — Cross-Validation Results
Layout: sample-18.png
Section label: SECTION 6 · EVALUATION AND RESULTS
Presenter: 2
Content:
5-fold CV ROC-AUC (mean ± std) per model and dataset; fold std ≤ 0.005.
In-period CV is higher than the future test for NHANES (≈0.91 vs 0.88–0.90) and BRFSS (≈0.83 vs 0.82–0.83), equal for natality (≈0.75).
Visuals: tables/t09_cross_validation.md | figures/fig09_cross_validation.png | charts_data/c09_cv_folds.csv
Caption: Table 5: 5-fold CV on the development periods. Fig. 6: CV ROC-AUC per fold.
Big numbers: none
Speaker notes:
Same pattern as random vs time: in-period evaluation is optimistic for the diabetes datasets.

## Slide 19 — Training Curves
Layout: sample-19.png
Section label: SECTION 6 · EVALUATION AND RESULTS
Presenter: 2
Content:
MLP train and validation loss per epoch (default split); dashed line = early-stopping epoch. Validation is a later period, so the gap includes temporal shift.
Visuals: figures/fig06_mlp_training_curves.png | charts_data/c06_mlp_history.csv
Caption: Fig. 7: MLP training curves.
Big numbers: none
Speaker notes:
Curves are the recorded history of the evaluated MLP (balanced log loss).

## Slide 20 — Confusion Matrices and ROC Curves
Layout: sample-20.png
Section label: SECTION 6 · EVALUATION AND RESULTS
Presenter: 2
Content:
Confusion matrix of the validation-selected model per dataset (max-F1 threshold); ROC curves of all models.
Recall at the chosen threshold: ≈0.67 (NHANES), ≈0.58 (BRFSS), ≈0.42 (natality).
Visuals: figures/fig05_confusion_matrices.png | figures/fig04_roc_curves.png | charts_data/c05_confusion_matrices.csv | charts_data/c04_roc_points.csv
Caption: Fig. 8: Confusion matrices. Fig. 9: ROC curves on the default test periods.
Big numbers: none
Speaker notes:
Preterm birth is much harder to predict from information available before delivery.

## Slide 21 — Random vs Time Splits (TabReD comparison)
Layout: sample-21.png
Section label: SECTION 6 · EVALUATION AND RESULTS
Presenter: 2
Content:
Random-split minus time-split test ROC-AUC per window and model with 95% CIs.
- NHANES: up to +0.02 in 2015-16 (4/15 significant); MLP ranks 2nd under random but 4th under time splits.
- BRFSS: +0.006 to +0.007 for every model; 9/15 significant (2021, 2022).
- Natality: −0.003 to −0.006 (time looks better!).
Visuals: figures/fig03_random_minus_time.png | tables/t06_random_vs_time.md | charts_data/c03_random_minus_time.csv
Caption: Fig. 10: Random minus time ROC-AUC per sliding window.
Big numbers: 2nd → 4th — MLP rank in NHANES, random → time
Speaker notes:
Positive = a random split would have promised more than the model delivers on the next period.

## Slide 22 — Same Test Period: Separating Time from Period Difficulty
Layout: sample-21.png
Section label: SECTION 6 · EVALUATION AND RESULTS
Presenter: 2
Content:
TabReD's random and time test sets contain different periods. Scoring the random-split model only on the time split's test period:
- Natality: difference vanishes (never significant) — later years are simply easier (0.735 on 2016 rows → 0.761 on 2020 rows).
- BRFSS: random higher in 15 of 15 model-windows.
- NHANES 2017-18: pooled random model significantly worse (≈ −0.04) — plausibly the lab-analyzer change.
Visuals: figures/fig10_random_minus_time_same_period.png | tables/t07_random_vs_time_same_period.md | charts_data/c10_random_minus_time_same_period.csv
Caption: Fig. 11: Random minus time on the same test period.
Big numbers: 15 / 15 — BRFSS model-windows where random is optimistic on the same period
Speaker notes:
Methodological contribution: without the same-period comparison, natality would look like "time splits are easier".

## Slide 23 — Drift and Its Clinical Causes
Layout: sample-21.png
Section label: SECTION 6 · EVALUATION AND RESULTS
Presenter: 2
Content:
- Domain classifier ROC-AUC (time splits): NHANES 0.86–0.97, BRFSS 0.76–0.79, natality 0.55–0.59; random splits ≈0.50.
- NHANES: lab analyzers replaced 2013-14 (RDW) and 2017-18 (LDH +14.9%, AST −18.0%); ACA 2014 (uninsured 25.6% → 15.2%).
- BRFSS: questionnaire revisions 2021–22 (personal doctor, insurance, race); cell-phone sample.
- Natality: California withholds marital status from 2017; hypertension doubles; smoking falls by two thirds.
Visuals: figures/fig08_drift_events.png | figures/fig07_domain_shift.png | tables/t10_domain_classifier.md | tables/t11_drift_events.md | charts_data/c07_domain_classifier.csv | charts_data/c08_drift_events.csv
Caption: Fig. 12: Documented changes per period. Fig. 13: Domain-classifier ROC-AUC.
Big numbers: 0.97 — NHANES 2017-18 domain AUC (largest shift)
Speaker notes:
Sources: CDC/NCHS documentation (CBC_H, BIOPRO_J, BRFSS codebooks, NCHS natality user guides).

## Slide 24 — Does Drift Explain the Gaps? / TabReD Table 4
Layout: sample-22.png
Section label: SECTION 6 · EVALUATION AND RESULTS
Presenter: 2
Content:
- Largest covariate shift (NHANES 2017-18, domain AUC 0.97) → no significant gap; the significant NHANES gap comes with the label shift (prevalence 12.6% → 15.1%) and the ACA change.
- BRFSS gaps appear in coding-revision years (2021, 2022), none in 2023.
- TabReD Table 4: XGBoost's lead over the MLP does not shrink under time splits — NHANES +0.28% (random) → +0.97% (time).
Visuals: tables/t12_drift_vs_performance.md | tables/t08_improvement_over_mlp.md
Caption: Table 6: Drift vs performance per window. Table 7: % ROC-AUC improvement over the MLP.
Big numbers: +0.97% — XGBoost over MLP, NHANES time splits (TabReD: the lead shrinks)
Speaker notes:
Type of drift matters more than size.

## Slide 25 — What the Results Mean
Layout: sample-23.png
Section label: SECTION 7 · DISCUSSION
Presenter: 2
Content:
Answers to TabReD's claims per dataset (table).
Strengths: three real clinical sources; TabReD split construction; bootstrap CIs; documented drift causes; reproducible notebook.
Limitations: one seed and small grids; no CatBoost / MLP-PLR; feature counts of BRFSS and natality below TabReD's; causal explanations are hypotheses; no survey weights.
Visuals: tables/t13_research_questions.md
Caption: Table 8: TabReD's claims on clinical data.
Big numbers: none
Speaker notes:
Interpret, do not repeat numbers.

## Slide 26 — Conclusion and Future Work
Layout: sample-24.png
Section label: SECTION 8 · CONCLUSION AND FUTURE WORK
Presenter: 2
Content:
Conclusion: GBDTs stay best on clinical data. Random splits overestimate and reorder models where labels shift or questionnaires change (NHANES, BRFSS), not for preterm birth. Compare random and time splits on the same period.
Future work: 1 More seeds and Optuna tuning · 2 MLP-PLR / FT-Transformer and CatBoost on GPU · 3 Shift-robust training (reweighting, recalibration under prevalence shift) · 4 EHR data with real deployment time
Visuals: none
Caption: none
Big numbers: 3 — clinical datasets | 7 — splits each | 5 — models
Speaker notes:
Close on the main finding and the next steps.

## Slide 27 — Code, Data and Academic Integrity
Layout: sample-25.png
Section label: SUBMISSION
Presenter: 2
Content:
1 Notebook: CSCI447P034.ipynb — the full procedure; reproduces the saved results
2 Code and results: project repository (scripts/, results_*/, figures/)
3 Data: public CDC NHANES, BRFSS and NCHS natality files (links in references)
4 Integrity: every source cited in IEEE style [CHECK course rules]
Visuals: none
Caption: none
Big numbers: none
Speaker notes:
State requirements; do not claim similarity scores.

## Slide 28 — References
Layout: sample-26.png
Section label: SECTION 9 · REFERENCES
Presenter: 2
Content:
IEEE-style list from references.md.
Visuals: references.md
Caption: none
Big numbers: none
Speaker notes:
—

## Slide 29 — Thank You
Layout: sample-27.png
Section label: —
Presenter: 2
Content:
Thank You
Questions and discussion
Nurbek Baktygali · Yelzhan Rakhimzhanov   |   Project ID: P034   |   CSCI 447, Nazarbayev University
Visuals: none
Caption: none
Big numbers: none
Speaker notes:
Invite questions.

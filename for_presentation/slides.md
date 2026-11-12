# Slide content — Clinical TabReD Lite

Every number below is generated from the results folder. Sentences marked [REVIEW] must be rewritten by the students.

## Slide 1 — Title
Layout: sample-01.png
Section label: —
Presenter: 1
Content:
- Course line: CSCI 447 MACHINE LEARNING: THEORY AND PRACTICE · FINAL PROJECT · FALL 2026
- Project ID: P034
- Title: Clinical TabReD Lite
- Subtitle: Evaluating Tabular Machine Learning Models under Temporal Distribution Shift in Healthcare Data
- Submitted by:
- Nurbek Baktygali, STUDENT_ID_1
- Yelzhan Rakhimzhanov, STUDENT_ID_2
- Submitted to: Dr. Hari Mohan Rai, Assistant Professor, Department of Computer Science, SCAI, Nazarbayev University
- Footer: CSCI 447 Machine Learning: Theory and Practice   |   Project ID: P034   |   Clinical TabReD Lite
Visuals: none
Caption: none
Big numbers: none
Speaker notes:
Introduce the team and the project title. The notebook file is CSCI447P034.ipynb (check the course naming rule).

## Slide 2 — Presentation Roadmap
Layout: sample-02.png
Section label: OUTLINE
Presenter: 1
Content:
1 Abstract · 2 Introduction · 3 Background and Related Work · 4 Methodology / Project Approach ·
5 Implementation and System Design · 6 Evaluation and Results · 7 Discussion ·
8 Conclusion and Future Work · 9 References
Callout: Presenter 1: Sections 1–4 | Presenter 2: Sections 5–9
Visuals: none
Caption: none
Big numbers: none
Speaker notes:
The outline follows the nine report sections. Presenter 1 covers sections 1–4 and Presenter 2 covers sections 5–9.

## Slide 3 — Abstract
Layout: sample-03.png
Section label: SECTION 1 · ABSTRACT
Presenter: 1
Content:
- Aim: Test whether tabular models trained on older NHANES cycles keep their performance on a newer cycle.
- Methodology: 5 models, chronological (2011-12 → 2013-14 → 2015-16) vs random vs in-period splits, 5-fold CV, bootstrap CIs, drift statistics.
- Results: XGBoost (selected on validation) ROC-AUC 0.791 [0.775, 0.807] on 2015-16; random-minus-chronological gap -0.004 [-0.033, +0.023].
- Conclusion: [REVIEW] Domain-classifier ROC-AUC 0.582; the random-vs-chronological gap is inconclusive (the 95% CI includes zero); the in-period-vs-future gap is inconclusive (the 95% CI includes zero).
Visuals: none
Caption: none
Big numbers: none
Speaker notes:
One sentence each for aim, method, result and conclusion. [REVIEW] Explain in your own words why the CIs matter for the conclusion.

## Slide 4 — Problem, Motivation and Objectives
Layout: sample-04.png
Section label: SECTION 2 · INTRODUCTION
Presenter: 1
Content:
Problem: clinical models are trained on past patients but used on later ones; random splits can hide this time gap [1].
Objectives:
1 Compare random vs chronological evaluation — RQ1
2 Find which models transfer to a newer cohort — RQ2
3 Measure feature and prevalence changes between cycles — RQ3
4 Relate observed changes to performance — RQ4
Visuals: none
Caption: none
Big numbers: 15.1% — adults reporting diagnosed diabetes in 2015-2016 (unweighted sample, Table 1)
Speaker notes:
Introduce the deployment-over-time problem with the TabReD motivation [1]. [REVIEW] Say why diabetes in NHANES is a reasonable small test case.

## Slide 5 — Survey Cycles at a Glance
Layout: sample-05.png
Section label: SECTION 2 · INTRODUCTION
Presenter: 1
Content:
Table of the three cycles: adults, positives, unweighted and survey-weighted prevalence, mean age, mean BMI.
Takeaway: [REVIEW] Prevalence rises in the newest cycle while age and BMI change only slightly.
Visuals: tables/t01_cycle_context.md
Caption: Table 1: Eligible adults and diabetes prevalence per NHANES cycle (computed from the data).
Big numbers: none
Speaker notes:
All values are computed from the downloaded data, not taken from external statistics. Weighted prevalence uses the interview weight WTINT2YR and is descriptive only.

## Slide 6 — Project Timeline
Layout: sample-06.png
Section label: SECTION 2 · INTRODUCTION
Presenter: 1
Content:
Gantt chart of the planned work (weeks): literature review, download and codebook check,
data preparation, splits and baselines, tuning/CV/MLP, evaluation and ablation, drift analysis,
notebook/report/presentation.
Visuals: tables/t02_timeline.md | charts_data/c10_gantt.csv
Caption: Table 2: Planned project timeline (same as the report).
Big numbers: none
Speaker notes:
This is the planned schedule; adjust the weeks to your real dates. It must match the timeline table in the report.

## Slide 7 — Related Work and Research Gap
Layout: sample-07.png
Section label: SECTION 3 · BACKGROUND AND RELATED WORK
Presenter: 1
Content:
Table of four related studies: approach, reference, data, split type, key finding.
Research gap: [REVIEW] few small, fully reproducible studies of temporal shift on public health-survey data.
Visuals: tables/t03_related_work.md
Caption: Table 3: Related work (findings quoted from the cited abstracts).
Big numbers: none
Speaker notes:
TabReD [1] motivates time-based splits; TableShift [2] includes a BRFSS diabetes task with a race-based shift. [REVIEW] State the gap in your own words.

## Slide 8 — Proposed Methodology
Layout: sample-08.png
Section label: SECTION 4 · METHODOLOGY / PROJECT APPROACH
Presenter: 1
Content:
8 steps: 1 Download (validated XPT) → 2 Prepare (merge, label, filter) → 3 Features (9 inputs) →
4 Split (A / A2 / B) → 5 Cross-validation (5-fold) → 6 Training (5 models) → 7 Evaluation
(metrics, CIs, ablation) → 8 Drift (KS, PSI, domain classifier)
Visuals: diagrams/d01_methodology.svg
Caption: Fig. 1: Proposed methodology.
Big numbers: none
Speaker notes:
Walk through the eight steps left to right. Stress that test data is used once and preprocessing is fitted on training data only.

## Slide 9 — Dataset Description
Layout: sample-09.png
Section label: SECTION 4 · METHODOLOGY / PROJECT APPROACH
Presenter: 1
Content:
CDC NHANES public files [5]: DEMO, BMX, DIQ for 2011-12 (G), 2013-14 (H), 2015-16 (I).
Target DIQ010: 1 = yes, 2 = no; 3/7/9/missing excluded. Adults ≥ 20 with an exam record.
Visuals: tables/t04_dataset.md
Caption: Table 4: Dataset composition per cycle.
Big numbers: 15,964 — eligible adults | 13.0% → 15.1% — prevalence 2011-12 → 2015-16 | 9 — input features
Speaker notes:
Data are downloaded from the CDC site inside the notebook; nothing is uploaded. The data-flow table in the notebook lists every exclusion.

## Slide 10 — Features and Preprocessing
Layout: sample-10.png
Section label: SECTION 4 · METHODOLOGY / PROJECT APPROACH
Presenter: 1
Content:
Table: feature, type, source file, missing % (2015-16), preprocessing.
Diagram: three feature groups → numeric / categorical pipelines → model input.
Visuals: tables/t05_features.md | diagrams/d04_features.svg
Caption: Table 5: Features and preprocessing. Fig. 2: Feature groups and pipelines.
Big numbers: none
Speaker notes:
Codes such as education 7/9 become missing. Laboratory diabetes markers and survey weights are never used as inputs.

## Slide 11 — Mathematical Formulation
Layout: sample-11.png
Section label: SECTION 4 · METHODOLOGY / PROJECT APPROACH
Presenter: 1
Content:
Six numbered equations (1)–(6) from equations.md: logistic model, precision/recall, F1,
KS statistic, PSI, CV mean and standard deviation.
Visuals: equations.md
Caption: Equations (1)–(6).
Big numbers: none
Speaker notes:
Explain each symbol briefly. KS and PSI measure distribution change; the bootstrap interval (7) is used on the results slides.

## Slide 12 — Technology Stack and Experimental Setup
Layout: sample-12.png
Section label: SECTION 5 · IMPLEMENTATION AND SYSTEM DESIGN
Presenter: 2
Content:
Table of tools and exact versions; hardware and reproducibility notes (seed, n_jobs, CPU only).
Visuals: tables/t06_tech_stack.md
Caption: Table 6: Technology stack (versions recorded at run time).
Big numbers: none
Speaker notes:
Every version is pinned in requirements.txt and recorded in results/run_metadata.json; the notebook installs only missing packages.

## Slide 13 — System Architecture
Layout: sample-13.png
Section label: SECTION 5 · IMPLEMENTATION AND SYSTEM DESIGN
Presenter: 2
Content:
Diagram: ColumnTransformer → 5 models → threshold (max F1 on validation) → metrics.
3 steps: 1 Fit preprocessing on training split only · 2 Tune each model on validation PR-AUC ·
3 Report test metrics at 0.5 and tuned threshold with bootstrap CIs
Visuals: diagrams/d02_architecture.svg
Caption: Fig. 3: System architecture.
Big numbers: none
Speaker notes:
All five models share the same preprocessing. Class imbalance is handled with balanced weights or scale_pos_weight.

## Slide 14 — Core Algorithm
Layout: sample-14.png
Section label: SECTION 5 · IMPLEMENTATION AND SYSTEM DESIGN
Presenter: 2
Content:
Algorithm 1 pseudocode (algorithm.md) with grouped explanations on the right.
Visuals: algorithm.md
Caption: Algorithm 1: Temporal benchmark for one protocol.
Big numbers: none
Speaker notes:
Explain the groups, not every line: tuning loop, MLP epoch loop with early stopping, threshold choice on validation, single test evaluation.

## Slide 15 — Experimental Design: Split and Cross-Validation
Layout: sample-15.png
Section label: SECTION 6 · EVALUATION AND RESULTS
Presenter: 2
Content:
A chronological: train 2011-12 → validate 2013-14 → test 2015-16 (once).
A2 in-period: same model, 20% holdout of 2011-12 vs 2015-16.
B random: pooled 70/15/15, 5 seeds. CV: 5-fold stratified on development data only.
Why both? CV measures in-period (i.i.d.) stability; chronological testing measures transfer to a later cohort.
Visuals: diagrams/d03_protocols.svg
Caption: Fig. 4: Evaluation protocols.
Big numbers: none
Speaker notes:
Say clearly that the test set is never used for tuning or thresholds. Random and chronological test sets contain different people.

## Slide 16 — Results on the Chronological Test Set
Layout: sample-16.png
Section label: SECTION 6 · EVALUATION AND RESULTS
Presenter: 2
Content:
All metrics for 5 models on 2015-16 (validation-tuned threshold); best value per column marked with ** in the table file.
Visuals: tables/t07_temporal_test_results.md
Caption: Table 7: Performance on the 2015-2016 test set.
Big numbers: 0.791 — XGBoost ROC-AUC | 0.368 — XGBoost PR-AUC | 65.6% — XGBoost recall
Speaker notes:
The selected model was chosen on validation, not on this table. [REVIEW] Point out how close the five models are.

## Slide 17 — Model Performance Comparison
Layout: sample-17.png
Section label: SECTION 6 · EVALUATION AND RESULTS
Presenter: 2
Content:
Chart: ROC-AUC and PR-AUC per model with 95% bootstrap CIs (Eq. 7).
Reading the chart: [REVIEW] compare the interval widths and overlaps; paired-bootstrap differences are in results/paired_comparisons.csv.
Visuals: figures/f05_auc_ci.png | charts_data/c01_auc_ci.csv
Caption: Fig. 5: ROC-AUC and PR-AUC on 2015-2016 with 95% bootstrap CIs.
Big numbers: none
Speaker notes:
The x-axis does not start at zero; say so. Paired results are in results/paired_comparisons.csv.

## Slide 18 — Cross-Validation Results
Layout: sample-18.png
Section label: SECTION 6 · EVALUATION AND RESULTS
Presenter: 2
Content:
Per-fold ROC-AUC and mean ± std for protocol A (2011-12 training cycle).
Consistency check: XGBoost CV ROC-AUC 0.804 ± 0.007 vs chronological test 0.791.
Visuals: tables/t08_cv_results.md | charts_data/c11_cv_folds.csv
Caption: Table 8: 5-fold stratified CV ROC-AUC (protocol A).
Big numbers: none
Speaker notes:
A small standard deviation shows stable in-period results. [REVIEW] Compare the CV mean with the chronological test value.

## Slide 19 — Training Curves
Layout: sample-19.png
Section label: SECTION 6 · EVALUATION AND RESULTS
Presenter: 2
Content:
MLP train and validation loss and accuracy per epoch; early stopping at epoch 22 (patience 10).
Visuals: figures/f06_mlp_curves.png | charts_data/c02_mlp_history.csv
Caption: Fig. 6: MLP loss and accuracy curves (protocol A).
Big numbers: none
Speaker notes:
Curves are the real history of the evaluated MLP. Loss uses balanced sample weights, the same as in training.

## Slide 20 — Confusion Matrix and ROC Curves
Layout: sample-20.png
Section label: SECTION 6 · EVALUATION AND RESULTS
Presenter: 2
Content:
XGBoost confusion matrix on 2015-16 at threshold 0.60 (left); ROC curves of all models (right).
Visuals: figures/f07_confusion_matrix.png | figures/f08_roc_curves.png | charts_data/c04_confusion_matrix.csv | charts_data/c03_roc_points.csv
Caption: Fig. 7: Confusion matrix (selected model). Fig. 8: ROC curves on 2015-2016.
Big numbers: none
Speaker notes:
The threshold maximises F1 on validation. [REVIEW] Comment on false positives versus missed cases.

## Slide 21 — Ablation Study
Layout: sample-21.png
Section label: SECTION 6 · EVALUATION AND RESULTS
Presenter: 2
Content:
XGBoost retrained with one change at a time (same split and seed).
Visuals: tables/t09_ablation.md | figures/f09_ablation.png | charts_data/c05_ablation_deltas.csv
Caption: Table 9: Ablation on 2015-2016. Fig. 9: Change vs the full model.
Big numbers: -0.086 — ROC-AUC without demographics | -0.043 — without body measurements
Speaker notes:
[REVIEW] Explain which feature group matters most and why threshold tuning changes F1 but not AUC.

## Slide 22 — Temporal Drift Analysis
Layout: sample-21.png
Section label: SECTION 6 · EVALUATION AND RESULTS
Presenter: 2
Content:
KS per numeric feature and largest category shifts; prevalence 13.0% → 15.1%.
Gaps (selected model): random − chronological -0.004 [-0.033, +0.023]; A2 in-period − future +0.036 [-0.001, +0.070].
Visuals: figures/f10_feature_drift_ks.png | figures/f11_random_vs_temporal.png | tables/t10_period_gaps.md | charts_data/c06_ks_values.csv | charts_data/c07_category_shifts.csv | charts_data/c08_protocol_comparison.csv
Caption: Fig. 10: Feature drift vs 2011-2012. Fig. 11: Test ROC-AUC by protocol. Table 10: Period gaps with 95% CIs.
Big numbers: 0.057 — largest KS | 0.582 — domain classifier ROC-AUC
Speaker notes:
These statistics describe what changed, not why. [REVIEW] Discuss whether the gaps are conclusive given their CIs.

## Slide 23 — Comparison with Published Work
Layout: sample-22.png
Section label: SECTION 6 · EVALUATION AND RESULTS
Presenter: 2
Content:
Published TableShift BRFSS Diabetes accuracies next to ours.
Fairness note: different survey (BRFSS vs NHANES), different shift (race vs time), accuracy depends on thresholds — not directly comparable.
Visuals: tables/t11_literature_comparison.md
Caption: Table 11: Published values (copied from TableShift Table 6) and this work.
Big numbers: none
Speaker notes:
Values were copied exactly from the paper. [REVIEW] Explain why only the direction of the shift gap can be compared.

## Slide 24 — What the Results Mean
Layout: sample-23.png
Section label: SECTION 7 · DISCUSSION
Presenter: 2
Content:
Strengths: [REVIEW] real public data; three protocols; bootstrap CIs; fully reproducible notebook.
Limitations: survey not EHR; unweighted models; self-reported label; possible reverse causation.
Threats to validity: random vs chronological test sets contain different people; three cycles only; small grids.
Visuals: none
Caption: none
Big numbers: none
Speaker notes:
Interpret, do not repeat numbers. [REVIEW] Rewrite the strengths in your own words.

## Slide 25 — Conclusion and Future Work
Layout: sample-24.png
Section label: SECTION 8 · CONCLUSION AND FUTURE WORK
Presenter: 2
Content:
Conclusion: [REVIEW] summarise the main finding in one or two sentences using the numbers below.
Future work: 1 Add later NHANES cycles · 2 Survey-weighted training · 3 Calibration under prevalence shift · 4 Shift-aware model selection
Visuals: none
Caption: none
Big numbers: 0.791 — chronological ROC-AUC | 0.804 — random-split ROC-AUC (mean) | +0.036 — A2 in-period − future
Speaker notes:
Close on the main finding and two or three concrete next steps.

## Slide 26 — Code, Data and Academic Integrity
Layout: sample-25.png
Section label: SUBMISSION
Presenter: 2
Content:
1 Report: IEEE template, at least 8 pages, figures at 300 DPI
2 Notebook: CSCI447P034.ipynb runs top to bottom in an empty folder
3 Dataset: public CDC NHANES files, downloaded by the notebook (link in references)
4 Integrity: every source cited in IEEE style; course similarity and AI-use rules must be checked by the students
Visuals: none
Caption: none
Big numbers: none
Speaker notes:
Phrase these as requirements; do not claim similarity or AI-detection scores.

## Slide 27 — References
Layout: sample-26.png
Section label: SECTION 9 · REFERENCES
Presenter: 2
Content:
IEEE-style list from references.md (numbered in order of first citation).
Visuals: references.md
Caption: none
Big numbers: none
Speaker notes:
All references were opened and checked; see report/README.md for items to verify.

## Slide 28 — Thank You
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

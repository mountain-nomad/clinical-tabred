# Placeholders and items to check

## Names and IDs (keep exactly as written until replaced)
- `STUDENT_NAME_1`, `STUDENT_ID_1`, `STUDENT_NAME_2`, `STUDENT_ID_2` (slides 1 and 28, report, notebook title cell).
- Notebook file name `CSCI447P034.ipynb`: the sample deck's notes say the notebook must be named `<ProjectID>.ipynb` (i.e. possibly `P034.ipynb`). Check the course rule.

## TO VERIFY
- See `report/README.md` → 'To verify' (venues of arXiv papers, LightGBM page numbers).
- The project timeline (slide 6) is a plan; replace the week numbers with your real schedule.
- Course IEEE template was not provided; the report uses the standard IEEEtran class.
- Sample slide rendering: Rendered 27 slides with LibreOffice + pdftoppm.

## TODO slides
- None: every slide has real content.

## [REVIEW] passages (rewrite in your own words)
- Slide 3 — Abstract: - Conclusion: [REVIEW] Domain-classifier ROC-AUC 0.582; the random-vs-chronological gap is inconclusive (the 95% CI includes zero); the in-period-vs-future gap 
- Slide 3 — Abstract: One sentence each for aim, method, result and conclusion. [REVIEW] Explain in your own words why the CIs matter for the conclusion.
- Slide 4 — Problem, Motivation and Objectives: Introduce the deployment-over-time problem with the TabReD motivation [1]. [REVIEW] Say why diabetes in NHANES is a reasonable small test case.
- Slide 5 — Survey Cycles at a Glance: Takeaway: [REVIEW] Prevalence rises in the newest cycle while age and BMI change only slightly.
- Slide 7 — Related Work and Research Gap: Research gap: [REVIEW] few small, fully reproducible studies of temporal shift on public health-survey data.
- Slide 7 — Related Work and Research Gap: TabReD [1] motivates time-based splits; TableShift [2] includes a BRFSS diabetes task with a race-based shift. [REVIEW] State the gap in your own words.
- Slide 16 — Results on the Chronological Test Set: The selected model was chosen on validation, not on this table. [REVIEW] Point out how close the five models are.
- Slide 17 — Model Performance Comparison: Reading the chart: [REVIEW] compare the interval widths and overlaps; paired-bootstrap differences are in results/paired_comparisons.csv.
- Slide 18 — Cross-Validation Results: A small standard deviation shows stable in-period results. [REVIEW] Compare the CV mean with the chronological test value.
- Slide 20 — Confusion Matrix and ROC Curves: The threshold maximises F1 on validation. [REVIEW] Comment on false positives versus missed cases.
- Slide 21 — Ablation Study: [REVIEW] Explain which feature group matters most and why threshold tuning changes F1 but not AUC.
- Slide 22 — Temporal Drift Analysis: These statistics describe what changed, not why. [REVIEW] Discuss whether the gaps are conclusive given their CIs.
- Slide 23 — Comparison with Published Work: Values were copied exactly from the paper. [REVIEW] Explain why only the direction of the shift gap can be compared.
- Slide 24 — What the Results Mean: Strengths: [REVIEW] real public data; three protocols; bootstrap CIs; fully reproducible notebook.
- Slide 24 — What the Results Mean: Interpret, do not repeat numbers. [REVIEW] Rewrite the strengths in your own words.
- Slide 25 — Conclusion and Future Work: Conclusion: [REVIEW] summarise the main finding in one or two sentences using the numbers below.


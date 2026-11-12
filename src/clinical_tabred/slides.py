"""Write for_presentation/slides.md: slide-by-slide content following the sample deck.

All numbers are read from the key numbers (results folder); interpretive sentences are
marked [REVIEW] so the students rewrite them in their own words.
"""

from pathlib import Path

from clinical_tabred.plotting import MODEL_LABELS
from clinical_tabred.summaries import describe_interval


def slide(number: int, title: str, layout: int, label: str, presenter: int, content: str,
          visuals: str = "none", caption: str = "none", big: str = "none",
          notes: str = "") -> str:
    """One slide block in the exact format required by the presentation brief."""
    return (f"## Slide {number} — {title}\nLayout: sample-{layout:02d}.png\n"
            f"Section label: {label}\nPresenter: {presenter}\nContent:\n{content.strip()}\n"
            f"Visuals: {visuals}\nCaption: {caption}\nBig numbers: {big}\n"
            f"Speaker notes:\n{notes.strip()}\n")


def opening_slides(k: dict, project: dict) -> list[str]:
    """Slides 1-7: title, outline, abstract, introduction, background."""
    students = "\n".join(f"- {s['name']}, {s['id']}" for s in project["students"])
    best = MODEL_LABELS[k["best_model"]]
    return [
        slide(1, "Title", 1, "—", 1, f"""
- Course line: {project['course'].upper()} · FINAL PROJECT · {project['semester'].upper()}
- Project ID: {project['project_id']}
- Title: {project['title']}
- Subtitle: {project['full_title']}
- Submitted by:
{students}
- Submitted to: {project['instructor']}
- Footer: {project['course']}   |   Project ID: {project['project_id']}   |   {project['title']}""",
              notes="Introduce the team and the project title. The notebook file is "
                    f"{project['notebook_name']} (check the course naming rule)."),
        slide(2, "Presentation Roadmap", 2, "OUTLINE", 1, """
1 Abstract · 2 Introduction · 3 Background and Related Work · 4 Methodology / Project Approach ·
5 Implementation and System Design · 6 Evaluation and Results · 7 Discussion ·
8 Conclusion and Future Work · 9 References
Callout: Presenter 1: Sections 1–4 | Presenter 2: Sections 5–9""",
              notes="The outline follows the nine report sections. Presenter 1 covers sections "
                    "1–4 and Presenter 2 covers sections 5–9."),
        slide(3, "Abstract", 3, "SECTION 1 · ABSTRACT", 1, f"""
- Aim: Test whether tabular models trained on older NHANES cycles keep their performance on a newer cycle.
- Methodology: 5 models, chronological (2011-12 → 2013-14 → 2015-16) vs random vs in-period splits, 5-fold CV, bootstrap CIs, drift statistics.
- Results: {best} (selected on validation) ROC-AUC {k['best_temporal_roc_auc']:.3f} [{k['best_temporal_roc_auc_low']:.3f}, {k['best_temporal_roc_auc_high']:.3f}] on 2015-16; random-minus-chronological gap {k['random_minus_temporal_roc_auc_diff']:+.3f} [{k['random_minus_temporal_roc_auc_diff_low']:+.3f}, {k['random_minus_temporal_roc_auc_diff_high']:+.3f}].
- Conclusion: [REVIEW] Domain-classifier ROC-AUC {k['domain_auc_2015']:.3f}; the random-vs-chronological gap is {describe_interval(k['random_minus_temporal_roc_auc_diff_low'], k['random_minus_temporal_roc_auc_diff_high'])}; the in-period-vs-future gap is {describe_interval(k['a2_gap_roc_auc_diff_low'], k['a2_gap_roc_auc_diff_high'])}.""",
              notes="One sentence each for aim, method, result and conclusion. [REVIEW] Explain "
                    "in your own words why the CIs matter for the conclusion."),
        slide(4, "Problem, Motivation and Objectives", 4, "SECTION 2 · INTRODUCTION", 1, """
Problem: clinical models are trained on past patients but used on later ones; random splits can hide this time gap [1].
Objectives:
1 Compare random vs chronological evaluation — RQ1
2 Find which models transfer to a newer cohort — RQ2
3 Measure feature and prevalence changes between cycles — RQ3
4 Relate observed changes to performance — RQ4""",
              big=f"{k['prevalence_2015'] * 100:.1f}% — adults reporting diagnosed diabetes in "
                  "2015-2016 (unweighted sample, Table 1)",
              notes="Introduce the deployment-over-time problem with the TabReD motivation [1]. "
                    "[REVIEW] Say why diabetes in NHANES is a reasonable small test case."),
        slide(5, "Survey Cycles at a Glance", 5, "SECTION 2 · INTRODUCTION", 1, """
Table of the three cycles: adults, positives, unweighted and survey-weighted prevalence, mean age, mean BMI.
Takeaway: [REVIEW] Prevalence rises in the newest cycle while age and BMI change only slightly.""",
              visuals="tables/t01_cycle_context.md", caption="Table 1: Eligible adults and "
              "diabetes prevalence per NHANES cycle (computed from the data).",
              notes="All values are computed from the downloaded data, not taken from external "
                    "statistics. Weighted prevalence uses the interview weight WTINT2YR and is "
                    "descriptive only."),
        slide(6, "Project Timeline", 6, "SECTION 2 · INTRODUCTION", 1, """
Gantt chart of the planned work (weeks): literature review, download and codebook check,
data preparation, splits and baselines, tuning/CV/MLP, evaluation and ablation, drift analysis,
notebook/report/presentation.""",
              visuals="tables/t02_timeline.md | charts_data/c10_gantt.csv",
              caption="Table 2: Planned project timeline (same as the report).",
              notes="This is the planned schedule; adjust the weeks to your real dates. It must "
                    "match the timeline table in the report."),
        slide(7, "Related Work and Research Gap", 7, "SECTION 3 · BACKGROUND AND RELATED WORK", 1,
              """
Table of four related studies: approach, reference, data, split type, key finding.
Research gap: [REVIEW] few small, fully reproducible studies of temporal shift on public health-survey data.""",
              visuals="tables/t03_related_work.md",
              caption="Table 3: Related work (findings quoted from the cited abstracts).",
              notes="TabReD [1] motivates time-based splits; TableShift [2] includes a BRFSS "
                    "diabetes task with a race-based shift. [REVIEW] State the gap in your own words."),
    ]


def method_slides(k: dict) -> list[str]:
    """Slides 8-14: methodology, dataset, features, maths, stack, architecture, algorithm."""
    return [
        slide(8, "Proposed Methodology", 8, "SECTION 4 · METHODOLOGY / PROJECT APPROACH", 1, """
8 steps: 1 Download (validated XPT) → 2 Prepare (merge, label, filter) → 3 Features (9 inputs) →
4 Split (A / A2 / B) → 5 Cross-validation (5-fold) → 6 Training (5 models) → 7 Evaluation
(metrics, CIs, ablation) → 8 Drift (KS, PSI, domain classifier)""",
              visuals="diagrams/d01_methodology.svg", caption="Fig. 1: Proposed methodology.",
              notes="Walk through the eight steps left to right. Stress that test data is used "
                    "once and preprocessing is fitted on training data only."),
        slide(9, "Dataset Description", 9, "SECTION 4 · METHODOLOGY / PROJECT APPROACH", 1, """
CDC NHANES public files [5]: DEMO, BMX, DIQ for 2011-12 (G), 2013-14 (H), 2015-16 (I).
Target DIQ010: 1 = yes, 2 = no; 3/7/9/missing excluded. Adults ≥ 20 with an exam record.""",
              visuals="tables/t04_dataset.md", caption="Table 4: Dataset composition per cycle.",
              big=f"{k['n_total']:,.0f} — eligible adults | {k['prevalence_2011'] * 100:.1f}% → "
                  f"{k['prevalence_2015'] * 100:.1f}% — prevalence 2011-12 → 2015-16 | "
                  "9 — input features",
              notes="Data are downloaded from the CDC site inside the notebook; nothing is uploaded. "
                    "The data-flow table in the notebook lists every exclusion."),
        slide(10, "Features and Preprocessing", 10, "SECTION 4 · METHODOLOGY / PROJECT APPROACH",
              1, """
Table: feature, type, source file, missing % (2015-16), preprocessing.
Diagram: three feature groups → numeric / categorical pipelines → model input.""",
              visuals="tables/t05_features.md | diagrams/d04_features.svg",
              caption="Table 5: Features and preprocessing. Fig. 2: Feature groups and pipelines.",
              notes="Codes such as education 7/9 become missing. Laboratory diabetes markers and "
                    "survey weights are never used as inputs."),
        slide(11, "Mathematical Formulation", 11, "SECTION 4 · METHODOLOGY / PROJECT APPROACH", 1,
              """
Six numbered equations (1)–(6) from equations.md: logistic model, precision/recall, F1,
KS statistic, PSI, CV mean and standard deviation.""",
              visuals="equations.md", caption="Equations (1)–(6).",
              notes="Explain each symbol briefly. KS and PSI measure distribution change; the "
                    "bootstrap interval (7) is used on the results slides."),
        slide(12, "Technology Stack and Experimental Setup", 12,
              "SECTION 5 · IMPLEMENTATION AND SYSTEM DESIGN", 2, """
Table of tools and exact versions; hardware and reproducibility notes (seed, n_jobs, CPU only).""",
              visuals="tables/t06_tech_stack.md", caption="Table 6: Technology stack (versions "
              "recorded at run time).",
              notes="Every version is pinned in requirements.txt and recorded in "
                    "results/run_metadata.json; the notebook installs only missing packages."),
        slide(13, "System Architecture", 13, "SECTION 5 · IMPLEMENTATION AND SYSTEM DESIGN", 2, """
Diagram: ColumnTransformer → 5 models → threshold (max F1 on validation) → metrics.
3 steps: 1 Fit preprocessing on training split only · 2 Tune each model on validation PR-AUC ·
3 Report test metrics at 0.5 and tuned threshold with bootstrap CIs""",
              visuals="diagrams/d02_architecture.svg", caption="Fig. 3: System architecture.",
              notes="All five models share the same preprocessing. Class imbalance is handled "
                    "with balanced weights or scale_pos_weight."),
        slide(14, "Core Algorithm", 14, "SECTION 5 · IMPLEMENTATION AND SYSTEM DESIGN", 2, """
Algorithm 1 pseudocode (algorithm.md) with grouped explanations on the right.""",
              visuals="algorithm.md", caption="Algorithm 1: Temporal benchmark for one protocol.",
              notes="Explain the groups, not every line: tuning loop, MLP epoch loop with early "
                    "stopping, threshold choice on validation, single test evaluation."),
    ]


def result_slides(k: dict) -> list[str]:
    """Slides 15-23: experimental design and all results."""
    best = MODEL_LABELS[k["best_model"]]
    return [
        slide(15, "Experimental Design: Split and Cross-Validation", 15,
              "SECTION 6 · EVALUATION AND RESULTS", 2, """
A chronological: train 2011-12 → validate 2013-14 → test 2015-16 (once).
A2 in-period: same model, 20% holdout of 2011-12 vs 2015-16.
B random: pooled 70/15/15, 5 seeds. CV: 5-fold stratified on development data only.
Why both? CV measures in-period (i.i.d.) stability; chronological testing measures transfer to a later cohort.""",
              visuals="diagrams/d03_protocols.svg", caption="Fig. 4: Evaluation protocols.",
              notes="Say clearly that the test set is never used for tuning or thresholds. Random "
                    "and chronological test sets contain different people."),
        slide(16, "Results on the Chronological Test Set", 16, "SECTION 6 · EVALUATION AND RESULTS",
              2, "All metrics for 5 models on 2015-16 (validation-tuned threshold); best value per "
                 "column marked with ** in the table file.",
              visuals="tables/t07_temporal_test_results.md",
              caption="Table 7: Performance on the 2015-2016 test set.",
              big=f"{k['best_temporal_roc_auc']:.3f} — {best} ROC-AUC | "
                  f"{k['best_temporal_pr_auc']:.3f} — {best} PR-AUC | "
                  f"{k['best_temporal_recall'] * 100:.1f}% — {best} recall",
              notes="The selected model was chosen on validation, not on this table. [REVIEW] "
                    "Point out how close the five models are."),
        slide(17, "Model Performance Comparison", 17, "SECTION 6 · EVALUATION AND RESULTS", 2, """
Chart: ROC-AUC and PR-AUC per model with 95% bootstrap CIs (Eq. 7).
Reading the chart: [REVIEW] compare the interval widths and overlaps; paired-bootstrap differences are in results/paired_comparisons.csv.""",
              visuals="figures/f05_auc_ci.png | charts_data/c01_auc_ci.csv",
              caption="Fig. 5: ROC-AUC and PR-AUC on 2015-2016 with 95% bootstrap CIs.",
              notes="The x-axis does not start at zero; say so. Paired results are in "
                    "results/paired_comparisons.csv."),
        slide(18, "Cross-Validation Results", 18, "SECTION 6 · EVALUATION AND RESULTS", 2, f"""
Per-fold ROC-AUC and mean ± std for protocol A (2011-12 training cycle).
Consistency check: {best} CV ROC-AUC {k['cv_temporal_roc_auc_mean']:.3f} ± {k['cv_temporal_roc_auc_std']:.3f} vs chronological test {k['best_temporal_roc_auc']:.3f}.""",
              visuals="tables/t08_cv_results.md | charts_data/c11_cv_folds.csv",
              caption="Table 8: 5-fold stratified CV ROC-AUC (protocol A).",
              notes="A small standard deviation shows stable in-period results. [REVIEW] Compare "
                    "the CV mean with the chronological test value."),
        slide(19, "Training Curves", 19, "SECTION 6 · EVALUATION AND RESULTS", 2,
              f"MLP train and validation loss and accuracy per epoch; early stopping at epoch "
              f"{k['mlp_best_epoch']:.0f} (patience 10).",
              visuals="figures/f06_mlp_curves.png | charts_data/c02_mlp_history.csv",
              caption="Fig. 6: MLP loss and accuracy curves (protocol A).",
              notes="Curves are the real history of the evaluated MLP. Loss uses balanced sample "
                    "weights, the same as in training."),
        slide(20, "Confusion Matrix and ROC Curves", 20, "SECTION 6 · EVALUATION AND RESULTS", 2,
              f"{best} confusion matrix on 2015-16 at threshold {k['best_temporal_threshold']:.2f}"
              " (left); ROC curves of all models (right).",
              visuals="figures/f07_confusion_matrix.png | figures/f08_roc_curves.png | "
                      "charts_data/c04_confusion_matrix.csv | charts_data/c03_roc_points.csv",
              caption="Fig. 7: Confusion matrix (selected model). Fig. 8: ROC curves on 2015-2016.",
              notes="The threshold maximises F1 on validation. [REVIEW] Comment on false "
                    "positives versus missed cases."),
        slide(21, "Ablation Study", 21, "SECTION 6 · EVALUATION AND RESULTS", 2, f"""
{best} retrained with one change at a time (same split and seed).""",
              visuals="tables/t09_ablation.md | figures/f09_ablation.png | "
                      "charts_data/c05_ablation_deltas.csv",
              caption="Table 9: Ablation on 2015-2016. Fig. 9: Change vs the full model.",
              big=f"{k['ablation_demographic_delta_roc_auc']:+.3f} — ROC-AUC without demographics"
                  f" | {k['ablation_anthropometric_delta_roc_auc']:+.3f} — without body measurements",
              notes="[REVIEW] Explain which feature group matters most and why threshold tuning "
                    "changes F1 but not AUC."),
        slide(22, "Temporal Drift Analysis", 21, "SECTION 6 · EVALUATION AND RESULTS", 2, f"""
KS per numeric feature and largest category shifts; prevalence {k['prevalence_2011'] * 100:.1f}% → {k['prevalence_2015'] * 100:.1f}%.
Gaps (selected model): random − chronological {k['random_minus_temporal_roc_auc_diff']:+.3f} [{k['random_minus_temporal_roc_auc_diff_low']:+.3f}, {k['random_minus_temporal_roc_auc_diff_high']:+.3f}]; A2 in-period − future {k['a2_gap_roc_auc_diff']:+.3f} [{k['a2_gap_roc_auc_diff_low']:+.3f}, {k['a2_gap_roc_auc_diff_high']:+.3f}].""",
              visuals="figures/f10_feature_drift_ks.png | figures/f11_random_vs_temporal.png | "
                      "tables/t10_period_gaps.md | charts_data/c06_ks_values.csv | "
                      "charts_data/c07_category_shifts.csv | charts_data/c08_protocol_comparison.csv",
              caption="Fig. 10: Feature drift vs 2011-2012. Fig. 11: Test ROC-AUC by protocol. "
                      "Table 10: Period gaps with 95% CIs.",
              big=f"{k['max_ks_2015']:.3f} — largest KS | {k['domain_auc_2015']:.3f} — domain "
                  "classifier ROC-AUC",
              notes="These statistics describe what changed, not why. [REVIEW] Discuss whether "
                    "the gaps are conclusive given their CIs."),
        slide(23, "Comparison with Published Work", 22, "SECTION 6 · EVALUATION AND RESULTS", 2, """
Published TableShift BRFSS Diabetes accuracies next to ours.
Fairness note: different survey (BRFSS vs NHANES), different shift (race vs time), accuracy depends on thresholds — not directly comparable.""",
              visuals="tables/t11_literature_comparison.md",
              caption="Table 11: Published values (copied from TableShift Table 6) and this work.",
              notes="Values were copied exactly from the paper. [REVIEW] Explain why only the "
                    "direction of the shift gap can be compared."),
    ]


def closing_slides(k: dict, project: dict) -> list[str]:
    """Slides 24-28: discussion, conclusion, submission, references, thank you."""
    names = " · ".join(s["name"] for s in project["students"])
    return [
        slide(24, "What the Results Mean", 23, "SECTION 7 · DISCUSSION", 2, """
Strengths: [REVIEW] real public data; three protocols; bootstrap CIs; fully reproducible notebook.
Limitations: survey not EHR; unweighted models; self-reported label; possible reverse causation.
Threats to validity: random vs chronological test sets contain different people; three cycles only; small grids.""",
              notes="Interpret, do not repeat numbers. [REVIEW] Rewrite the strengths in your own "
                    "words."),
        slide(25, "Conclusion and Future Work", 24, "SECTION 8 · CONCLUSION AND FUTURE WORK", 2, """
Conclusion: [REVIEW] summarise the main finding in one or two sentences using the numbers below.
Future work: 1 Add later NHANES cycles · 2 Survey-weighted training · 3 Calibration under prevalence shift · 4 Shift-aware model selection""",
              big=f"{k['best_temporal_roc_auc']:.3f} — chronological ROC-AUC | "
                  f"{k['random_roc_auc_mean']:.3f} — random-split ROC-AUC (mean) | "
                  f"{k['a2_gap_roc_auc_diff']:+.3f} — A2 in-period − future",
              notes="Close on the main finding and two or three concrete next steps."),
        slide(26, "Code, Data and Academic Integrity", 25, "SUBMISSION", 2, f"""
1 Report: IEEE template, at least 8 pages, figures at 300 DPI
2 Notebook: {project['notebook_name']} runs top to bottom in an empty folder
3 Dataset: public CDC NHANES files, downloaded by the notebook (link in references)
4 Integrity: every source cited in IEEE style; course similarity and AI-use rules must be checked by the students""",
              notes="Phrase these as requirements; do not claim similarity or AI-detection scores."),
        slide(27, "References", 26, "SECTION 9 · REFERENCES", 2,
              "IEEE-style list from references.md (numbered in order of first citation).",
              visuals="references.md", caption="none",
              notes="All references were opened and checked; see report/README.md for items to "
                    "verify."),
        slide(28, "Thank You", 27, "—", 2, f"""
Thank You
Questions and discussion
{names}   |   Project ID: {project['project_id']}   |   CSCI 447, Nazarbayev University""",
              notes="Invite questions."),
    ]


def write_slides_md(numbers: dict, project: dict, out_path: Path) -> Path:
    """Assemble all 28 slides into slides.md."""
    blocks = (opening_slides(numbers, project) + method_slides(numbers)
              + result_slides(numbers) + closing_slides(numbers, project))
    header = ("# Slide content — Clinical TabReD Lite\n\nEvery number below is generated from "
              "the results folder. Sentences marked [REVIEW] must be rewritten by the students.\n\n")
    Path(out_path).write_text(header + "\n".join(blocks))
    return Path(out_path)

# Decisions

Each entry: **decision** → alternatives → reason.

Entries 1-32 belong to the original 9-feature NHANES version; its pipeline and outputs (`results/`,
report, old notebook and presentation pack) were removed on 16 Nov 2026 (decision 41) and remain in
the git history (commit `1dc9dba`).

## Environment and inputs
1. **Python 3.12 virtual environment (`.venv`)** → system Python 3.14 / conda → 3.12 has stable
   wheels for every dependency; versions pinned in `requirements.txt`.
2. **Installed `libomp`, `poppler`, `tectonic` and LibreOffice with Homebrew** → skip PDF rendering
   and LaTeX → xgboost needs `libomp` on macOS; `pdftoppm`/`soffice` render the sample slides;
   `tectonic` compiles the report because `latexmk`/`pdflatex` were not installed.
   Keynote export via AppleScript was tried first and timed out (likely a permission dialog).
3. **Sample deck copied from `~/Downloads` into `for_presentation/reference/`** → reference it in
   place → the pack must be self-contained.
4. **Existing one-line stub files in `src/` were replaced** by the layout requested in the brief.

## Data
5. **Nine base features only** (no extra features) → add e.g. arm circumference or blood pressure →
   the base set is verified in all three codebooks and keeps the model easy to explain.
6. **Adults without a BMX (body-measurement) record are excluded** (interview-only respondents) →
   keep them and impute all four body measurements → imputing an entire feature group for
   these adults is less defensible than excluding them (counts per cycle:
   `results/missing_components.csv`); the exclusion is recorded in
   `results/data_flow.csv` and `results/missing_components.csv`.
7. **Remaining missing values are imputed inside the model pipeline** (median / most frequent),
   without missing-indicator columns → indicators or dropping rows → simple and fitted on training
   data only; rates per feature and cycle are in `results/missingness.csv`.
8. **Weighted prevalence uses `WTINT2YR`** (interview weight) → `WTMEC2YR` → the diabetes question
   is part of the interview. It is descriptive only, computed on the analysed sample (not a
   design-correct population estimate).
9. **Raw XPT files are git-ignored**; `data/raw/manifest.csv` (URL, status, size, SHA-256) is
   committed → commit raw files → files are public and re-downloadable; the hash proves identity.

## Models and training
10. **Same preprocessing (including scaling) for all models** → no scaling for trees → the brief
    asks for one pipeline; scaling does not change tree splits.
11. **Grids:** LR `C` ∈ {0.01, 0.1, 1, 10}; RF depth {6, 10, None} × min leaf {1, 10}; XGBoost
    lr {0.03, 0.1} × depth {3, 5}; LightGBM lr {0.03, 0.1} × leaves {7, 31}; MLP layers {(64),
    (64, 32)} × alpha {1e-4, 1e-2}. Selection by validation PR-AUC (ROC-AUC tiebreak).
12. **MLP imbalance handling: "balanced" sample weights in `partial_fit`** (supported in
    scikit-learn 1.9) → random oversampling → same idea as `class_weight="balanced"` for LR.
    Train/validation log loss in `mlp_history.csv` is computed with the same balanced weights;
    accuracy is unweighted at threshold 0.5. Adam, learning rate 1e-3, batch 128, max 200 epochs,
    patience 10, minimum improvement 1e-4, best-epoch weights restored.
13. **LightGBM `deterministic=True, force_row_wise=True`** → defaults → reproducible results across runs.
14. **No refit on train + validation in protocol A** (as required by the brief) → refit → keeps
    the temporal setup clean.
15. **Protocol A2 reuses protocol A's hyperparameters**; early stopping and the threshold use the
    2013-2014 validation cycle → tune A2 separately → the only difference between the two A2 test
    sets is then the period.
16. **Protocol B is tuned separately for each of 5 seeds**; bootstrap CIs, paired tests and gaps
    use the seed-42 split only → bootstrap every seed → runtime; seed variation is reported as
    mean ± std in `random_seeds_summary.csv`.
17. **Cross-validation uses the selected hyperparameters of each protocol; the MLP trains for the
    best epoch found in the main run; threshold-based CV metrics use 0.5** → nested tuning → no
    validation data exists inside a fold; ranking metrics (ROC/PR-AUC) are threshold-free.
    Protocol B's CV uses train + validation of the seed-42 split.
18. **`metrics.csv` format:** one row per protocol × seed × model × split × threshold rule, with
    one column per metric (a "long" table over experiments, wide over metrics) → fully melted
    → easier to read and filter.

## Evaluation and drift
19. **Percentile bootstrap (1000 resamples, seed 42)**; paired for models on the same test set,
    independent resampling for differences between different test sets.
20. **Ablation re-tunes the threshold on validation for every variant except "threshold 0.5"** →
    keep the full model's threshold → each variant is evaluated as it would be deployed.
21. **Domain classifier = logistic regression with the shared preprocessing, 5-fold CV ROC-AUC** →
    gradient boosting → simple and easy to explain; PSI uses 10 quantile bins fixed on 2011-2012.

## Code organisation
22. **Extra modules beyond the suggested layout:** `experiment.py` (tuning, protocol runs, CV,
    ablation), `benchmark.py` (training stage + saving), `summaries.py` (traceable key numbers),
    `diagrams.py`, `slides.py`, `presentation_docs.py` → put everything in `run.py` and
    `presentation_pack.py` → keeps files short (≤ ~250 lines) and lets the notebook reuse the same
    functions.
23. **Notebook templates** use `<!-- source: module.function -->` directives and fenced code blocks;
    the builder copies function source with `inspect` and constants with `ast`.

## Notebook
24. **File name `CSCI447P034.ipynb`** as given in the brief; the sample deck's notes say
    `ProjectID.ipynb` (possibly `P034.ipynb`) → listed in `for_presentation/PLACEHOLDERS.md`.
25. **Notebook vs repository metrics:** compared with `scripts/compare_notebook_metrics.py` on all
    metric rows plus the ablation, CV, CI and drift tables, with an absolute tolerance of 1e-9;
    the measured maximum difference is written to `results/notebook_consistency.json`.
26. **Notebook conclusion text is not written for the students**; the notebook prints rule-based
    findings computed from the results and leaves the interpretation to them.

## Report and presentation
27. **IEEEtran conference class** (course template not provided) → wait for template → documented
    in `report/README.md`.
28. **Literature comparison uses TableShift's BRFSS Diabetes accuracies** (Table 6, copied from the
    arXiv HTML) → TabReD numbers → TabReD has no comparable health dataset; the table flags the
    dataset, shift and metric differences.
29. **Timeline is a plan** (`report/timeline.csv`), clearly labelled for students to adjust.
30. **Model colours** are deck hues adjusted after a colour-vision-deficiency check
    (blue #1F5FA8, gold #A8822C, teal #109A92, rust #B4432E, purple #6B4E9B).
31. **Slide figure numbering:** Fig. 1–4 are diagrams (`diagrams/d01–d04`), so slide figures are
    `figures/f05–f11`, matching their figure numbers in `slides.md`.
32. **`--quick` never runs the report, presentation or notebook stages** (and refuses them when
    requested explicitly) → write quick versions to separate folders → these stages write fixed
    project files; a smoke test overwrote them once during development, so they are now guarded.

## Three-dataset benchmark (16 Nov 2026)
33. **BRFSS and natality are subsampled to 50,000 eligible rows per year (seed 42)** → use all rows
    (~3M / ~33M) → TabReD also subsampled its large datasets; keeps tuning feasible. Natality is
    streamed through `unzip -p` with reservoir sampling (zips use Deflate64; 5 GB per year unzipped).
34. **Feature counts were not targeted** → pad BRFSS/natality to TabReD's range (odd-year questions,
    engineered aggregates) → every leakage-free feature present in all years is kept, including the
    agencies' derived variables (CDC calculated variables, NCHS recodes/imputation flags), as TabReD
    keeps production feature sets; nothing is added just to raise the count.
35. **Natality target = preterm birth (OE gestation < 37 weeks) from pre-delivery items only** →
    birth weight / NICU / cesarean targets → preterm is the standard birth-certificate prediction task;
    every item recorded at or after delivery (and 3rd-trimester smoking, prenatal visit count, weight
    gain) is excluded as leakage.
36. **Natality windows test 2020, 2022, 2024; default split 2016-21 / 2022 / 2023-24** → contiguous
    windows → 9 years allow spaced windows; one window tests the COVID-19 year; default is ~80/20.
37. **Same-period random-vs-time comparison added** → only TabReD's comparison → TabReD's random and
    time test sets contain different periods; when later periods are easier (natality) the sign of the
    difference flips, so the random split is also scored on its rows from the time test period.
38. **CV uses the default split's development periods and the selected hyperparameters** → nested CV
    → same rule as #17 of the original project; the test periods are never used.
39. **Jobs are run one at a time** → in parallel → two jobs each starting one thread per core slowed
    XGBoost/LightGBM 10-50x (thread oversubscription).
40. **Domain classifier subsamples each side to 20,000 rows** → all rows → runtime; same logistic
    regression + 5-fold CV as the original project.
41. **Removed the original 9-feature version from the working tree** (16 Nov) → keep both → the old
    pipeline (`clinical_tabred.run`) still wrote `results/`, `figures/`, `report/` and
    `for_presentation/`, so running it would have overwritten the new material; everything stays in
    the git history. Folders/scripts renamed without the `wide` prefix (`results_nhanes/`, `figures/`,
    `run_benchmark.py`, ...).
42. **The notebook is generated from the repository code and runs on saved results by default** →
    a hand-written notebook that re-runs everything → the full benchmark takes ~3 h; by default the
    notebook re-trains the selected configuration of every model on the default split and compares
    with the saved results (14/15 identical; NHANES XGBoost differs by 0.0004 because the NHANES
    benchmark ran on another computer), and `RUN_EXPERIMENTS = True` re-runs everything.

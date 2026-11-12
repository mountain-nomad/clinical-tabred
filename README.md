# Clinical TabReD Lite

**Evaluating Tabular Machine Learning Models under Temporal Distribution Shift in Healthcare Data**
CSCI 447 Machine Learning: Theory and Practice · Fall 2026 · Project P034

## Objective
A small, reproducible benchmark (inspired by TabReD, Rubachev et al., 2024) that compares Logistic
Regression, Random Forest, XGBoost, LightGBM and an MLP on three NHANES survey cycles and asks:
1. How does performance differ between random and chronological splits?
2. Which models generalise better to a newer survey cohort?
3. Which feature distributions and target prevalences change between cycles?
4. Can these observed changes help explain performance differences?

This is an educational prototype, not a validated clinical prediction system.
Main results: see `results/model_comparison.md` (generated) and the figures in `figures/`.

## Dataset
CDC National Health and Nutrition Examination Survey (NHANES), public-use files:
DEMO, BMX and DIQ for 2011-2012 (G), 2013-2014 (H) and 2015-2016 (I) from
<https://wwwn.cdc.gov/nchs/nhanes/>. Target: `DIQ010` (self-reported diagnosed diabetes; adults ≥ 20).
NHANES data are public-use; cite them as "Centers for Disease Control and Prevention, National Center
for Health Statistics, National Health and Nutrition Examination Survey" with the URL above.
Variables, coding and exclusions: `docs/FEATURES.md`.

## Installation
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
# macOS only: xgboost/lightgbm need OpenMP
brew install libomp
```
Optional tools: `tectonic` or `latexmk` (report PDF), LibreOffice + `pdftoppm` (sample-slide images).

## Data download
```bash
python -m clinical_tabred.run --config configs/experiment.yaml --stage download
```
Files are validated (no HTML pages, SAS transport header, readable by pandas) and cached in
`data/raw/`; `data/raw/manifest.csv` records URL, HTTP status, size and SHA-256.

## Reproducing the experiment
```bash
python -m clinical_tabred.run --config configs/experiment.yaml --stage all      # everything
python -m clinical_tabred.run --config configs/experiment.yaml --stage train    # one stage
python -m clinical_tabred.run --config configs/experiment.yaml --stage all --quick   # smoke test → results_quick/
pytest -q                     # tests (no network); add --run-network for the real-download test
ruff check src tests scripts
```
Stages: `download, prepare, train, drift, figures, report, presentation, notebook`. Runtimes of the
last run are stored in `results/run_metadata.json` (with package versions, seed and hardware).

## The standalone notebook
`notebooks/CSCI447P034.ipynb` is the submitted file. It is **generated** from the package source
(`notebooks/templates/*.md` + `inspect.getsource`), so notebook and repository use identical code.
```bash
python -m clinical_tabred.run --stage notebook            # rebuild after any code change
mkdir /tmp/nb_standalone && cp notebooks/CSCI447P034.ipynb /tmp/nb_standalone/ && cd /tmp/nb_standalone
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1800 \
    CSCI447P034.ipynb --output CSCI447P034.ipynb          # needs internet; installs missing packages
python <repo>/scripts/compare_notebook_metrics.py /tmp/nb_standalone/notebook_outputs <repo>/results
```
It downloads data into `./nhanes_data/` and writes all outputs to `./notebook_outputs/`.

## Regenerating `for_presentation/`
```bash
python -m clinical_tabred.run --stage presentation
```
The folder is self-contained (slides.md, style guide, tables, chart data, figures, diagrams,
references, sample deck + rendered sample slides). See `for_presentation/README.md`.

## Output files
| File | Content |
|---|---|
| `results/data_flow.csv` | Attrition per step and cycle (rows kept/removed, reason) |
| `results/missingness.csv`, `missing_components.csv` | Missing % per feature/cycle; adults without BMX/DIQ |
| `results/splits.csv` | Size and prevalence of every split |
| `results/metrics.csv` | One row per protocol × seed × model × split × threshold rule; all metrics, confusion counts, timing |
| `results/metrics_ci.csv` | Bootstrap 95% CIs of ROC-AUC and PR-AUC |
| `results/paired_comparisons.csv` | Paired bootstrap: selected model minus each other model (2015-2016) |
| `results/period_gaps.csv` | A2 in-period minus future, and random minus chronological, with CIs |
| `results/random_seeds_summary.csv` | Protocol B mean ± std over 5 seeds |
| `results/cv_folds.csv`, `cv_metrics.csv` | 5-fold CV per fold and mean ± std |
| `results/tuning.csv`, `selected_models.json` | Every tried configuration; chosen parameters and thresholds |
| `results/mlp_history.csv` | MLP loss/accuracy per epoch, early-stopping epoch |
| `results/ablation.csv` | Ablation of the selected model (deltas vs full model) |
| `results/cycle_summary.csv`, `drift_numeric.csv`, `drift_categorical.csv`, `domain_classifier.csv` | Drift analysis |
| `results/predictions/<protocol>_<model>.csv` | Test labels, probabilities, cycle |
| `results/related_work.csv`, `literature_comparison.csv` | Literature tables (values copied from opened sources) |
| `results/model_comparison.md` | Readable summary with computed findings |
| `figures/fig01–fig09.{png,pdf}`, `figures/captions.md` | Figures (300 DPI) and captions |
| `report/main.tex`, `report/main.pdf` | IEEE report (tables generated into `report/tables/`) |

## Limitations
* NHANES is a cross-sectional population survey, not longitudinal hospital EHR data.
* Models are trained without survey weights (weighted prevalence is reported only descriptively).
* The label is self-reported diagnosed diabetes; undiagnosed diabetes counts as negative.
* Body measurements are taken after a possible diagnosis (possible reverse causation).
* Random and chronological test sets contain different people, so their difference is not purely temporal.
* Three cycles, nine features and small grids; observed changes are described, not explained causally.

## Differences from the full TabReD benchmark
The TabReD repository (`yandex-research/tabred`) contains `paper/` (code to reproduce the paper),
`preprocessing/` (dataset preparation) and `src/tabred/` (a downloader package/CLI) for eight
industry-grade datasets with a default split plus `random-{0,1,2}` and `sliding-window-{0,1,2}`
splits, and evaluates many deep-learning and GBDT methods. This project instead uses three public
NHANES cycles (sizes in `results/cycle_summary.csv`), nine features, five standard models with small grids, CPU-only
training, and adds an in-period reference protocol and bootstrap CIs. Its datasets were not downloaded.

## Project documents
`DECISIONS.md` (choices and reasons) · `docs/FEATURES.md` · `docs/WALKTHROUGH.md` · `report/README.md`.

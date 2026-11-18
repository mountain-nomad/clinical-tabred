# Clinical TabReD

**Evaluating Tabular Machine Learning Models under Temporal Distribution Shift in Healthcare Data**
CSCI 447 Machine Learning: Theory and Practice · Fall 2026 · Project P034

A clinical version of the TabReD benchmark (Rubachev et al., 2024): three public US health data
sources with a time axis, five tabular models, TabReD's time-based and random splits, and a drift
analysis in which every drift is linked to its clinical cause.

| Dataset | Target | Periods | Rows x features |
|---|---|---|---|
| NHANES (interview + exam + lab) | diagnosed diabetes | 7 two-year cycles 2005-2018 | 37,287 x 280 |
| BRFSS (telephone survey) | diagnosed diabetes | 7 years 2017-2023 | 350,000 x 93 |
| NCHS natality (birth certificates) | preterm birth (< 37 weeks) | 9 years 2016-2024 | 450,000 x 73 |

**Results:** [`docs/RESULTS.md`](docs/RESULTS.md) · tables: `results_summary/` · figures: `figures/` ·
notebook: [`notebooks/CSCI447P034.ipynb`](notebooks/CSCI447P034.ipynb) · presentation material:
`for_presentation/` · data building and decisions: [`docs/HANDOFF_EXPANDED_NHANES.md`](docs/HANDOFF_EXPANDED_NHANES.md),
[`docs/HANDOFF_BRFSS_NATALITY.md`](docs/HANDOFF_BRFSS_NATALITY.md), [`DECISIONS.md`](DECISIONS.md).

This is an educational benchmark, not a validated clinical prediction system.

## Repository layout
| Path | Content |
|---|---|
| `scripts/build_nhanes_table.py`, `parse_brfss_codebooks.py`, `build_brfss_table.py`, `build_natality_table.py` | raw files -> wide tables in `data/processed/` |
| `scripts/leakage_audit.py` | univariate and missingness ROC-AUC of every feature (leakage screen) |
| `scripts/run_benchmark.py` | 5 models x 7 splits per dataset -> `results_<dataset>/` |
| `scripts/run_cv.py` | 5-fold CV on the default split's development periods |
| `scripts/analyze_drift.py` | domain classifier, per-feature drift, documented clinical-cause events |
| `scripts/summarize_benchmarks.py` | all datasets -> `results_summary/*.csv` + `summary.md` |
| `scripts/make_figures.py`, `make_diagrams.py` | `figures/` (PNG 300 DPI, PDF, SVG; captions in `figures/captions.md`) |
| `scripts/make_presentation_pack.py` | tables, chart data, figures and key numbers in `for_presentation/` |
| `scripts/build_notebook.py` | generates `notebooks/CSCI447P034.ipynb` from the code above |
| `src/clinical_tabred/` | shared code: preprocessing + models, tuning/evaluation, metrics + bootstrap, PSI, plot style |
| `configs/experiment.yaml` | seed, hyperparameter grids, CV, bootstrap and MLP settings |

## Installation
```bash
python3.12 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Data
Raw files go to `data/raw/` (not in git): NHANES `scripts/download_nhanes.ps1` (196 XPT files),
BRFSS `scripts/download_brfss.ps1` (LLCP 2017-2023 zips + codebooks in `data/raw/brfss/codebooks/`),
natality: `Nat<YEAR>us.zip` 2016-2024 from the NCHS Vital Statistics Online page into
`data/raw/natality/` (never unzip them: ~5 GB each; the build script streams them).
The processed BRFSS and natality tables are in git (`data/processed/*.csv.gz`); the NHANES table is
rebuilt with `python scripts/build_nhanes_table.py`.

## Reproducing the experiment
```bash
for d in nhanes brfss natality; do
  python scripts/leakage_audit.py  --dataset $d
  python scripts/run_benchmark.py  --dataset $d      # ~10-80 min per dataset on 16 cores
  python scripts/run_cv.py         --dataset $d
  python scripts/analyze_drift.py  --dataset $d
done
python scripts/summarize_benchmarks.py && python scripts/make_figures.py && python scripts/make_diagrams.py
python scripts/make_presentation_pack.py
pytest -q
```
Run the jobs **one at a time**: XGBoost, LightGBM and scikit-learn each use all cores, and two jobs in
parallel slowed both down 10-50x.

## The notebook
`notebooks/CSCI447P034.ipynb` is the submitted notebook: the whole procedure (data loading,
preprocessing and leakage audit, EDA, splits, training, CV, evaluation, drift, results) with the real
code of `scripts/` and `src/`, copied in by `scripts/build_notebook.py`. Run it from `notebooks/`:
by default it uses the saved results and re-trains the selected configuration of every model on the
default split to check that they are reproduced (~10 min); `RUN_EXPERIMENTS = True` re-runs everything
(~3 h). It writes its tables and figures to `notebooks/outputs/`.
```bash
python scripts/build_notebook.py
cd notebooks && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 CSCI447P034.ipynb
```

## Outputs per dataset (`results_nhanes/`, `results_brfss/`, `results_natality/`)
| File | Content |
|---|---|
| `metrics.csv` | one row per split x model x part (train/val/test) x threshold rule: all metrics, confusion counts, timing |
| `tuning.csv`, `selected_models.json` | every tried configuration; selected parameters, threshold, MLP best epoch |
| `predictions/test_predictions.csv(.gz)` | test labels and probabilities of every split and model |
| `default_test_ci.csv`, `default_paired_comparisons.csv` | bootstrap CIs and paired model differences, default split |
| `random_minus_time_by_window.csv`, `ranking_time_vs_random.csv` | TabReD random-vs-time comparison |
| `cv_folds.csv`, `cv_summary.csv` | 5-fold CV |
| `drift_*.csv` | periods, domain classifier, feature drift, clinical-cause events, drift vs performance |
| `leakage_audit.csv` | leakage screen |

## Limitations
One seed and small grids; no CatBoost / MLP-PLR; BRFSS and natality have fewer features than
TabReD's datasets; survey weights are not used; NHANES and BRFSS labels are self-reported diagnoses;
causal explanations of drift are hypotheses consistent with the data. See `docs/RESULTS.md`, section 4.

The original 9-feature NHANES 2011-2016 version of the project (report, notebook, presentation pack)
is in the git history (commit `1dc9dba`, before this version).

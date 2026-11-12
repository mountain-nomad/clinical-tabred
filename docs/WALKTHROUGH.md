# Code walkthrough

The pipeline is one Python package, `src/clinical_tabred/`. Each stage of the command line
(`python -m clinical_tabred.run --stage <name>`) calls a few functions; every function receives
its paths and settings as arguments, so the notebook can reuse it unchanged.

```text
download → prepare → train (splits, models, evaluation, CV, ablation, bootstrap) → drift
        → figures → report → presentation → notebook
```

## Data stages

| Module | Purpose | Key functions | Inputs → outputs |
|---|---|---|---|
| `download.py` | Get the 9 NHANES XPT files safely | `build_urls`, `validate_xpt`, `fetch_url`, `download_component`, `download_all` | config → `data/raw/*.xpt`, `data/raw/manifest.csv` |
| `prepare.py` | Build one clean table of adults | `read_xpt` (rounds 5.4e-79 to 0, checks SEQN), `merge_cycle` (left join from DEMO), `map_target` (1→1, 2→0, rest excluded), `clean_special_codes`, `filter_cycle` (attrition rows), `build_dataset`, `check_prevalence`, `save_processed`, `feature_matrix` | raw XPT → `data/processed/nhanes_adults.csv`, `schema.json`, `results/data_flow.csv`, `missingness.csv` |

## Training stage

| Module | Purpose | Key functions |
|---|---|---|
| `splits.py` | Protocol splits | `temporal_split` (A), `in_period_split` (A2), `random_split` (B), `check_no_overlap`, `describe_splits` |
| `models.py` | Shared preprocessing and the 5 models | `make_preprocessor` (ColumnTransformer fitted on train only), `make_estimator` (imbalance handling), `expand_grid`, `train_mlp` (explicit epoch loop with early stopping), `fit_model`, `predict_proba` |
| `evaluation.py` | Metrics and uncertainty | `compute_metrics` (NaN + warning when undefined), `best_f1_threshold` (validation only), `metric_ci`, `paired_difference`, `unpaired_difference`, `ci_table`, `paired_comparisons`, `gap_table` |
| `experiment.py` | One protocol end to end | `tune_model` (grid, validation PR-AUC), `evaluate_model` (all splits, two thresholds), `run_protocol`, `cross_validate`, `ablation_variants`, `run_ablation` |
| `benchmark.py` | Runs A, A2, B and saves everything | `make_all_splits`, `run_all_protocols`, `save_protocol_results`, `summarize_random_seeds`, `select_best_model`, `save_selected_models`, `run_cv_all`, `run_uncertainty`, `write_run_metadata` |

Flow of `--stage train` (`run.py: stage_train`): load the processed table → `run_all_protocols`
(tuning + evaluation for each protocol/seed) → save `metrics.csv`, `tuning.csv`, predictions,
`mlp_history.csv`, confusion matrices → pick the best protocol-A model on validation → ablation →
cross-validation → bootstrap tables.

## Analysis and outputs

| Module | Purpose | Key functions | Outputs |
|---|---|---|---|
| `drift.py` | Describe change between cycles | `cycle_summary` (incl. weighted prevalence), `numeric_drift` (KS, PSI), `categorical_drift`, `domain_classifier_auc`, `run_drift` | `cycle_summary.csv`, `drift_numeric.csv`, `drift_categorical.csv`, `domain_classifier.csv` |
| `plotting.py` | 9 figures from saved CSVs | `apply_style` (paper/slide), one `plot_*` per figure, `save_figure`, `make_all_figures` | `figures/fig01–fig09` (PNG 300 DPI + PDF), `captions.md` |
| `summaries.py` | Traceable headline numbers | `lookup` (file + column + filter), `key_numbers`, `key_findings`, `interpretation` (CI-based wording), `model_comparison_table`, `cv_summary_table` | used by notebook, report and slides |
| `report_tables.py` | LaTeX/Markdown tables | `write_report_tables`, `write_number_macros` (`\kn{id}`), `write_literature_tables`, `write_model_comparison_md`, `write_captions` | `report/tables/*.tex`, `results/model_comparison.md` |
| `diagrams.py` | Box-and-arrow diagrams as data | `draw_diagram`, `describe_diagram`, `make_all_diagrams` | methodology, architecture, protocols, features diagrams |
| `presentation_pack.py`, `slides.py`, `presentation_docs.py` | Build `for_presentation/` | `build_pack`, `write_slides_md`, `write_style_guide`, `render_sample_slides` | slides.md, tables, charts data, figures, style guide |
| `build_notebook.py` | Standalone notebook | `object_source` (inspect/ast), `imports_cell`, `setup_cell`, `config_cell`, `build_notebook` | `notebooks/CSCI447P034.ipynb` |
| `run.py` | Command line | `load_config`, one `stage_*` per stage, `main` | — |

## Why some things are done this way
* **Preprocessing inside the pipeline:** medians and category frequencies are learned from the
  training split only, so validation/test data never influence training (no leakage).
* **Thresholds from validation:** choosing the threshold on test data would inflate test scores.
* **Paired bootstrap:** two models scored on the same resamples isolate the model difference.
* **Protocol A2:** the same fitted model on two test sets separates "time" from "different model".
* **Key numbers with sources:** every number in the notebook, report and slides can be traced to
  a file and re-checked by the tests.

## Tests (`tests/`)
Synthetic NHANES-like XPT files are written by `tests/xpt_writer.py` (tests only). Tests cover XPT
reading, target mapping, special codes, schema consistency, split overlap and order, forbidden
features, preprocessing fitted on training data, the MLP loop, undefined metrics, an end-to-end
`--quick` run, notebook static checks and presentation-pack consistency.

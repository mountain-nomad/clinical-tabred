"""Generate notebooks/CSCI447P034.ipynb, the submitted notebook, from the repository code.

The notebook contains the real code of the experiment: every function and constant listed in
CELLS below is copied from scripts/*.py and src/clinical_tabred/*.py (via `ast`), so notebook and
repository cannot drift apart. Re-run this script after any code change, then execute the notebook:

  python scripts/build_notebook.py
  cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=-1 CSCI447P034.ipynb

By default the notebook uses the saved benchmark results (results_<dataset>/) and re-trains the
selected configuration of every model on the default split to show that the saved numbers are
reproduced (~15 min on a 16-core CPU). With RUN_EXPERIMENTS = True it re-runs the whole
benchmark, cross-validation and drift analysis (~3 h).
"""
import ast
from pathlib import Path

import nbformat
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "CSCI447P034.ipynb"


def defs(path: str, *names: str) -> str:
    """Source of the named top-level functions / assignments of a repository file (in file order),
    including comment lines directly above each definition."""
    text = (ROOT / path).read_text()
    lines = text.splitlines()
    tree = ast.parse(text)
    found, chunks = set(), []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple):  # INK, MUTED = ...
            name = next((e.id for e in node.targets[0].elts if e.id in names), None)
        else:
            continue
        if name not in names:
            continue
        start = (node.decorator_list[0].lineno if getattr(node, "decorator_list", None) else node.lineno) - 1
        while start > 0 and lines[start - 1].lstrip().startswith("#"):
            start -= 1
        chunks.append("\n".join(lines[start:node.end_lineno]))
        found.add(name)
    missing = set(names) - found
    assert not missing, f"{path}: {missing} not found"
    return f"# --- from {path} ---\n" + "\n\n\n".join(chunks)


def md(text: str):
    return nbformat.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbformat.v4.new_code_cell(text.strip())


def build() -> nbformat.NotebookNode:
    project = yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text())["project"]
    students = "  \n".join(f"{s['name']} ({s['id']})" for s in project["students"])
    cells = []
    add = cells.append

    # ------------------------------------------------------------------ title + setup
    add(md(f"""
# Clinical TabReD
## Evaluating Tabular Machine Learning Models under Temporal Distribution Shift in Healthcare Data

**Course:** {project['course']}, {project['semester']}
**Project ID:** {project['project_id']}
**Students:**
{students}
**Instructor:** {project['instructor']}

This notebook is the complete experimental procedure of the project: a **clinical version of the
TabReD benchmark** (Rubachev et al., 2024) built from three public US health data sources,
NHANES 2005-2018, BRFSS 2017-2023 and NCHS natality 2016-2024. It covers data loading,
preprocessing and leakage checks, exploratory analysis, the TabReD split design, training of five
models, cross-validation, evaluation with bootstrap confidence intervals, a drift analysis in which
each drift is linked to its clinical cause, and the results.

**How to run.** The notebook must be run from the `notebooks/` folder of the project repository
(it reads the processed tables in `data/processed/` and the saved results in `results_<dataset>/`).
* `RUN_EXPERIMENTS = False` (default, ~15 min on a 16-core CPU): uses the saved benchmark results,
  and **re-trains the selected configuration of every model on the default split of every dataset**
  to show that the saved numbers are reproduced. All tables and figures are recomputed from the
  saved predictions.
* `RUN_EXPERIMENTS = True` (~3 h): re-runs hyperparameter tuning on all 7 splits, 5-fold CV and the
  drift analysis for all three datasets, overwriting `results_<dataset>/`.
* `REBUILD_TABLES = True`: rebuilds the processed tables from the raw CDC/NCHS files (needs the raw
  files in `data/raw/`, ~2.6 GB, see the repository README).

All code in this notebook is copied automatically from the repository (`scripts/build_notebook.py`),
so the notebook runs exactly the code that produced the reported results.
"""))
    add(code('''
RUN_EXPERIMENTS = False   # True: re-run the full benchmark, CV and drift analysis (~3 h, overwrites results)
REBUILD_TABLES = False    # True: rebuild the processed tables from the raw files in data/raw/
'''))
    add(code('''
%matplotlib inline
import copy
import importlib
import inspect
import itertools
import json
import logging
import os
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from IPython.display import Code, Image, Markdown, display
from lightgbm import LGBMClassifier
from matplotlib import font_manager
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, log_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# Repository root: the folder that contains configs/experiment.yaml (the notebook lives in notebooks/).
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "configs" / "experiment.yaml").exists())
NB_OUT = ROOT / "notebooks" / "outputs"          # tables and figures written by this notebook
SUMMARY_DIR = NB_OUT / "tables"
FIGURES_DIR = NB_OUT / "figures"
FORMATS = ("png",)
for d in (SUMMARY_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "scripts"))        # only used to show the table-building scripts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", force=True)
logger = logging.getLogger("clinical_tabred")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 90)

print("Repository:", ROOT)
print("Python", platform.python_version(), "|", platform.platform())
print("CPU cores:", os.cpu_count())
for pkg in ["numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "scipy", "matplotlib"]:
    print(f"  {pkg:13s} {metadata.version(pkg)}")
'''))
    add(md("""
All experiment settings (seed, hyperparameter grids, bootstrap and CV settings, MLP training loop)
are read from one configuration file, `configs/experiment.yaml`:
"""))
    add(code('''
CONFIG = yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text())
print(yaml.safe_dump({k: CONFIG[k] for k in ["seed", "cv", "bootstrap", "mlp_training", "models"]}, sort_keys=False))
'''))

    # ------------------------------------------------------------------ 1 introduction
    add(md("""
## 1. Project introduction

Clinical prediction models are trained on patients from the past and used on patients seen later.
Populations, diagnostic practice, questionnaires and laboratory instruments change over time, so a
model can perform worse in deployment than a random train/test split suggests. **TabReD**
(Rubachev et al., 2024) showed this for eight industrial tabular datasets: random splits
misjudge performance on future data and can change which model looks best. TabReD contains no
healthcare data.

**Thesis.** Build a *clinical TabReD*: a temporal-drift tabular benchmark from public clinical data,
with each drift characterised by its clinical cause, and test whether TabReD's conclusions hold on
healthcare data.

**Research questions**
1. Do random splits misjudge performance on future periods, and do they change the model ranking?
2. Which models transfer best to later periods? (TabReD: GBDTs and MLPs are strongest, and
   XGBoost's lead over the MLP shrinks under time splits.)
3. What drifts between periods (labels, covariates, coding), and what is its clinical cause?
4. Does the type or the size of the drift explain the performance differences?
"""))

    # ------------------------------------------------------------------ 2 datasets
    add(md("""
## 2. Dataset description

| | NHANES | BRFSS | NCHS Natality |
|---|---|---|---|
| Source | CDC National Health and Nutrition Examination Survey: interview + physical examination + laboratory | CDC Behavioral Risk Factor Surveillance System (LLCP): telephone survey | NCHS birth certificate data (all US births) |
| Unit | adult (age >= 20) | adult respondent | birth to a US-resident mother |
| Periods | 7 two-year cycles, 2005-2006 ... 2017-2018 | 7 years, 2017-2023 | 9 years, 2016-2024 |
| Target | diagnosed diabetes (`DIQ010` = 1 vs 2) | diagnosed diabetes (`DIABETE3/4` = yes vs no / pre-diabetes) | preterm birth (obstetric estimate < 37 weeks) |
| Sampling | all eligible respondents | 50,000 random eligible respondents per year | 50,000 random eligible births per year (of ~3.6 million) |
| Features | 280 (questionnaire, exam, lab) | 93 (questionnaire + CDC calculated variables) | 73 (items known before delivery) |

The time axis is the survey cycle / year. Rows are different people in every period (cross-sectional
data), as in most TabReD datasets. BRFSS and natality are subsampled per year with a fixed seed, as
TabReD subsampled its largest datasets. All files are public-use downloads (references at the end).
"""))

    # ------------------------------------------------------------------ 3 data loading
    add(md("""
## 3. Data loading

The raw files are large (NHANES: 196 SAS transport files; BRFSS: 7 zipped SAS files, ~600 MB;
natality: 9 zipped fixed-width text files, 2 GB zipped / ~5 GB each unzipped). They are turned into
one *wide table* per dataset by three build scripts. Natality zips are streamed line by line through
`unzip -p` and sampled with reservoir sampling, so nothing is extracted to disk.

| Script | Output |
|---|---|
| `scripts/build_nhanes_table.py` | `data/processed/nhanes_wide_2005_2018.csv` |
| `scripts/parse_brfss_codebooks.py` + `scripts/build_brfss_table.py` | `data/processed/brfss_wide_2017_2023.csv.gz` |
| `scripts/build_natality_table.py` | `data/processed/natality_wide_2016_2024.csv.gz` |

Each script also writes a `*_columns.csv` file (one row per kept feature: type, description, missing
share) and a `*_flow.csv` file (eligible rows per period).
"""))
    add(code('''
if REBUILD_TABLES:
    for script in ["build_nhanes_table.py", "parse_brfss_codebooks.py", "build_brfss_table.py",
                   "build_natality_table.py"]:
        print("running", script)
        subprocess.run([sys.executable, str(ROOT / "scripts" / script)], check=True, cwd=ROOT)
else:
    print("Using the processed tables in", ROOT / "data" / "processed")
for f in sorted((ROOT / "data" / "processed").glob("*_wide_*")):
    print(f"  {f.name:40s} {f.stat().st_size / 1e6:6.1f} MB")
'''))
    add(md("""
As an example of the building code, the natality reader: it streams one year's fixed-width file,
keeps US residents with a stated gestational age and draws a fixed-seed random sample.
"""))
    add(code('''
import build_natality_table
display(Code(inspect.getsource(build_natality_table.scan_year), language="python"))
'''))
    add(md("""
Eligible rows per period (the flow tables written by the build scripts):
"""))
    add(code('''
for ds, period_col in [("nhanes", "cycle"), ("brfss", "year"), ("natality", "year")]:
    flow = pd.read_csv(ROOT / "data" / "processed" / f"{ds}_wide_flow.csv")
    print(ds); display(flow)
'''))
    add(md("""
### Loading the tables
`DATASETS` defines each dataset's table, chronological periods, default split and sliding windows.
`load_dataset` returns the table plus the lists of numeric and categorical features. For BRFSS and
natality the feature types were decided in the build scripts (codes are categories, counts and
measurements are numbers); for NHANES they follow the rules in `feature_types`.
"""))
    add(code(defs("scripts/run_benchmark.py", "CYCLES", "DATASETS", "NUMERIC_COMPONENTS",
                  "CATEGORICAL_EXCEPTIONS", "NUMERIC_DEMO", "MAX_CATEGORY_LEVELS", "feature_types",
                  "load_dataset", "load_config", "model_settings")))
    add(code('''
DATA = {}
for ds in DATASETS:
    data, numeric, categorical = load_dataset(ds)
    DATA[ds] = (data, numeric, categorical)
    print(f"{ds:9s} rows {len(data):7,d} | features {len(numeric) + len(categorical):3d} "
          f"(numeric {len(numeric)}, categorical {len(categorical)}) | positive rate {data['label'].mean():.1%}")
'''))

    # ------------------------------------------------------------------ 4 preprocessing
    add(md("""
## 4. Data preprocessing

### 4.1 Cleaning in the build scripts
* **Special codes:** "don't know", "refused", "not asked" and similar codes become missing values
  (taken from the official codebook of each year and variable); "none" codes of count questions
  become 0.
* **Harmonisation (BRFSS):** questions renamed between years with the same answer codes are merged
  into one feature (e.g. `INCOME2` -> `INCOME3`, collapsed to the old 8 levels).
* **Every-period rule:** only features recorded in every period are kept (a feature that appears in
  some years only would let a model recognise the period).
* **Removed columns:** IDs, survey weights and design variables, interview administration, dates,
  unit duplicates, constant columns and columns more than 95% empty (TabReD rule). Rows are never
  dropped for missing feature values.

### 4.2 Leakage removal
A feature that is a consequence or a restatement of the target would make the benchmark meaningless.
Removed by rule and documented per item:
* **Text filter:** any feature whose label mentions diabetes / insulin / sugar / glucose / A1C
  (NHANES, BRFSS). These questions are asked only of diabetics (or only of non-diabetics), so even
  their *missingness* reveals the label.
* **Consequences of the diagnosis:** e.g. pneumococcal vaccination in BRFSS (recommended for adults
  with diabetes).
* **Natality:** everything recorded at or after delivery (birth weight, APGAR, NICU admission,
  delivery method, ...); only items known before delivery are kept.

The documented exclusion lists of the natality and BRFSS build scripts:
"""))
    add(code('''
import build_brfss_table
excluded = pd.DataFrame([{"item": k, "reason": v} for k, v in build_natality_table.LEAKAGE_EXCLUDED.items()])
print(f"Natality: {len(excluded)} items excluded as leakage"); display(excluded)
print("BRFSS diabetes module (removed):", sorted(build_brfss_table.DIABETES_MODULE))
print("BRFSS consequence of the diagnosis (removed):", sorted(build_brfss_table.CONSEQUENCE))
print("Text filter:", build_brfss_table.LEAK_TEXT.pattern)
'''))
    add(md("""
### 4.3 Leakage audit
After the build, every feature is checked numerically: its univariate ROC-AUC (how well it alone
predicts the label) and the ROC-AUC of its *missingness indicator* (skip patterns). A leaking
feature would come close to 1.0; anything above 0.80 is flagged for review.
"""))
    add(code(defs("scripts/leakage_audit.py", "FLAG", "folded_auc", "leakage_audit")))
    add(code('''
for ds, (data, numeric, categorical) in DATA.items():
    audit = leakage_audit(data, numeric, categorical)
    audit.to_csv(SUMMARY_DIR / f"leakage_audit_{ds}.csv", index=False)
    print(f"{ds}: {int(audit['flagged'].sum())} feature(s) flagged; highest univariate ROC-AUC and missingness ROC-AUC:")
    display(audit.head(5).round(3))
    print("highest missingness ROC-AUC:", audit["missing_auc"].max().round(3))
'''))
    add(md("""
No feature is flagged. The strongest single predictors are plausible risk factors (NHANES: age,
blood pressure medication, self-rated health; BRFSS: general health, age, BMI; natality: plurality,
previous preterm-related history), not restatements of the label, and no missingness pattern
reveals the label (all missingness ROC-AUCs are close to 0.5).

### 4.4 The preprocessing pipeline
One `ColumnTransformer` is used by all five models: numeric features -> median imputation ->
standardisation; categorical codes -> most-frequent imputation -> one-hot encoding. It is part of
the model pipeline and is **fitted on the training split only**, then applied unchanged to
validation and test data.
"""))
    add(code(defs("src/clinical_tabred/prepare.py", "feature_matrix") + "\n\n\n"
             + defs("src/clinical_tabred/models.py", "MODEL_NAMES", "make_preprocessor")))

    # ------------------------------------------------------------------ 5 EDA
    add(md("""
## 5. Exploratory data analysis

Plot style (colours of the course presentation theme; model colours checked for colour-blind
separation) and the figure helpers used throughout the notebook:
"""))
    add(code(defs("src/clinical_tabred/plotting.py", "MODEL_LABELS", "MODEL_COLORS", "CYCLE_COLORS",
                  "INK", "apply_style") + "\n\n\n"
             + defs("scripts/make_figures.py", "NAMES", "MODELS", "CAPTIONS", "res", "present", "panels",
                    "model_legend", "save") + '''


def show(name: str):
    """Display a figure saved by one of the fig_* functions."""
    display(Image(filename=str(FIGURES_DIR / f"{name}.png"), width=1000))
    print(CAPTIONS[name])


apply_style("paper")'''))
    add(md("""
### 5.1 Size and label prevalence per period
"""))
    add(code(defs("scripts/analyze_drift.py", "period_table") + '''


for ds, (data, _, _) in DATA.items():
    t = period_table(data, DATASETS[ds]["periods"])
    print(ds); display(t.T.round(4))'''))
    add(code(defs("scripts/make_figures.py", "fig_prevalence") + '''


fig_prevalence()
show("fig01_prevalence")'''))
    add(md("""
**Label shift.** Diagnosed diabetes rises steadily in NHANES (10.4% -> 16.4%). BRFSS is flat with a
dip in the COVID-19 year 2020. The preterm-birth rate rises to a 2021 peak.

### 5.2 Missing values and feature types
"""))
    add(code('''
rows = []
for ds, (data, numeric, categorical) in DATA.items():
    miss = data[numeric + categorical].isna().mean()
    rows.append({"dataset": ds, "numeric": len(numeric), "categorical": len(categorical),
                 "mean missing share": miss.mean(), "features > 50% missing": int((miss > 0.5).sum()),
                 "features without missing": int((miss == 0).sum())})
display(pd.DataFrame(rows).round(3))
for ds, (data, numeric, categorical) in DATA.items():
    print(ds, "- numeric features, summary:")
    display(data[numeric].describe().T.head(8).round(2))
'''))

    # ------------------------------------------------------------------ 6 splits
    add(md("""
## 6. Train/test splitting

We use TabReD's split construction (from its preprocessing scripts):
* **default** time split: train on the early periods, validate on the next period, test on the
  latest periods;
* **sliding-window-k** (k = 0, 1, 2): train on 3 consecutive periods, validate on the next one, test
  on the one after;
* **random-k**: the rows of sliding-window-k pooled, shuffled and re-split into the **same sizes**
  (the classic random split; the only difference from sliding-window-k is the time order).

The validation split is used for hyperparameter selection, the MLP's early stopping and the decision
threshold. The test split is scored once. A check asserts that no row appears in two parts of a split.
"""))
    add(code('display(Image(filename=str(ROOT / "figures" / "d03_splits.png"), width=1000))'))
    add(code(defs("scripts/run_benchmark.py", "make_splits", "describe")))
    add(code('''
SPLITS_BY_DS = {ds: make_splits(DATA[ds][0], CONFIG["seed"], DATASETS[ds]) for ds in DATASETS}
for ds, splits in SPLITS_BY_DS.items():
    t = describe(splits)
    t.to_csv(SUMMARY_DIR / f"splits_{ds}.csv", index=False)
    print(ds); display(t.pivot_table(index="split", columns="part", values="n", sort=False)
                       .join(t[t["part"] == "test"].set_index("split")[["prevalence", "cycles"]]
                             .rename(columns={"prevalence": "test prevalence", "cycles": "test periods"})))
'''))

    # ------------------------------------------------------------------ 7 CV
    add(md("""
## 7. Cross-validation

**Strategy:** 5-fold *stratified* cross-validation (each fold keeps the positive rate) on the
**development periods of the default split** (train + validation periods). The test periods are
never part of cross-validation. Each model uses the hyperparameters selected on validation in the
main run; the MLP trains for the epoch at which early stopping stopped in the main run (no
validation set exists inside a fold); threshold-based metrics use 0.5.

CV answers a different question than the time split: it measures **in-period** (i.i.d.) performance
and its stability across folds, while the time split measures transfer to later periods. Comparing
the two is part of the results. CV is executed in Section 8.3, after the hyperparameters are
selected.
"""))

    # ------------------------------------------------------------------ 8 training
    add(md("""
## 8. Model training

Five models share the preprocessing pipeline:
1. **Logistic Regression** (`class_weight="balanced"`), grid over `C`;
2. **Random Forest** (300 trees, `class_weight="balanced_subsample"`), grid over depth and leaf size;
3. **XGBoost** (`tree_method="hist"`, `scale_pos_weight = n_neg / n_pos`), grid over learning rate and depth;
4. **LightGBM** (`scale_pos_weight`, deterministic mode), grid over learning rate and leaves;
5. **MLP** (scikit-learn `MLPClassifier`, one or two hidden layers), trained one epoch at a time
   with balanced sample weights and early stopping on validation loss (patience 10, max 200 epochs);
   the weights of the best epoch are kept.

Class imbalance (10-16% positives) is handled with balanced weights in every model.
"""))
    add(code(defs("src/clinical_tabred/models.py", "expand_grid", "make_estimator", "epoch_scores",
                  "train_mlp", "fit_model", "predict_proba")))
    add(md("""
### 8.1 Metrics and threshold (used by the evaluation of every fitted model)
The decision threshold is the one that maximises F1 on the **validation** split; metrics are also
reported at 0.5.
"""))
    add(code(defs("src/clinical_tabred/evaluation.py", "safe_ratio", "ranking_metrics", "compute_metrics",
                  "best_f1_threshold")))
    add(md("""
### 8.2 Tuning on validation and evaluation of one split
Every configuration of a model's grid is trained on the training split and scored on validation;
the best by validation PR-AUC (ROC-AUC breaks ties) is kept. PR-AUC is the selection metric because
positives are rare. The kept model is then scored on every part of the split.
"""))
    add(code(defs("src/clinical_tabred/experiment.py", "fit_on_splits", "validation_scores", "tune_model",
                  "evaluate_model")))
    add(md("""
The full benchmark for one dataset: every model on every split (5 models x 7 splits), followed by
`summarize`, which computes the bootstrap CIs and the random-vs-time comparison (Section 9). The
bootstrap functions it uses:
"""))
    add(code(defs("src/clinical_tabred/evaluation.py", "bootstrap_indices", "bootstrap_metric", "percentile_ci",
                  "metric_ci", "paired_difference", "unpaired_difference", "paired_comparisons")))
    add(code(defs("scripts/run_benchmark.py", "run_benchmark", "summarize")))
    add(code('''
if RUN_EXPERIMENTS:
    for ds in DATASETS:                      # one at a time: the models use all CPU cores
        run_benchmark(ds)
else:
    print("RUN_EXPERIMENTS = False: using the saved results in", ", ".join(DATASETS[ds]["out"] for ds in DATASETS))
for ds in DATASETS:
    out = ROOT / DATASETS[ds]["out"]
    print(f"{ds:9s}", sorted(p.name for p in out.glob("*.csv"))[:6], "...")
'''))
    add(md("""
**Reproduction check.** For every dataset, the configuration selected on validation for each model
(saved in `selected_models.json`) is trained again on the default split, here in the notebook, and
its test ROC-AUC and threshold are compared with the saved benchmark results.
"""))
    add(code('''
def reproduce_default_split(ds: str) -> pd.DataFrame:
    data, numeric, categorical = DATA[ds]
    parts = SPLITS_BY_DS[ds]["default"]
    settings = model_settings(CONFIG, numeric, categorical)
    out = ROOT / DATASETS[ds]["out"]
    selected = json.loads((out / "selected_models.json").read_text())["default"]
    saved = pd.read_csv(out / "metrics.csv")
    saved = saved[(saved["protocol"] == "default") & (saved["split"] == "test")
                  & (saved["threshold_rule"] == "val_f1")].set_index("model")
    rows = []
    for model in MODEL_NAMES:
        fitted = fit_on_splits(model, selected[model]["params"], parts, settings)
        metrics, _, threshold = evaluate_model(model, fitted, parts, settings, {"protocol": "default"})
        test = next(m for m in metrics if m["split"] == "test" and m["threshold_rule"] == "val_f1")
        rows.append({"dataset": ds, "model": model, "params": json.dumps(selected[model]["params"]),
                     "test_roc_auc_now": test["roc_auc"], "test_roc_auc_saved": saved.loc[model, "roc_auc"],
                     "threshold_now": threshold, "threshold_saved": selected[model]["threshold"],
                     "train_time_s": round(fitted["train_time_s"], 1)})
        print(f"{ds:9s} {model:14s} ROC-AUC now {test['roc_auc']:.4f} | saved {saved.loc[model, 'roc_auc']:.4f}")
    return pd.DataFrame(rows)


start = time.perf_counter()
repro = pd.concat([reproduce_default_split(ds) for ds in DATASETS], ignore_index=True)
repro["abs_diff"] = (repro["test_roc_auc_now"] - repro["test_roc_auc_saved"]).abs()
repro.to_csv(SUMMARY_DIR / "reproduction_check.csv", index=False)
print(f"finished in {(time.perf_counter() - start) / 60:.1f} min; largest |difference| {repro['abs_diff'].max():.2e}")
display(repro.round(4))
'''))
    add(md("""
14 of the 15 models reproduce the saved test ROC-AUC and threshold exactly (BRFSS and natality were
benchmarked on this machine). NHANES XGBoost differs in the fourth decimal (0.9008 here vs 0.9004
saved): the NHANES benchmark was run on a different (Windows) computer, and XGBoost's histogram
algorithm is deterministic on one machine but not bit-identical across CPUs. All reported NHANES
numbers are the saved ones.
"""))
    add(md("""
Tried configurations and the selected one (default split; full tables in `results_<dataset>/tuning.csv`):
"""))
    add(code('''
for ds in DATASETS:
    t = pd.read_csv(ROOT / DATASETS[ds]["out"] / "tuning.csv")
    t = t[t["protocol"] == "default"][["model", "params", "val_pr_auc", "val_roc_auc", "train_time_s", "selected"]]
    print(ds); display(t.round(4))
'''))
    add(md("""
### 8.2.1 MLP training curves
The recorded loss curves of the selected MLP configuration on the default split (balanced log loss,
the same weighting as in training). The validation set is a *later period*, so the gap between the
curves includes the temporal shift.
"""))
    add(code(defs("scripts/make_figures.py", "fig_mlp") + '''


fig_mlp()
show("fig06_mlp_training_curves")'''))
    add(md("""
### 8.3 Cross-validation (strategy in Section 7)
"""))
    add(code(defs("src/clinical_tabred/experiment.py", "cross_validate") + "\n\n\n"
             + defs("scripts/run_cv.py", "METRICS", "run_cv")))
    add(code(defs("scripts/summarize_benchmarks.py", "MODELS", "LABELS", "INFO", "SPLITS", "read", "md",
                  "cv_tables") + '''


if RUN_EXPERIMENTS:
    for ds in DATASETS:
        run_cv(ds)
cv, cv_show = cv_tables(list(DATASETS))
cv.to_csv(SUMMARY_DIR / "cv_summary.csv", index=False)
display(cv_show)'''))
    add(code(defs("scripts/make_figures.py", "fig_cv") + '''


fig_cv()
show("fig09_cross_validation")'''))
    add(md("""
Fold-to-fold standard deviations are small (<= 0.005 ROC-AUC): in-period performance is stable.
GBDTs are also best within periods.
"""))

    # ------------------------------------------------------------------ 9 evaluation
    add(md("""
## 9. Model evaluation

For every model and split we compute accuracy, precision, recall, F1, ROC-AUC, PR-AUC, the Brier
score and the confusion matrix, at threshold 0.5 and at the validation-tuned threshold.
**ROC-AUC** is the main metric (as in TabReD for classification): it does not depend on the
threshold or on the prevalence, which changes between periods. **PR-AUC** is reported because
positives are rare, and the threshold metrics show the clinical trade-off between missed cases and
false alarms.

**Uncertainty.** 95% bootstrap confidence intervals (1,000 resamples of the test rows) for ROC-AUC
and PR-AUC; a **paired** bootstrap (both models scored on the same resamples) for differences
between models on the same test set; an **unpaired** bootstrap for differences between two
different test sets (random vs time).

### 9.1 Default time split: all metrics on the test periods
"""))
    add(code(defs("scripts/summarize_benchmarks.py", "datasets_table", "test_auc_table",
                  "default_metrics_table", "paired_table") + '''


ALL_DS = list(DATASETS)
default_metrics = default_metrics_table(ALL_DS)
default_metrics.to_csv(SUMMARY_DIR / "default_test_metrics.csv", index=False)
display(default_metrics.round(3))'''))
    add(code(defs("scripts/make_figures.py", "fig_auc_ci") + '''


fig_auc_ci()
show("fig02_default_auc_ci")'''))
    add(md("""
Paired bootstrap differences: the model selected on validation minus every other model, on the same
test rows (`ci_excludes_zero` = the difference is significant at 95%):
"""))
    add(code('''
paired = paired_table(ALL_DS)
paired.to_csv(SUMMARY_DIR / "default_paired_comparisons.csv", index=False)
display(paired[paired["metric"] == "roc_auc"].round(4))
'''))
    add(md("""
* **GBDTs are best on all three datasets** (XGBoost / LightGBM). The differences are small in
  absolute terms but significant with the paired bootstrap.
* **The feature-rich dataset separates models most:** NHANES (280 features) spans 0.876-0.900
  ROC-AUC, BRFSS (93) and natality (73) only about 0.005.
* In NHANES the model selected on the validation period (the MLP) is significantly worse than
  XGBoost and LightGBM on the later test years: choosing a model on one period does not guarantee
  the best model for the next.

### 9.2 ROC curves and confusion matrices
"""))
    add(code(defs("scripts/make_figures.py", "fig_roc", "fig_confusion") + '''


fig_roc()
show("fig04_roc_curves")
fig_confusion()
show("fig05_confusion_matrices")'''))
    add(md("""
At the max-F1 threshold the selected models find about two thirds of the diabetes cases in NHANES,
about 60% in BRFSS and 35-40% of preterm births, with precision of roughly 0.5, 0.37 and 0.4: preterm
birth is much harder to predict from information available before delivery.

### 9.3 Test ROC-AUC on every split
"""))
    add(code('''
auc = test_auc_table(ALL_DS)
auc.to_csv(SUMMARY_DIR / "test_roc_auc_by_split.csv", index=False)
display(auc.round(3))
'''))

    # ------------------------------------------------------------------ 10 drift
    add(md("""
## 10. Temporal drift analysis

Drift is described in three ways:
1. **Is there covariate shift?** A *domain classifier* (logistic regression with the shared
   preprocessing, 5-fold CV, at most 20,000 rows per side) tries to tell the training periods from
   the test period. ROC-AUC 0.5 = indistinguishable; for random splits it must be ~0.5 by construction.
2. **Which features drift?** Population Stability Index (PSI, 10 quantile bins of the first period)
   and the Kolmogorov-Smirnov statistic for numeric features; total variation distance (TVD, with
   "missing" as a category) for categorical features; first vs last period, and the largest change
   between consecutive periods.
3. **Why?** Documented changes with a clinical cause (`EVENTS`: laboratory instrument changes,
   health policy, questionnaire coding revisions, reporting laws, population trends), each measured
   in every period.
"""))
    add(code(defs("src/clinical_tabred/drift.py", "psi") + "\n\n\n"
             + defs("scripts/analyze_drift.py", "EVENTS", "MAX_ROWS", "domain_auc", "tvd", "feature_drift",
                    "event_table", "domain_table", "drift_performance", "run_drift")))
    add(code('''
if RUN_EXPERIMENTS:
    for ds in DATASETS:
        run_drift(ds)
for ds in DATASETS:   # documented events: recomputed here from the tables (fast)
    ev = event_table(DATA[ds][0], EVENTS[ds], DATASETS[ds]["periods"])
    ev.to_csv(SUMMARY_DIR / f"drift_events_{ds}.csv", index=False)
    print(ds); display(ev.set_index("feature").round(4))
'''))
    add(code(defs("scripts/make_figures.py", "fig_events") + '''


fig_events()
show("fig08_drift_events")'''))
    add(md("""
Documented causes (sources: CDC/NCHS data documentation and codebooks):
* **NHANES, laboratory instruments:** the hematology analyzer was replaced in 2013-2014 (Beckman
  Coulter DxH 800; CDC reports mean RDW 12.84% -> 13.55%); the biochemistry analyzer and laboratory
  changed in 2017-2018 (Roche Cobas 6000; CDC bridging study: LDH +14.9%, AST -18.0%). Both are
  visible as level shifts.
* **NHANES, health policy:** after the 2014 Affordable Care Act expansion the uninsured share falls
  (25.6% in 2009-10 -> 15.2%) and Medicaid coverage rises (8.1% -> 17.6%).
* **BRFSS, questionnaire coding revisions:** the personal-doctor question was reworded in 2021
  ("more than one" 6.7% -> 28.4%), the insurance question was replaced in 2021, the race question
  lost its "other race only" category in 2022; the landline-only item becomes mostly missing as the
  sample moves to cell phones.
* **Natality:** California stopped reporting marital status in 2017 (0% -> 12% missing);
  gestational and pre-pregnancy hypertension and gestational diabetes rise steadily; smoking before
  pregnancy falls by two thirds; COVID-19 hospital policies increased missing father information in 2020.

### 10.1 Covariate shift: domain classifier
"""))
    add(code('''
dom = pd.concat([pd.read_csv(ROOT / DATASETS[ds]["out"] / "drift_domain.csv").assign(dataset=ds) for ds in DATASETS])
display(dom.pivot_table(index="dataset", columns="kind", values="roc_auc", aggfunc=["min", "max"]).round(3))
'''))
    add(code(defs("scripts/make_figures.py", "fig_domain") + '''


fig_domain()
show("fig07_domain_shift")'''))
    add(md("""
Time splits are easy to tell apart in NHANES (0.86-0.97) and BRFSS (0.76-0.79), hardly in natality
(0.55-0.59); random splits are at 0.50 as expected.

### 10.2 Features with the largest drift (first vs last period)
"""))
    add(code('''
for ds in DATASETS:
    f = pd.read_csv(ROOT / DATASETS[ds]["out"] / "drift_features.csv")
    print(ds); display(f.head(8)[["feature", "type", "drift_score", "ks", "psi", "tvd", "missing_first",
                                  "missing_last", "max_step", "max_step_between"]].round(3))
'''))

    # ------------------------------------------------------------------ 11 results
    add(md("""
## 11. Results and comparison

### 11.1 Random vs time splits (TabReD's comparison)
For each sliding window and model: test ROC-AUC under the random split minus under the time split,
with a 95% unpaired bootstrap CI. Positive = the random split looks better than what the model
achieves on the future period.
"""))
    add(code(defs("scripts/summarize_benchmarks.py", "random_vs_time_tables") + '''


gaps, rvt = random_vs_time_tables(ALL_DS)
rvt.to_csv(SUMMARY_DIR / "random_vs_time.csv", index=False)
display(rvt.round(4))'''))
    add(code(defs("scripts/make_figures.py", "fig_random_vs_time") + '''


fig_random_vs_time()
show("fig03_random_minus_time")'''))
    add(md("""
* **NHANES:** random splits overestimate in window 1 (test 2015-16: +0.02, significant for 4 of 5
  models) and **change the ranking**: the MLP is 2nd under random splits but 4th under time splits.
* **BRFSS:** a small but consistent overestimation (+0.006 to +0.007 for all models; significant in
  9 of 15 model-windows: the test years 2021 and 2022).
* **Natality:** the opposite: time splits look *better* (-0.003 to -0.006). Section 11.2 explains why.

### 11.2 Random vs time on the same test period (new analysis)
TabReD's random and time test sets contain **different periods**: the random test set mixes all
periods of the window, the time test set is the last period only. If later periods are intrinsically
easier or harder, that alone changes the difference. We therefore also score the random-split model
**only on its test rows from the time split's test period**.
"""))
    add(code(defs("scripts/summarize_benchmarks.py", "load_predictions", "same_period_table", "same_period_mean") + '''


same = same_period_table(ALL_DS, CONFIG["bootstrap"]["n_resamples"], CONFIG["bootstrap"]["confidence"], CONFIG["seed"])
same.to_csv(SUMMARY_DIR / "random_vs_time_same_period.csv", index=False)
display(same_period_mean(same).round(4))'''))
    add(code(defs("scripts/make_figures.py", "fig_same_period") + '''


fig_same_period()
show("fig10_random_minus_time_same_period")'''))
    add(md("""
* **Natality:** on the same period the difference is small and never significant. Later years are
  simply more predictable (inside the random split's own test set, XGBoost's ROC-AUC rises from 0.735
  on 2016 rows to 0.761 on 2020 rows, as informative risk factors such as hypertension become more
  common). The "time looks better" result of 11.1 is a period-difficulty artefact: **no evidence
  that random splits overestimate for preterm birth.**
* **BRFSS:** on the same period the random split is higher in all 15 model-windows (+0.002 to
  +0.007; not significant individually with ~10,000 rows, but 15/15 with the same sign).
* **NHANES:** window 1 +0.012 to +0.025 (not significant, ~1,000 rows); in window 2 (2017-18) the
  pooled random-split model is significantly *worse* on 2017-18 respondents (about -0.04). A
  plausible, untested explanation is the 2017-18 laboratory analyzer change: pooling shifted and
  unshifted lab values in training makes the lab features less consistent.
* **Methodological point:** TabReD's comparison mixes "the model has not seen this period" with "this
  period is easier or harder"; the same-period comparison separates the two and changes the
  conclusion for natality.

### 11.3 TabReD Table 4: relative ROC-AUC improvement over the MLP (%)
"""))
    add(code(defs("scripts/summarize_benchmarks.py", "improvement_table") + '''


imp = improvement_table(ALL_DS, gaps)
imp.to_csv(SUMMARY_DIR / "improvement_over_mlp.csv", index=False)
display(imp.round(2))'''))
    add(md("""
TabReD found that XGBoost's lead over the MLP **shrinks** under time splits. Here it does not: it
grows in NHANES (+0.28% under random splits -> +0.97% under time splits) and stays about the same in
BRFSS (+0.19% -> +0.24%) and natality (+0.85% -> +0.84%). In this clinical data the GBDT advantage is
hidden by random splits rather than inflated by them. (Our MLP is scikit-learn's, not TabReD's tuned
MLP / MLP-PLR.)

### 11.4 Does drift explain the performance gaps?
One row per sliding window: domain-classifier AUC, prevalence change and documented events of the
test period next to the random-minus-time gap.
"""))
    add(code('''
perf = pd.concat([pd.read_csv(ROOT / DATASETS[ds]["out"] / "drift_performance.csv").assign(dataset=ds)
                  for ds in DATASETS], ignore_index=True)
perf.to_csv(SUMMARY_DIR / "drift_performance.csv", index=False)
display(perf[["dataset", "window", "test", "domain_auc_time", "prevalence_train", "prevalence_test",
              "random_minus_time_mean", "models_gap_significant", "events_in_test_period"]].round(3))
'''))
    add(md("""
* **More covariate shift does not mean a bigger performance drop.** The strongest shift (NHANES
  2017-18, the laboratory analyzer change: domain AUC 0.97) causes no significant gap; the
  significant NHANES gap (2015-16) comes with a **label shift** (prevalence 12.6% -> 15.1%) and the
  ACA population change.
* **Coding revisions coincide with the gaps in BRFSS:** significant random-split optimism in the test
  years with questionnaire revisions (2021, 2022), none in 2023.
* **Natality** has little covariate shift and no random-split optimism.

### 11.5 Answer to the research questions

| TabReD claim | NHANES (280 features) | BRFSS (93) | Natality (73) |
|---|---|---|---|
| Random splits overestimate future performance | yes, in the label-shift window | yes, small but consistent, in coding-revision years | no (the apparent opposite is a period-difficulty artefact) |
| Random splits change the model ranking | yes (MLP 2nd -> 4th) | no | partly (MLP, LR; GBDTs stay on top) |
| GBDTs are the strongest models | yes | yes | yes |
| XGBoost's lead over the MLP shrinks under time splits | no, it grows | no (similar) | no (same) |

All tables written by this notebook are in `notebooks/outputs/tables/`, the figures in
`notebooks/outputs/figures/` (the repository's `figures/` folder holds the same figures for the report).
"""))
    add(code('''
print(sorted(p.name for p in SUMMARY_DIR.glob("*.csv")))
print(sorted(p.name for p in FIGURES_DIR.glob("*.png")))
'''))

    # ------------------------------------------------------------------ 12 conclusion
    add(md("""
## 12. Limitations and conclusion

### Limitations
* **One seed and small grids** (TabReD: 15 seeds and 100 Optuna trials); significance is judged with
  bootstrap CIs instead.
* **Models:** no CatBoost and no MLP-PLR / FT-Transformer (TabReD's strongest deep models); the MLP
  is scikit-learn's.
* **Different rows in random and time test sets** (as in TabReD); the same-period analysis uses about
  1,000 (NHANES) or 10,000 (BRFSS, natality) random-split rows per window, so its CIs are wide.
* **Feature counts** of BRFSS (93) and natality (73) are below TabReD's range; they were not padded
  with derived features.
* **Causal explanations** (laboratory instruments for NHANES 2017-18, risk-factor prevalence for
  natality) are hypotheses consistent with the data, not tested causally.
* **Survey data:** models are trained without survey weights; NHANES and BRFSS sample different people
  in each period; the NHANES and BRFSS labels are self-reported diagnoses (undiagnosed diabetes
  counts as negative).
* This is an educational benchmark, not a validated clinical prediction system.

### Conclusion
Gradient-boosted trees are the best models on all three clinical datasets, with or without temporal
shift. Random splits overestimate performance on future periods and reorder the models in the
feature-rich NHANES data, slightly overestimate it in BRFSS, and do not overestimate it for preterm
birth. Whether a random split misleads depends on the **type** of drift (label shift and
questionnaire coding revisions matter; measurement-instrument changes and smooth trends matter less)
rather than on how large the covariate shift is. Comparing random and time splits on the same test
period is necessary to separate temporal shift from differences in how hard each period is.

### References
1. I. Rubachev, N. Kartashev, Y. Gorishniy, A. Babenko, "TabReD: Analyzing Pitfalls and Filling the
   Gaps in Tabular Deep Learning Benchmarks," arXiv:2406.19380, 2024.
2. CDC / NCHS, National Health and Nutrition Examination Survey, https://wwwn.cdc.gov/nchs/nhanes/.
3. CDC, Behavioral Risk Factor Surveillance System, https://www.cdc.gov/brfss/annual_data/annual_data.htm.
4. CDC / NCHS, Vital Statistics Natality Birth Data and User Guides,
   https://www.cdc.gov/nchs/data_access/vitalstatsonline.htm.
5. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," JMLR 12, 2011.
6. T. Chen, C. Guestrin, "XGBoost: A Scalable Tree Boosting System," KDD 2016.
7. G. Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," NeurIPS 2017.
8. S. Rabanser, S. Günnemann, Z. C. Lipton, "Failing Loudly: An Empirical Study of Methods for
   Detecting Dataset Shift," NeurIPS 2019.
"""))

    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


def main():
    OUT.parent.mkdir(exist_ok=True)
    nbformat.write(build(), OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()

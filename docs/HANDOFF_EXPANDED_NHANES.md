# Handoff: Expanded NHANES benchmark (session of 15 Nov 2026)

Everything needed to continue the project on another machine or in a new chat.
Branch: `expanded-nhanes`. Read this first, then `results_wide/`.

---

## 1. Project framing (what we are actually asked to do)

Course project CSCI447P034. The official topic description says:

> TabReD (industry-style tabular datasets with time-based splits and hundreds of features; under
> realistic temporal drift, GBDTs retain advantages that academic benchmarks conceal). Gap: datasets
> are industrial (finance, e-commerce) with no healthcare. Clinical temporal drift has a different
> character: protocol changes, coding revisions, population shift. Task: **build and release a
> clinical temporal-drift tabular benchmark (the clinical TabReD), with drift characterised by its
> clinical cause.**

Research question we use: *Do TabReD's conclusions still hold when the data comes from healthcare?*

TabReD key facts (verified in the paper PDF and the official repo `yandex-research/tabred`):
- 8 datasets, median 261 features (98 to 1,026); time-based train/val/test splits; the paper itself
  says it "does not cover some important domains such as medicine".
- Splits per dataset: a default time split, **3 sliding-window time splits**, and **3 random splits
  of identical sizes** made by shuffling the rows of each window (`preprocessing/*.py`).
- **Rows are never dropped for missing values.** Missing values become the column mean at training
  time (`paper/lib/data.py`, which itself calls this "a quick hack"). Row filters exist only for data
  errors/outliers (e.g. cooking time < 1 minute, impossible Sberbank floor areas/prices).
- Column filters: HomeCredit drops columns > 95% empty, constant columns, text columns with > 50
  values; Homesite drops all-empty columns and turns "only one non-null value" columns into 0/1.
- Features are NOT curated to be "good": realistic, noisy, correlated features are the point. The
  only hard rule is **no leakage**. Their "top-20 features only" experiment changed conclusions.

---

## 2. What existed before this session (partner's version)

Same source as now, much smaller: NHANES 2011-2016 (3 cycles), 15,964 adults, 2,191 diabetes cases,
**9 features** (age, income-to-poverty ratio, BMI, waist, height, weight, gender, race, education),
one time split, test ROC-AUC ~0.79 for every model. Judged too small/feature-poor vs TabReD.

---

## 3. What we built: expanded NHANES 2005-2018

Source: CDC NHANES public-use files, 7 cycles D-J (2005-06 ... 2017-18), 28 files per cycle
(196 files, 321 MB). Target: DIQ010 "Doctor told you have diabetes" (1 = yes, 2 = no; others excluded).
Eligibility (same as the original pipeline): age >= 20, has body-measurement (BMX) record, has DIQ
record, DIQ010 in {1, 2}.

| Cycle | Eligible adults | Diabetes cases | Prevalence |
|---|---|---|---|
| 2005-2006 | 4,682 | 485 | 10.4% |
| 2007-2008 | 5,608 | 723 | 12.9% |
| 2009-2010 | 5,935 | 715 | 12.0% |
| 2011-2012 | 5,202 | 676 | 13.0% |
| 2013-2014 | 5,416 | 707 | 13.1% |
| 2015-2016 | 5,346 | 808 | 15.1% |
| 2017-2018 | 5,098 | 836 | 16.4% |
| **Total** | **37,287** | **4,950** | |

(2011-16 counts match the partner's `results/data_flow.csv` exactly. Prevalence rises steadily:
a real **label shift**.) Other cycles exist (1999-2004, 2017-Mar 2020 pre-pandemic, 2021-23) but
2005-2018 has the most variables shared across all cycles (2021-23 cut most questionnaires).

**Final table: 37,287 rows x 280 features** (`data/processed/nhanes_wide_2005_2018.csv`, 41 MB).
151 features are <= 10% missing, 186 are <= 50% missing. Comparable to TabReD's smallest dataset
(Sberbank Housing, 28K rows) and near TabReD's median feature count (261).

Components used: DEMO, DIQ (target only), BMX, BPX, BPQ, SMQ, ALQ, PAQ, MCQ, KIQ_U, HIQ, HUQ, DPQ,
SLQ, WHQ, CDQ, OCQ, DBQ, HSQ, PFQ, DUQ, RHQ, TCHOL, HDL, TRIGLY, BIOPRO, CBC, ALB_CR.
Never downloaded (label-encoding): GLU, GHB (HbA1c), INS, OGTT, RXQ_RX (prescriptions).

### Cleaning rules (all in `scripts/build_wide_table.py`)
1. Merge the 28 files per cycle on SEQN (checked: one row per person in every file); stack cycles;
   keep only columns present in all 7 cycles; add `cycle`, `start_year`, `label`.
2. Drop admin/weights/design columns, lab comment codes, SI-unit duplicates, unit duplicates.
3. Refused / don't-know codes (7/9, 77/99, 777/999, 7777/9999, ...) -> missing, using the official
   CDC codebook codes saved in `configs/nhanes_missing_codes.json` (19,751 answers recoded).
4. "Mark all that apply" items (one code or blank) -> 0/1 indicators (25 columns).
5. Drop text columns (cigarette brand, UPC) and constant columns.
6. Drop columns > 95% empty (TabReD HomeCredit rule; 34 columns). Rows never dropped.
7. **Leakage removals:**
   - whole DIQ questionnaire except the target; serum glucose LBXSGL/LBDSGLSI in BIOPRO;
   - **LBXSOSSI serum osmolality**: rises with blood glucose -> glucose proxy
     (removing it alone cut test ROC-AUC by 0.02 in a check);
   - **BPQ090D/BPQ100D** cholesterol prescription: guideline treatment for diagnosed diabetics;
   - **PFQ063A-E "Health problem causing difficulty"**: answer code **16 = Diabetes** (the label).
     Found AFTER the first full run: it inflated LightGBM-like models from 0.899 to 0.924.
     The first run's results were discarded; `results_wide/` is the clean re-run.
   - Kept on purpose: MCQ300C "close relative had diabetes" (family history is known before diagnosis).
   - A scan of every remaining feature's answer options in the CDC codebooks found no other
     diabetes/insulin/sugar/glucose codes. No single feature has univariate ROC-AUC above 0.74 (age).
8. **Gray area, kept, mention as limitation:** answers that may partly reflect having been diagnosed
   (diet/weight-loss attempts WHD080*, DBQ700; "doctor said overweight" MCQ080; BP medication BPQ050A;
   general health HUQ010). Known at survey time so not leakage in TabReD's sense, but the task is best
   described as *identifying diagnosed diabetes from a health survey*, not forecasting onset
   (the original project already had the same caveat for BMI: "possible reverse causation").

Patient leakage: none possible. Each NHANES cycle samples different people (SEQN unique across cycles;
the benchmark asserts train/val/test SEQN sets are disjoint).

---

## 4. Benchmark design (`scripts/run_wide_benchmark.py`)

Reuses the original pipeline's code (`clinical_tabred.experiment.tune_model/evaluate_model`, bootstrap
CIs in `clinical_tabred.evaluation`), so the method matches the original project.

| Split | Train | Val | Test |
|---|---|---|---|
| default | 2005-2012 (21,427; 2,599 pos) | 2013-14 (5,416) | 2015-2018 (10,444; 1,644 pos) |
| sliding-window-0 | 2005-2010 | 2011-12 | 2013-14 |
| sliding-window-1 | 2007-2012 | 2013-14 | 2015-16 |
| sliding-window-2 | 2009-2014 | 2015-16 | 2017-18 |
| random-k | rows of window k shuffled and re-split into the same sizes (TabReD method) | | |

The default split is 72% train+val / 28% test, i.e. a time-ordered ~70/30 split (course asks 70/30 or 80/20).

Models: logreg, random_forest, xgboost, lightgbm, mlp (sklearn MLP). Small grids from
`configs/experiment.yaml`, selection by validation PR-AUC (ROC-AUC tiebreak), threshold = max F1 on
validation, test scored once. Preprocessing fitted on train only (median impute + scaling for numbers;
most-frequent impute + one-hot for categories). Feature typing: lab/exam components numeric;
questionnaire answers with <= 12 distinct codes categorical, else numeric (107 numeric / 173 categorical).
Bootstrap: 1,000 resamples, 95% CIs. Run time on Yelzhan's laptop ~30 min.

---

## 5. Final results (clean run, `results_wide/`)

### Test ROC-AUC per split
| Model | Default | Win 0 | Win 1 | Win 2 | Random 0 | Random 1 | Random 2 |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.882 | 0.903 | 0.892 | 0.899 | 0.900 | 0.915 | 0.901 |
| Random Forest | 0.876 | 0.889 | 0.877 | 0.871 | 0.883 | 0.904 | 0.880 |
| XGBoost | 0.900 | 0.914 | 0.901 | 0.903 | 0.903 | 0.918 | 0.906 |
| LightGBM | 0.899 | 0.914 | 0.900 | 0.903 | 0.901 | 0.913 | 0.905 |
| MLP | 0.887 | 0.902 | 0.892 | 0.897 | 0.905 | 0.914 | 0.901 |

### Default split, test set (threshold chosen on validation)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC [95% CI] | PR-AUC [95% CI] |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.849 | 0.515 | 0.720 | 0.601 | 0.882 [0.872, 0.891] | 0.619 [0.594, 0.645] |
| Random Forest | 0.829 | 0.471 | 0.707 | 0.566 | 0.876 [0.867, 0.884] | 0.571 [0.544, 0.597] |
| XGBoost | 0.868 | 0.571 | 0.659 | 0.612 | 0.900 [0.892, 0.908] | 0.654 [0.630, 0.677] |
| LightGBM | 0.861 | 0.548 | 0.675 | 0.605 | 0.899 [0.892, 0.906] | 0.646 [0.622, 0.669] |
| MLP | 0.863 | 0.555 | 0.671 | 0.607 | 0.887 [0.878, 0.895] | 0.634 [0.608, 0.661] |

(Original 9-feature project: ROC-AUC ~0.79 for every model.)

- Validation selected the **MLP** (best validation PR-AUC), but on the future test years
  **XGBoost and LightGBM beat it** with CIs excluding zero (ROC-AUC MLP minus XGBoost -0.014
  [-0.020, -0.007]; MLP minus LightGBM -0.013 [-0.019, -0.006]). Validation-on-2013-14 picked a
  model that generalises less well to 2015-2018: itself a temporal-drift finding.

### Random split vs time split (mean over the 3 windows, ROC-AUC)
| Model | Time | Random | Random minus time | Rank (time) | Rank (random) |
|---|---|---|---|---|---|
| LightGBM | 0.906 | 0.906 | +0.000 | 1 | 3 |
| XGBoost | 0.906 | 0.909 | +0.003 | 2 | 1 |
| Logistic Regression | 0.898 | 0.906 | +0.008 | 3 | 4 |
| MLP | 0.897 | 0.906 | +0.009 | 4 | 2 |
| Random Forest | 0.879 | 0.889 | +0.010 | 5 | 5 |

Reading (TabReD-consistent, but small effects; be careful with claims):
- Random splits look better for every model; the gap is **largest for MLP/LogReg/RF (~0.01) and
  ~0 for the GBDTs**. Under random splits the MLP looks as good as the GBDTs (rank 2); under time
  splits it falls to rank 4. This is the TabReD message: random splits hide the GBDT advantage.
- Per window, the gap is significant only in **window 1** (train 2007-12, test 2015-16): LogReg +0.023,
  RF +0.027, XGBoost +0.018, MLP +0.021 (CIs exclude 0), LightGBM +0.013 (CI includes 0).
  Windows 0 and 2: no significant gap. 2015-16 is where prevalence jumped 13.1% -> 15.1%.
- PR-AUC is higher under time splits only because later test years have higher prevalence
  (PR-AUC depends on prevalence). Use ROC-AUC for the random-vs-time comparison.

---

## 6. Course requirements (report due **Fri 20 Nov 2026 23:59**, Moodle) and status

Deliverables: IEEE-template report (>= 8 pages; Word, or LaTeX PDF + source ZIP), notebook named
`ProjectID.ipynb`, dataset (upload if < 100 MB: the 41 MB wide CSV qualifies, else a link).
Required content: abstract ... conclusion sections; project timeline, tech stack, results tables,
architecture/methodology block diagram; algorithms + equations; **train-test split (70/30 or 80/20)
AND 5- or 10-fold cross-validation**; accuracy/precision/recall/F1/ROC-AUC; >= 1 comparison table and
>= 1 performance chart; comparison with state of the art; training curves; confusion matrix; IEEE
citations; plagiarism < 10%, **AI-generated content < 20%**.

| Item | Status |
|---|---|
| Expanded NHANES dataset + leakage cleaning | Done |
| 5 models x 7 splits, bootstrap CIs, random-vs-time, rankings | Done (`results_wide/`) |
| Accuracy/precision/recall/F1/ROC-AUC/PR-AUC | Done (`results_wide/metrics.csv`) |
| Test predictions (for ROC curves, confusion matrices) | Done (`results_wide/predictions/test_predictions.csv`) |
| MLP training curves data | Done (`results_wide/mlp_history.csv`) |
| 5-fold cross-validation | Done for all three datasets (`scripts/run_wide_cv.py`, `docs/RESULTS.md` 2.7) |
| Drift characterised by clinical cause | Done (`scripts/analyze_drift.py`, `docs/RESULTS.md` 2.5-2.6) |
| Second and third datasets (benchmark = several datasets) | Done: BRFSS (93 features) and natality (73), full benchmark run (`docs/RESULTS.md`, `docs/HANDOFF_BRFSS_NATALITY.md`) |
| More deep tabular models (TabReD compares several) | Optional TODO |
| Slides updated to the new results | TODO (inputs ready: `docs/RESULTS.md`, `figures_wide/`) |
| Notebook `ProjectID.ipynb` updated to the new pipeline | TODO |
| Report rewritten with new results; [REVIEW] passages in own words | TODO |

---

## 7. Drift with a clinical cause: candidates to measure and link to performance
- NHANES blood-pressure measurement protocol: BPX (auscultatory) -> BPXO (oscillometric) around
  2017-18: a measurement **protocol change**.
- NHANES diabetes prevalence 10.4% (2005-06) -> 16.4% (2017-18): **population/label shift**.
- Window-1 performance drop at 2015-16 coincides with the prevalence jump.
- BRFSS: question renames/recodes (DIABETE3 -> DIABETE4 in 2019; INCOME2 -> INCOME3 in 2021 with
  more categories; ALCDAY5 -> ALCDAY4 in 2023): **coding revisions**. 2020 = COVID disruption.
  Blood pressure and cholesterol questions only asked in odd years (2017/19/21/23).

---

## 8. Second dataset: BRFSS (assessed from CDC docs, not yet downloaded/processed)

Partner already has LLCP 2021, 2022, 2023 locally. Plan: **2017-2023 (7 years)**, same split design
as NHANES. Download the missing years: `scripts/download_brfss.ps1 -Years 2017,2018,2019,2020`
(CDC returned 403 to the first script version; browser-like headers were added; if it still fails,
download the zips manually from the links in the script into `data/raw/brfss/`).

| Year | Respondents | Zip |
|---|---|---|
| 2017 | 450,016 | 107 MB |
| 2018 | 437,436 | 73 MB |
| 2019 | 418,268 | 98 MB |
| 2020 | 401,958 | 67 MB |
| 2021 | 438,693 | 82 MB |
| 2022 | 445,132 | 85 MB |
| 2023 | 433,323 | 93 MB |

- Total ~3.02M respondents; exact interview date (IDATE) available; independent samples each year.
- Target: DIABETE3 (2017-18) / DIABETE4 (2019+). 2019: 57,401 "yes" (13.7%); code 2 "only during
  pregnancy" (0.9%) -> exclude; code 4 "prediabetes" (2.2%) -> treat as "no".
- 275-358 columns per year; 159 present in all 7 years (incl. renamed). Of these: 36 admin/weights,
  **8 diabetes-module (leaky: DIABAGE, PREDIAB, INSULIN, CHKHEMO3, EYEEXAM, DIABEYE, DIABEDU, target)**,
  34 optional state modules (mostly empty), 52 real questions, 29 CDC calculated variables
  -> **~80 usable features** (vs 280 NHANES). Self-reported only: no labs/exams.
- Renamed variables sometimes changed coding (income, alcohol): match manually.
- Subsample (e.g. 50,000 per year = 350K rows, about TabReD scale); TabReD also subsampled big datasets.
- Avoid 2011 (method change) and 2024 (CDC modified the file to comply with executive orders).

Credentialed options (MIMIC-IV, SICdb, eICU) need PhysioNet approval (typically 1-2 weeks): not feasible
before 20 Nov.

---

## 8b. Third dataset: NCHS natality (US birth certificates), NOT yet assessed in detail

What we know so far (from the research report of this session; verify before relying on it):
- CDC/NCHS public-use natality micro-data, annual files 1968-2023, free download.
- ~3.6 million births per year (3,596,017 in 2023): subsample (TabReD subsampled big datasets).
- Time granularity in the public file: month and year (exact date of birth is restricted; no geography).
- Hundreds of birth-certificate items (exact count not verified).
- Possible targets: preterm birth, low birthweight, NICU admission, cesarean delivery.
- **Leakage risk is high**: many items are recorded at or after delivery (birthweight, gestational age,
  NICU, Apgar, delivery method, ...). Use only information available BEFORE delivery (maternal
  demographics, pregnancy history, prenatal care, risk factors) and check each item against the
  target. Example: never use gestational age to predict preterm birth.
- Likely clinical-cause drift: the 2003 revision of the US Standard Certificate of Live Birth was
  adopted by states gradually over many years (a **coding revision**); check the NCHS user guides
  for which years/states use the revised items before choosing the time window.
- The same mother can appear in several years (repeat births) and the public file has no person ID,
  so patient-level disjointness cannot be guaranteed; time splits reduce but do not remove this.
  State it as a limitation.

---

## 9. How to reproduce / continue

```
git checkout expanded-nhanes
uv venv --python 3.12
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
# Only if you need to rebuild the table (it is committed):
powershell -ExecutionPolicy Bypass -File scripts\download_nhanes.ps1   # Windows; 196 files, 321 MB
python scripts\build_wide_table.py                                       # ~40 s
# Re-run the benchmark (~30 min); --quick for a smoke test; --models / --splits for subsets:
python scripts\run_wide_benchmark.py
```

Files:
- `scripts/build_wide_table.py`: builds the wide table (all cleaning/leakage rules above).
- `scripts/run_wide_benchmark.py`: TabReD-style splits, tuning, evaluation, summary tables.
- `scripts/download_nhanes.ps1`, `scripts/download_brfss.ps1`: CDC downloaders (Windows PowerShell).
- `configs/nhanes_missing_codes.json`: refused/don't-know codes from the CDC codebooks.
- `data/processed/nhanes_wide_columns.csv`: every feature with source file, CDC label, % missing.
- `data/processed/nhanes_wide_flow.csv`: eligibility counts per cycle.
- `results_wide/`: `test_roc_auc_by_split.csv`, `metrics.csv`, `default_test_ci.csv`,
  `default_paired_comparisons.csv`, `random_minus_time_by_window.csv`, `random_minus_time_mean.csv`,
  `ranking_time_vs_random.csv`, `predictions/test_predictions.csv`, `mlp_history.csv`, `tuning.csv`,
  `selected_models.json`, `splits.csv`, `feature_types.csv`.
- `src/clinical_tabred/models.py`: XGBoost/LightGBM now imported lazily.

Note: on Windows, git shows ~60 files as modified purely because of line endings (CRLF); only
`.gitignore` and `models.py` had real edits in this session.

---

## 10. Instructions for processing BRFSS and natality (same standard as NHANES)

Goal: every dataset goes through the SAME benchmark so results are comparable.
1. Build ONE wide table per dataset: one row per respondent/birth, a time column (year, or
   year-month), the binary target, and features present in every year of the chosen window
   (TabReD-style: keep realistic/noisy features; do not curate for usefulness).
2. Same cleaning rules as `scripts/build_wide_table.py`: refused/don't-know/unknown codes -> missing
   (from the official codebook), drop admin/weight/ID columns, "mark all that apply" -> 0/1,
   drop columns > 95% empty, never drop rows for missing values.
3. **Leakage checks before any training** (each one found a real leak in NHANES):
   a. drop the target's own module/section and anything recorded after the outcome;
   b. scan every feature's answer options in the codebook for the target condition
      (NHANES PFQ063 had "16 = Diabetes" hidden in a list of causes);
   c. univariate ROC-AUC of every feature vs the target: anything far above the rest is suspicious;
   d. look for lab/physiological proxies of the target (NHANES serum osmolality ~ glucose);
   e. after training, check the most important features of the GBDT models; if GBDTs beat
      linear models by a lot, look for a leak the trees can exploit (that is how PFQ063 was found).
4. Same splits: default time split + 3 sliding windows + 3 size-matched random splits
   (copy `make_splits` from `scripts/run_wide_benchmark.py`; adapt the list of periods).
   Subsample large datasets per year (e.g. 50,000 rows/year) with a fixed seed.
5. Same models, grids, tuning (validation PR-AUC), threshold (validation max F1), bootstrap CIs,
   random-minus-time table and ranking table. Easiest: generalise `run_wide_benchmark.py` to take
   the table path and the period column as arguments.
6. Add 5-fold CV (course requirement) for all datasets.
7. Record the clinical cause of each drift (protocol change / coding revision / population shift)
   next to the per-window performance changes; that is the project's stated contribution.

---

## 11. Immediate next steps
1. Presentation (16 Nov): update the deck with section 5 results; add a slide "Toward a Clinical
   TabReD benchmark": NHANES done; BRFSS (and maybe Mexico COVID) next; drift-by-clinical-cause plan.
2. Add 5-fold CV to `run_wide_benchmark.py` (on the training+validation years only).
3. BRFSS and natality pipelines (section 10): loaders, harmonised features, targets, leakage checks,
   same benchmark.
4. Drift analysis by clinical cause (section 7) and link to the per-window performance gaps.
5. Notebook + report (IEEE template, >= 8 pages), own-words rewriting.

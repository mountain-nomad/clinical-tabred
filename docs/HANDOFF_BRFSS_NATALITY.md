# Handoff: BRFSS and natality tables (session of 16 Nov 2026)

Continues `docs/HANDOFF_EXPANDED_NHANES.md` (sections 8, 8b and 10). Both datasets are now built
with the same rules as the NHANES wide table and run through the same benchmark script.
**Status (16 Nov, 03:30): tables done, leakage checks done, all experiments done for all three
datasets (benchmark, 5-fold CV, drift analysis). Results: `docs/RESULTS.md`, tables in
`results_summary/`, figures in `figures_wide/`.**

| | NHANES (done before) | BRFSS (new) | Natality (new) |
|---|---|---|---|
| Source | CDC NHANES 2005-2018 | CDC BRFSS LLCP 2017-2023 | NCHS birth certificates 2016-2024 |
| Unit / time | adult / 2-year cycle | adult / survey year | birth / birth year (+ month) |
| Target | diagnosed diabetes (DIQ010) | diagnosed diabetes (DIABETE3/4) | preterm birth (< 37 weeks, OE) |
| Rows | 37,287 (all eligible) | 350,000 (50,000 per year sampled) | 450,000 (50,000 per year sampled) |
| Positives | 4,950 (13.3%) | 47,523 (13.6%) | 45,824 (10.2%) |
| Features | 280 (107 num / 173 cat) | 93 (19 num / 74 cat) | 73 (16 num / 57 cat) |
| Table | `nhanes_wide_2005_2018.csv` | `brfss_wide_2017_2023.csv.gz` (11 MB) | `natality_wide_2016_2024.csv.gz` (11 MB) |

All files are in `data/processed/`, each with a `*_columns.csv` (feature, type, label, % missing)
and a `*_flow.csv` (counts per year). Table columns: `row_id, period, year, month, label, features`
(`month` = BRFSS interview month / natality birth month: metadata for over-time plots, not a feature).

## Feature count (decision)
Feature counts were not targeted. Each table keeps every leakage-free feature its source provides in
every year, **including the agencies' own derived variables** (CDC calculated variables in BRFSS,
NCHS recodes and imputation flags in natality), mirroring TabReD's use of production feature sets
with their correlated features. A first version dropped these as "recodes" (64 / 50 features); that
was curation, so it was reversed. Clinical survey/registry data are narrower than industrial logs:
TabReD's datasets have 103-986 features (median 261). NHANES (280) meets both TabReD conditions;
BRFSS (93) and natality (73) mainly test the temporal-shift condition, which also gives a
feature-richness contrast comparable to TabReD's reduced-feature experiment (Table 4).
Not done on purpose (would add features only to reach a number): keeping BRFSS odd-year-only
questions (blank in even years) or engineering historical aggregates.

## How to rebuild / run

```
python scripts/build_natality_table.py            # ~35 s, streams the 9 zips (no extraction)
python scripts/parse_brfss_codebooks.py           # only if configs/brfss_codebook.json must be rebuilt
python scripts/build_brfss_table.py               # ~1.5 min, reads the XPT files inside the zips
python scripts/run_wide_benchmark.py --dataset brfss      # -> results_brfss/
python scripts/run_wide_benchmark.py --dataset natality   # -> results_natality/
python scripts/run_wide_benchmark.py                      # NHANES, unchanged -> results_wide/
```
`--per-year N` changes the sample size (default 50,000, seed 42). `--quick` works for every dataset.

### Large zips
Nothing is unzipped to disk (one natality year is ~5 GB of text; 2 GB of zips in total).
Natality is streamed line by line through `unzip -p` (several NCHS zips use Deflate64, which
Python's `zipfile` cannot read) and sampled with reservoir sampling, so every eligible birth has
the same chance of selection and the full-year counts in `natality_wide_flow.csv` are exact.
BRFSS XPT files (~1 GB each) are read straight from the zip with pandas.

## BRFSS 2017-2023

- **Target**: 1 = yes; 3 = no and 4 = pre-diabetes -> 0; 2 (only during pregnancy), 7, 9 -> excluded.
  Prevalence per year (all eligible respondents): 13.6, 14.0, 13.9, **13.1 (2020)**, 13.3, 13.9, 13.9%.
- **Codebooks**: cdc.gov blocks scripted downloads, so the 7 LLCP codebooks were saved by hand into
  `data/raw/brfss/codebooks/` (PDF for 2017/18/20/21, HTML for 2019/22/23).
  `scripts/parse_brfss_codebooks.py` extracts every variable's value codes into
  `configs/brfss_codebook.json`. Codes labelled don't know / not sure / refused / not asked ->
  missing; "None" (88/888) -> 0 for count questions (days, drinks, children). If a year's codebook
  lacks a variable (TRNSGNDR in 2023), the nearest year's codes are used.
- **Harmonised renames** (same codes, checked year by year): SEX/SEX1/SEXVAR, ADDEPEV2/3, ALCDAY5/4,
  AVEDRNK2/3, CHCCOPD1/2/3, CHCKIDNY/CHCKDNY1/2, CHCOCNCR/1, CHCSCNCR/1, DRNKANY5/6, FLUSHOT6/7,
  HAVARTH3/4/5, HIVTST6/7, MEDCOST/1, PERSDOC2/3, TETANUS/1. Recoded: INCOME3 (2021+, top codes
  9-11 = $100K+) collapsed into INCOME2's top code 8 (>= $75K); ALCDAY (1xx per week / 2xx per
  month) -> days per 30; health coverage HLTHPLN1 (to 2020) continued by CDC's `_HLTHPLN/_HLTHPL1`;
  adults in household = NUMADULT (landline) or HHADULT (cell).
- **Only features recorded in every year**: rotating core questions asked only in odd years
  (blood pressure, cholesterol, fruit/vegetables, physical activity) are therefore not in the table.
- **CDC calculated variables kept** and harmonised across renames (_RFBING5/6, _RFDRHV5-8,
  _DRNKWEK/_DRNKWK1/2, DROCDY3_/4_, _AIDTST3/4, _FLSHOT6/7, _HCVU651-3, _DRDXAR1-3, race recodes).
  Recoded: _INCOMG1 (2021+) collapsed to the old 5 groups; _MRACE2 (2022: "other race only"
  removed, multiracial renumbered 7 -> 6, 88 = no choice) mapped back; _PRACE2 88 -> missing.
- **Dropped**: admin, weights, design, phone-screening columns; the random-child module (describes a
  child, not the respondent); unit duplicates (HTIN4, WEIGHT2, HEIGHT3); date-coded items
  (HIVTSTD3, COVID shot dates); 13 columns > 95% empty.
- **Leakage removals**: the diabetes module and pre-diabetes module (25 columns: DIABAGE, INSULIN,
  CHKHEMO3, EYEEXAM, DIABEYE, DIABEDU, BLDSUGAR, FEETCHK, PREDIAB, PDIABTST, ...). These are asked
  only of diabetics or only of non-diabetics, so even their *missingness* gives the label away.
  Also the label-text filter (diabet/insulin/sugar/glucose/A1C), and **pneumococcal vaccination**
  (PNEUVAC, _PNEUMO). That vaccine is recommended for adults with diabetes, so it is a consequence of
  the diagnosis (same reasoning as NHANES BPQ090D).
- **Gray area, kept** (same as NHANES): kidney disease, difficulty seeing, time since last checkup
  and BMI may partly reflect the diagnosis or its care.
- **Checks**: highest univariate ROC-AUC GENHLTH 0.71, age 0.68, BMI 0.67; highest
  missingness-only ROC-AUC 0.59 (drinking follow-ups asked only of drinkers). Default split
  (train 2017-20, test 2022-23): LightGBM 0.825 vs logistic regression 0.822; top LightGBM gain:
  GENHLTH 29%, age 14% (+ age groups 8%), BMI 12%, CHECKUP1 5%. No sign of a leak.

## Natality 2016-2024

- **Eligible**: US residents (RESTATUS != 4, as NCHS does) with stated gestation. ~3.6-3.96M births
  per year, 0.3% excluded. Preterm rate per year 9.85, 9.93, 10.02, 10.23, 10.09, 10.49, 10.38, 10.41,
  10.41%. These match the published NCHS figures, which confirms the field positions.
- **Target**: preterm = obstetric estimate of gestation (OEGest_Comb, NCHS standard since 2014) < 37 weeks.
- **Field positions**: from the NCHS User Guides; the layout is identical 2016-2024 for every field
  used (only bridged race was discontinued and MHISPX/FHISPX added; the latter are empty in 2016
  and are dropped by the every-year rule).
- **Features (73)**: only items known before delivery, plus the NCHS recodes (age groups, race
  recode 6, birth order, interval recodes, care-began recode, cigarette recodes, BMI recode, payer
  recode) and imputation flags (age, race, marital status, father's age, plurality, sex) of those items: mother's age, nativity, residence, race and
  Hispanic origin, marital status, education; father's age, race, education; prior births,
  terminations and birth intervals; month prenatal care began; WIC; cigarettes before pregnancy and
  in trimesters 1-2; height, pre-pregnancy weight and BMI; risk factors (pre-pregnancy/gestational
  diabetes and hypertension, eclampsia, previous preterm birth, infertility treatment, previous
  cesareans); infections; payer; plurality; infant sex. Unknown codes (9, 99, 999, U) -> missing.
- **Leakage removals** (full list with reasons in `LEAKAGE_EXCLUDED` in the script):
  gestation variables; last-menstrual-period date (with the birth date it *is* gestational age);
  birth weight; **number of prenatal visits** and **weight gain / delivery weight** (both grow with
  pregnancy length); **3rd-trimester smoking**, which the User Guide says is edited to "unknown"
  when gestation < 27 weeks, so it encodes very preterm birth; external cephalic version (done at
  >= 36 weeks); labor and delivery items (e.g. antenatal steroids are given *because* preterm birth
  is expected); method of delivery; maternal morbidity; place/attendant/transfer; time and weekday
  of birth; Apgar, NICU and other newborn items; paternity acknowledgement (after birth).
  Reporting flags (F_*) are administrative and not read; bridged race was discontinued in 2019.
- **Gray area, kept**: infant sex (known prenatally for most births); eclampsia (may be diagnosed
  at delivery, rare: 0.2-0.3%).
- **Checks**: highest univariate ROC-AUC "no risk factors" 0.60, plurality 0.59. Default split
  (train 2016-21, test 2023-24): LightGBM 0.753 vs logistic regression 0.744; top gain: plurality
  42% (twins/triplets are mostly preterm, a real pre-delivery predictor), month care began 7%,
  no risk factors 6%, previous preterm birth 6%. No sign of a leak. Preterm birth from pre-delivery
  data is a hard task (published models ~0.65-0.75), so lower scores than NHANES are expected.
- **Limitation**: no mother ID in the public file. Twins of one delivery or births to the same mother
  can appear in the same year. With 50,000 of ~3.7M births per year sampled, very few such pairs
  land in the sample.

## Splits (`scripts/run_wide_benchmark.py`, `DATASETS`)

| Dataset | Default (train / val / test) | Sliding windows (train 3, val 1, test 1) |
|---|---|---|
| NHANES | 2005-12 / 2013-14 / 2015-18 | tests 2013-14, 2015-16, 2017-18 |
| BRFSS | 2017-20 / 2021 / 2022-23 (57/14/29%) | tests 2021, 2022, 2023 |
| Natality | 2016-21 / 2022 / 2023-24 (67/11/22%, ~80/20) | tests **2020 (COVID)**, 2022, 2024 |

Random splits are the same TabReD construction (window rows shuffled, same sizes). Feature types
come from the `type` column of `*_columns.csv` for the new datasets. The NHANES rules are unchanged.

## Drift with a clinical cause seen while building (candidates for section 7 of the NHANES handoff)

| Dataset | Change | Cause type |
|---|---|---|
| BRFSS | 2021: PERSDOC question reworded; "more than one personal doctor" 8% -> 28% | coding revision |
| BRFSS | 2021: health-coverage yes/no replaced by "primary source of insurance"; uninsured 8% -> 5% | coding revision |
| BRFSS | 2021: income question gets 3 new top categories (INCOME2 -> INCOME3) | coding revision |
| BRFSS | 2020: diabetes prevalence dips to 13.1% (COVID-year sample) | population shift |
| BRFSS | landline-only MSCODE missing 57% -> 81% (shift to cell-phone sample) | survey-mode change |
| Natality | from 2017: marital status withheld for California births (state law): 0% -> 12% missing | reporting change |
| Natality | gestational hypertension 5.8% -> 10.3%, pre-pregnancy hypertension 1.7% -> 3.4%, gestational diabetes 5.9% -> 8.5% (2016 -> 2024) | population shift / diagnostic practice |
| Natality | pre-pregnancy smoking (mean cigarettes/day) 1.23 -> 0.39; WIC 40% -> 32% | population shift |
| Natality | 2020-23: more unknown father information (hospital COVID policies, per User Guide) | protocol change |
| Natality | preterm rate 9.85% -> 10.49% (2021 peak) | label shift |

## Next steps
1. Presentation (on another machine): storyline and numbers in `docs/RESULTS.md`, figures in
   `figures_wide/` (PNG for slides, PDF for the report).
2. Report and notebook were **not** updated to the three-dataset results in this session (out of
   scope by decision on 16 Nov). The tables in `results_summary/summary.md` and the figures are ready
   to drop in. `report/main.tex` and `notebooks/CSCI447P034.ipynb` still describe the original
   9-feature NHANES 2011-2016 version.
3. Optional: MLP-PLR / FT-Transformer (TabReD's strongest deep models) on the RTX 5080 GPU; CatBoost.

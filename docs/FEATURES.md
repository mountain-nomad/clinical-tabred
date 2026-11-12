# Features and exclusions

All variables were checked in the official NHANES codebooks of cycles G (2011-2012),
H (2013-2014) and I (2015-2016) on 2026-11-12 (`https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/<year>/DataFiles/<FILE>.htm`).
Coding and special codes are identical in the three cycles.
Missing-value rates per cycle are in `results/missingness.csv`.

## Model inputs (9)

| Feature | Meaning | Type | Coding / range | Special codes | Source file |
|---|---|---|---|---|---|
| `RIDAGEYR` | Age in years at screening | numeric | 0–79; **80 = 80 years and over (top-coded)** | none | DEMO |
| `RIAGENDR` | Gender | categorical | 1 Male, 2 Female | none | DEMO |
| `RIDRETH3` | Race/Hispanic origin incl. Non-Hispanic Asian | categorical | 1 Mexican American, 2 Other Hispanic, 3 NH White, 4 NH Black, 6 NH Asian, 7 Other/multi-racial | none | DEMO |
| `DMDEDUC2` | Education level, adults 20+ | categorical | 1 < 9th grade … 5 College graduate+ | **7 Refused, 9 Don't know → missing** | DEMO |
| `INDFMPIR` | Ratio of family income to poverty | numeric | 0–4.99; **5 = 5.00 or more (top-coded)** | missing when income not reported | DEMO |
| `BMXBMI` | Body mass index (kg/m²) | numeric | continuous | missing if not measured | BMX |
| `BMXWAIST` | Waist circumference (cm) | numeric | continuous | missing if not measured | BMX |
| `BMXHT` | Standing height (cm) | numeric | continuous | missing if not measured | BMX |
| `BMXWT` | Weight (kg) | numeric | continuous | missing if not measured | BMX |

Preprocessing (fitted on the training split only): numeric → median imputation → standard scaling;
categorical → most-frequent imputation → one-hot encoding (codes are categories, not quantities).

Feature groups used in the ablation: demographic = `RIDAGEYR`, `RIAGENDR`, `RIDRETH3`;
socioeconomic = `DMDEDUC2`, `INDFMPIR`; anthropometric = `BMXBMI`, `BMXWAIST`, `BMXHT`, `BMXWT`.

## Target

| Variable | Meaning | Mapping | Source |
|---|---|---|---|
| `DIQ010` | "Doctor told you have diabetes" | 1 Yes → **1**; 2 No → **0**; 3 Borderline, 7 Refused, 9 Don't know, missing → **excluded** | DIQ |

## Metadata (never model inputs)

| Column | Use |
|---|---|
| `SEQN` | Respondent ID: merging and split-overlap checks only |
| `cycle` | Survey cycle: defines the chronological splits |
| `WTINT2YR` | Interview weight: descriptive weighted prevalence only |

## Excluded variables and reasons

| Variable(s) | Reason |
|---|---|
| `SEQN`, cycle/year | Identifiers; would leak the split structure, not health information |
| `DIQ010` | The target itself |
| Other DIQ answers (e.g. age at diagnosis, insulin/pill use, `DIQ050`, `DIQ070`, `DIQ160`) | Reveal diagnosis or treatment → target leakage |
| Glucose, HbA1c, insulin laboratory values (GLU, GHB, INS files) | Diagnostic criteria of diabetes → near-tautological task |
| `WTMEC2YR`, `WTINT2YR`, `SDMVPSU`, `SDMVSTRA` | Survey weights/design variables, not respondent characteristics |
| Any variable derived from the target | Leakage |

## 2. Dataset description

**Source:** CDC National Health and Nutrition Examination Survey (NHANES), public-use files from
<https://wwwn.cdc.gov/nchs/nhanes/>. Three two-year cycles are used: 2011-2012 (file suffix `G`),
2013-2014 (`H`) and 2015-2016 (`I`). From each cycle we use three components:

| Component | Content | Variables used |
|---|---|---|
| DEMO | Demographics | `RIDAGEYR` age (top-coded at 80), `RIAGENDR` gender, `RIDRETH3` race/Hispanic origin, `DMDEDUC2` education (7 = refused, 9 = don't know → missing), `INDFMPIR` family income-to-poverty ratio (top-coded at 5.0), `WTINT2YR` interview weight (description only) |
| BMX | Body measures | `BMXBMI` BMI, `BMXWAIST` waist circumference, `BMXHT` height, `BMXWT` weight |
| DIQ | Diabetes questionnaire | `DIQ010` target |

**Target:** `DIQ010` "Doctor told you have diabetes": 1 = yes → **1**, 2 = no → **0**; 3 (borderline),
7 (refused), 9 (don't know) and missing are **excluded**. This is a cross-sectional classification of
*self-reported diagnosed* diabetes, not a prediction of future onset.

**Population:** adults aged 20 or older with a body-measurement record and a valid label.

**Never used as features:** the respondent ID `SEQN`, the cycle, the target, other diabetes answers,
glucose/HbA1c/insulin laboratory values (they define diabetes, which would make the task trivial),
and survey weights or design variables. Every variable was checked in the official codebook of all three cycles.

NHANES is a complex multistage survey; we train **unweighted** models and only report weighted prevalence
as a descriptive check.

| dataset   | feature   | statistic   | cause                            | change_at   | description                                                                                               |
|:----------|:----------|:------------|:---------------------------------|:------------|:----------------------------------------------------------------------------------------------------------|
| nhanes    | label     | mean        | label / population shift         | steady      | Diagnosed-diabetes prevalence rises cycle after cycle                                                     |
| nhanes    | HIQ011    | share == 2  | policy change                    | 2015-2016   | Not covered by health insurance: falls after the 2014 Affordable Care Act coverage expansion              |
| nhanes    | HIQ031D   | mean        | policy change                    | 2015-2016   | Covered by Medicaid: rises with the 2014 ACA Medicaid expansion                                           |
| nhanes    | BMXBMI    | mean        | population shift                 | steady      | Mean measured BMI rises (obesity trend)                                                                   |
| nhanes    | LBXRDW    | mean        | protocol change (lab instrument) | 2013-2014   | Hematology analyzer replaced by Beckman Coulter DxH 800; CDC: mean RDW 12.84% -> 13.55% (CBC_H docs)      |
| nhanes    | LBXSLDSI  | mean        | protocol change (lab instrument) | 2017-2018   | Biochemistry analyzer and lab changed to Roche Cobas 6000; CDC bridging study: LDH +14.9% (BIOPRO_J docs) |
| nhanes    | LBXSASSI  | mean        | protocol change (lab instrument) | 2017-2018   | Same 2017-2018 analyzer change; CDC bridging study: AST -18.0% (BIOPRO_J docs)                            |
| brfss     | label     | mean        | population shift                 | 2020        | Diagnosed-diabetes prevalence dips in the COVID-19 year 2020                                              |
| brfss     | PERSDOC   | share == 2  | coding revision                  | 2021        | PERSDOC2 -> PERSDOC3 rewording: 'more than one personal doctor' jumps                                     |
| brfss     | HLTHPLN   | share == 2  | coding revision                  | 2021        | Yes/no coverage question replaced by 'primary source of insurance' (CDC-derived series)                   |
| brfss     | _HCVU65   | share == 2  | coding revision                  | 2021        | CDC-derived 'no coverage, age 18-64' follows the same 2021 question change                                |
| brfss     | _MRACE    | share == 6  | coding revision                  | 2022        | 2022 race question: 'other race only' category removed                                                    |
| brfss     | MSCODE    | missing     | survey-mode change               | steady      | Landline-only metro code: missing grows as the sample shifts to cell phones                               |
| natality  | label     | mean        | label shift                      | 2021        | Preterm-birth rate rises to a 2021 peak                                                                   |
| natality  | DMAR      | missing     | reporting change (state law)     | 2017        | Marital status withheld for California births from 2017                                                   |
| natality  | RF_GHYPE  | mean        | population / diagnostic practice | steady      | Gestational hypertension nearly doubles                                                                   |
| natality  | RF_PHYPE  | mean        | population / diagnostic practice | steady      | Pre-pregnancy hypertension doubles                                                                        |
| natality  | RF_GDIAB  | mean        | population / diagnostic practice | steady      | Gestational diabetes rises                                                                                |
| natality  | CIG_0     | mean        | population shift                 | steady      | Cigarettes per day before pregnancy fall by about two thirds                                              |
| natality  | FAGECOMB  | missing     | protocol change (COVID-19)       | 2020        | Hospital COVID-19 policies: more births with unknown father information (NCHS User Guide)                 |

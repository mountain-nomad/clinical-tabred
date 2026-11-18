| TabReD claim                                          | NHANES (280 features)          | BRFSS (93)                                          | Natality (73)                   |
|:------------------------------------------------------|:-------------------------------|:----------------------------------------------------|:--------------------------------|
| Random splits overestimate future performance         | yes, in the label-shift window | yes, small but consistent, in coding-revision years | no (period-difficulty artefact) |
| Random splits change the model ranking                | yes (MLP 2nd -> 4th)           | no                                                  | partly (MLP, LR)                |
| GBDTs are the strongest models                        | yes                            | yes                                                 | yes                             |
| XGBoost's lead over the MLP shrinks under time splits | no, it grows                   | no (similar)                                        | no (same)                       |

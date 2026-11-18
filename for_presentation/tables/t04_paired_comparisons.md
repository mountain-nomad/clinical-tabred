| reference   | other         | metric   |    diff |   diff_low |   diff_high | ci_excludes_zero   | dataset   |
|:------------|:--------------|:---------|--------:|-----------:|------------:|:-------------------|:----------|
| mlp         | lightgbm      | roc_auc  | -0.0125 |    -0.0193 |     -0.0058 | True               | nhanes    |
| mlp         | logreg        | roc_auc  |  0.0049 |     0.0022 |      0.0078 | True               | nhanes    |
| mlp         | random_forest | roc_auc  |  0.0109 |     0.0030 |      0.0186 | True               | nhanes    |
| mlp         | xgboost       | roc_auc  | -0.0137 |    -0.0203 |     -0.0069 | True               | nhanes    |
| xgboost     | lightgbm      | roc_auc  | -0.0019 |    -0.0026 |     -0.0011 | True               | brfss     |
| xgboost     | logreg        | roc_auc  |  0.0021 |     0.0009 |      0.0032 | True               | brfss     |
| xgboost     | mlp           | roc_auc  | -0.0000 |    -0.0010 |      0.0010 | False              | brfss     |
| xgboost     | random_forest | roc_auc  |  0.0025 |     0.0014 |      0.0036 | True               | brfss     |
| xgboost     | lightgbm      | roc_auc  |  0.0000 |    -0.0008 |      0.0008 | False              | natality  |
| xgboost     | logreg        | roc_auc  |  0.0062 |     0.0044 |      0.0080 | True               | natality  |
| xgboost     | mlp           | roc_auc  |  0.0049 |     0.0031 |      0.0067 | True               | natality  |
| xgboost     | random_forest | roc_auc  |  0.0062 |     0.0043 |      0.0082 | True               | natality  |

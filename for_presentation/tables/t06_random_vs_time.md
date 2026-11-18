| dataset   | model               |   time |   random |   random_minus_time |   windows_significant |   rank_time |   rank_random |
|:----------|:--------------------|-------:|---------:|--------------------:|----------------------:|------------:|--------------:|
| brfss     | LightGBM            | 0.8263 |   0.8334 |              0.0071 |                     2 |           1 |             1 |
| brfss     | XGBoost             | 0.8259 |   0.8330 |              0.0071 |                     2 |           2 |             2 |
| brfss     | MLP                 | 0.8240 |   0.8314 |              0.0074 |                     2 |           3 |             3 |
| brfss     | Logistic Regression | 0.8224 |   0.8296 |              0.0072 |                     2 |           4 |             4 |
| brfss     | Random Forest       | 0.8221 |   0.8285 |              0.0064 |                     1 |           5 |             5 |
| natality  | XGBoost             | 0.7533 |   0.7473 |             -0.0060 |                     1 |           1 |             2 |
| natality  | LightGBM            | 0.7532 |   0.7487 |             -0.0045 |                     1 |           2 |             1 |
| natality  | MLP                 | 0.7470 |   0.7410 |             -0.0060 |                     1 |           3 |             5 |
| natality  | Logistic Regression | 0.7465 |   0.7427 |             -0.0038 |                     0 |           4 |             3 |
| natality  | Random Forest       | 0.7454 |   0.7421 |             -0.0033 |                     0 |           5 |             4 |
| nhanes    | LightGBM            | 0.9059 |   0.9063 |              0.0004 |                     0 |           1 |             3 |
| nhanes    | XGBoost             | 0.9057 |   0.9089 |              0.0031 |                     1 |           2 |             1 |
| nhanes    | Logistic Regression | 0.8979 |   0.9056 |              0.0077 |                     1 |           3 |             4 |
| nhanes    | MLP                 | 0.8970 |   0.9063 |              0.0093 |                     1 |           4 |             2 |
| nhanes    | Random Forest       | 0.8791 |   0.8887 |              0.0097 |                     1 |           5 |             5 |

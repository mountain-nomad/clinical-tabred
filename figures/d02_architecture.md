# System architecture

## Boxes
1. **NHANES table 6 numeric + 3 categorical** (position x=0, y=2.4)
2. **ColumnTransformer impute + scale / impute + one-hot (fit on train only)** (position x=2.9, y=2.4)
3. **Logistic Regression** (position x=5.8, y=4.8)
4. **Random Forest** (position x=5.8, y=3.6)
5. **XGBoost** (position x=5.8, y=2.4)
6. **LightGBM** (position x=5.8, y=1.2)
7. **MLP (epoch loop, early stopping)** (position x=5.8, y=0.0)
8. **Threshold max F1 on validation** (position x=8.7, y=2.4)
9. **Metrics ROC/PR-AUC, F1, Brier, bootstrap CI** (position x=11.6, y=2.4)

## Arrows
- NHANES table 6 numeric + 3 categorical → ColumnTransformer impute + scale / impute + one-hot (fit on train only)
- ColumnTransformer impute + scale / impute + one-hot (fit on train only) → Logistic Regression
- ColumnTransformer impute + scale / impute + one-hot (fit on train only) → Random Forest
- ColumnTransformer impute + scale / impute + one-hot (fit on train only) → XGBoost
- ColumnTransformer impute + scale / impute + one-hot (fit on train only) → LightGBM
- ColumnTransformer impute + scale / impute + one-hot (fit on train only) → MLP (epoch loop, early stopping)
- Logistic Regression → Threshold max F1 on validation
- Random Forest → Threshold max F1 on validation
- XGBoost → Threshold max F1 on validation
- LightGBM → Threshold max F1 on validation
- MLP (epoch loop, early stopping) → Threshold max F1 on validation
- Threshold max F1 on validation → Metrics ROC/PR-AUC, F1, Brier, bootstrap CI

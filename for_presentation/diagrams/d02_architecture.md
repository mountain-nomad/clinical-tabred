# Model pipeline (identical for every dataset and split)

## Boxes (left to right, top to bottom)
1. **Wide table**: NHANES 280 / BRFSS 93 / Natality 73 features
2. **ColumnTransformer**: numeric: median + scale categorical: mode + one-hot (fit on train only)
3. **Logistic Regression**: C grid, balanced
4. **Random Forest**: depth / leaf grid
5. **XGBoost**: scale_pos_weight
6. **LightGBM**: scale_pos_weight
7. **MLP**: epoch loop, early stopping
8. **Model selection**: validation PR-AUC (ROC-AUC tiebreak)
9. **Threshold**: max F1 on validation
10. **Test, scored once**: metrics, confusion, bootstrap 95% CI

## Arrows
- Wide table -> ColumnTransformer
- ColumnTransformer -> Logistic Regression
- ColumnTransformer -> Random Forest
- ColumnTransformer -> XGBoost
- ColumnTransformer -> LightGBM
- ColumnTransformer -> MLP
- Logistic Regression -> Model selection
- Random Forest -> Model selection
- XGBoost -> Model selection
- LightGBM -> Model selection
- MLP -> Model selection
- Model selection -> Threshold
- Threshold -> Test, scored once

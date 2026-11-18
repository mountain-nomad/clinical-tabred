# Methodology: a clinical TabReD benchmark

## Boxes (left to right, top to bottom)
1. **1 Public data**: NHANES 2005-18 BRFSS 2017-23 Natality 2016-24
2. **2 Build tables**: harmonise codes, missing codes, every-period rule
3. **3 Leakage audit**: text filter, univariate AUC, missingness AUC
4. **4 TabReD splits**: default time split 3 sliding windows 3 random (same size)
5. **5 Train 5 models**: LR, RF, XGBoost, LightGBM, MLP tuned on validation
6. **6 Evaluate**: metrics + bootstrap CI random vs time same-period, 5-fold CV
7. **7 Drift by cause**: domain classifier PSI / TVD per feature clinical-cause events

## Arrows
- 1 Public data -> 2 Build tables
- 2 Build tables -> 3 Leakage audit
- 3 Leakage audit -> 4 TabReD splits
- 4 TabReD splits -> 5 Train 5 models
- 5 Train 5 models -> 6 Evaluate
- 6 Evaluate -> 7 Drift by cause

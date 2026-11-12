# Model comparison

Generated from the CSV files in this folder.

## Chronological test set (2015-2016)

| Model | Acc. | Prec. | Rec. | F1 | ROC-AUC [95% CI] | PR-AUC [95% CI] | Brier | Train (s) |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.767 | 0.343 | 0.594 | 0.435 | 0.790 [0.774, 0.805] | 0.364 [0.336, 0.399] | 0.198 | 0.01 |
| Random Forest | 0.772 | 0.350 | 0.588 | 0.438 | 0.790 [0.775, 0.807] | 0.362 [0.332, 0.396] | 0.149 | 0.30 |
| XGBoost | 0.752 | 0.336 | 0.656 | 0.444 | 0.791 [0.775, 0.807] | 0.368 [0.339, 0.402] | 0.186 | 0.09 |
| LightGBM | 0.747 | 0.331 | 0.661 | 0.441 | 0.785 [0.769, 0.800] | 0.360 [0.330, 0.394] | 0.186 | 0.11 |
| MLP | 0.744 | 0.325 | 0.646 | 0.433 | 0.792 [0.777, 0.807] | 0.362 [0.333, 0.394] | 0.193 | 0.26 |

## Test ROC-AUC by protocol

| Model | A2 in-period | A2 future | A chronological | B random (mean ± std) |
|---|---|---|---|---|
| Logistic Regression | 0.813 | 0.788 | 0.790 | 0.799 ± 0.011 |
| Random Forest | 0.821 | 0.792 | 0.790 | 0.804 ± 0.012 |
| XGBoost | 0.823 | 0.787 | 0.791 | 0.804 ± 0.015 |
| LightGBM | 0.819 | 0.783 | 0.785 | 0.803 ± 0.015 |
| MLP | 0.822 | 0.791 | 0.792 | 0.805 ± 0.011 |

## Key findings

- Eligible adults: 15,964; unweighted prevalence 13.0% (2011-2012), 13.1% (2013-2014), 15.1% (2015-2016).
- Model selected on validation: **XGBoost**. Chronological test (2015-2016) ROC-AUC 0.791 [0.775, 0.807], PR-AUC 0.368 [0.339, 0.402], F1 0.444 at threshold 0.60.
- Same model, random split: ROC-AUC 0.804 ± 0.015 (5 seeds); seed-42 random split minus chronological test -0.004 [-0.033, 0.023].
- Protocol A2 (same fitted model): in-period holdout minus 2015-2016 ROC-AUC +0.036 [-0.001, 0.070].
- Covariate shift 2011-2012 vs 2015-2016: largest numeric KS statistic 0.057; domain-classifier ROC-AUC 0.582.
- Ablation: dropping demographic features changes ROC-AUC by -0.086; dropping body measurements by -0.043.

- Paired bootstrap on 2015-2016: XGBoost has a clearly higher ROC-AUC than: LightGBM; clearly lower than: none; all other pairwise differences are inconclusive.
- Random-split minus chronological ROC-AUC (selected model) is inconclusive (the 95% CI includes zero).
- In-period holdout minus future-cycle ROC-AUC (protocol A2, selected model) is inconclusive (the 95% CI includes zero).

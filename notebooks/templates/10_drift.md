## 10. Temporal drift analysis

We compare the 2013-2014 and 2015-2016 cycles with 2011-2012:
* **Numeric features:** means, medians, standard deviations and the two-sample **Kolmogorov-Smirnov (KS)
  statistic** (largest gap between the two cumulative distributions). With thousands of rows almost any
  difference has a tiny p-value, so we read the KS statistic itself as an effect size.
* **PSI** (Population Stability Index) with quantile bins fixed on 2011-2012.
* **Categorical features:** category shares and their change in percentage points.
* **Domain classifier:** can a logistic regression tell two cycles apart from the features alone?
  Cross-validated ROC-AUC ≈ 0.5 means little covariate shift.

These statistics describe **what** changed. They do not tell us **why** (for example, sampling design,
population change or diagnosis practice); we do not claim causes without external evidence.

<!-- source: drift.psi drift.numeric_drift drift.categorical_drift drift.domain_classifier_auc drift.run_drift plotting.plot_feature_drift -->

```python
drift = run_drift(dataset, CONFIG, RESULTS_DIR)
display(drift["drift_numeric"].round(4))
display(drift["domain_classifier"].round(4))
categorical = drift["drift_categorical"]
categorical[categorical["compared"] == "2015-2016"].round(2)
```

```python
fig = plot_feature_drift(RESULTS_DIR, "paper")
save_figure(fig, FIGURES_DIR, "fig07_feature_drift_ks", ["png", "pdf"])
plt.show()
```

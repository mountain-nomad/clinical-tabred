## 9. Model evaluation

For each model, protocol and split we compute accuracy, precision, recall, F1, ROC-AUC, PR-AUC, the
Brier score and the confusion matrix, at threshold 0.5 and at the threshold that maximises F1 on
**validation** data. Inference time per 1,000 rows and training time are recorded too.

### Uncertainty
95% bootstrap confidence intervals (1,000 resamples) are computed for test ROC-AUC and PR-AUC.
For two models on the same test set we use a **paired** bootstrap (both models scored on identical
resamples). We only call a difference real when its CI excludes zero.

<!-- source: evaluation.bootstrap_indices evaluation.bootstrap_metric evaluation.percentile_ci evaluation.metric_ci evaluation.paired_difference evaluation.unpaired_difference evaluation.ci_table evaluation.paired_comparisons evaluation.gap_table benchmark.run_uncertainty -->

```python
uncertainty = run_uncertainty(tables["predictions"], best_model, CONFIG, RESULTS_DIR)
uncertainty["paired_comparisons"].round(4)
```

### Model comparison table: chronological test set (2015-2016), validation-tuned threshold

<!-- source: summaries.model_comparison_table -->

```python
model_comparison_table(RESULTS_DIR).round(3)
```

<!-- source: plotting.plot_auc_with_ci plotting.plot_roc_curves plotting.plot_confusion_matrix -->

```python
for name, plot in [("fig03_auc_ci", plot_auc_with_ci), ("fig05_roc_curves", plot_roc_curves),
                   ("fig06_confusion_matrix", plot_confusion_matrix)]:
    fig = plot(RESULTS_DIR, "paper")
    save_figure(fig, FIGURES_DIR, name, ["png", "pdf"])
    plt.show()
```

Confusion matrices of all models on the chronological test set:

```python
cm = pd.read_csv(RESULTS_DIR / "confusion_matrices.csv")
cm[(cm["protocol"] == "temporal") & (cm["split"] == "test")].round(3)
```

### Ablation study
The model selected on validation in protocol A is retrained with the same split, seed and hyperparameters,
changing one thing at a time. Scores are on the 2015-2016 test set.

<!-- source: experiment.ablation_variants experiment.run_ablation plotting.plot_ablation -->

```python
temporal_run = runs["outputs"][("temporal", CONFIG["seed"])]
ablation = run_ablation(best_model, temporal_run["params"][best_model], temporal_run["splits"],
                        CONFIG, base_settings(CONFIG))
ablation.to_csv(RESULTS_DIR / "ablation.csv", index=False)
display(ablation.round(4))
fig = plot_ablation(RESULTS_DIR, "paper")
save_figure(fig, FIGURES_DIR, "fig09_ablation", ["png", "pdf"])
plt.show()
```

## 8. Model training

Five models share the same preprocessing pipeline:
1. **Logistic Regression** (`class_weight="balanced"`)
2. **Random Forest** (`class_weight="balanced_subsample"`)
3. **XGBoost** (`tree_method="hist"`, `scale_pos_weight = n_neg / n_pos` from the training data)
4. **LightGBM** (`scale_pos_weight`, deterministic mode)
5. **MLP** (scikit-learn `MLPClassifier`), trained with an explicit epoch loop

Only about 13% of respondents are positive, so every model uses class-imbalance handling. scikit-learn's
MLP has no `class_weight`, so it receives "balanced" **sample weights** instead.

Each model has a small grid of at most 6 configurations. Every configuration is trained on the training
split and scored on validation; the one with the best validation **PR-AUC** (ROC-AUC breaks ties) is kept.
There is no refit on train + validation, which keeps the temporal setup clean.

<!-- source: models.MODEL_NAMES models.expand_grid models.make_estimator -->

### MLP training loop
`partial_fit` performs one pass (epoch) over the training data. After each epoch we record train and
validation log loss and accuracy. Training stops when validation loss has not improved for `patience`
epochs, and the weights of the best epoch are kept. These recorded curves are therefore the real
training curves of the evaluated MLP.

<!-- source: models.epoch_scores models.train_mlp models.fit_model models.predict_proba -->

### Tuning and evaluation for one protocol

<!-- source: experiment.base_settings experiment.fit_on_splits experiment.validation_scores experiment.tune_model experiment.evaluate_model experiment.run_protocol -->

<!-- source: benchmark.run_all_protocols benchmark.save_protocol_results benchmark.summarize_random_seeds benchmark.select_best_model benchmark.save_selected_models -->

Run all protocols (A tuned; A2 reuses A's hyperparameters; B tuned separately for each of the 5 seeds).
This is the slowest cell (a few minutes on a laptop CPU).

```python
start = time.perf_counter()
runs = run_all_protocols(dataset, CONFIG)
tables = save_protocol_results(runs, RESULTS_DIR)
summarize_random_seeds(tables["metrics"]).to_csv(RESULTS_DIR / "random_seeds_summary.csv", index=False)
best_model = select_best_model(tables["metrics"])
save_selected_models(runs, best_model, CONFIG, RESULTS_DIR)
print(f"Training finished in {time.perf_counter() - start:.0f} s; best model on validation: {best_model}")
```

Every configuration that was tried for protocol A (selected rows marked):

```python
tuning = tables["tuning"]
tuning[tuning["protocol"] == "temporal"].round(4)
```

### MLP accuracy and loss curves (protocol A)

<!-- source: plotting.plot_mlp_history -->

```python
fig = plot_mlp_history(RESULTS_DIR, "paper")
save_figure(fig, FIGURES_DIR, "fig08_mlp_curves", ["png", "pdf"])
plt.show()
```

### Cross-validation (strategy described in Section 7)

<!-- source: benchmark.run_cv_all summaries.cv_summary_table -->

```python
cv_folds = run_cv_all(runs, CONFIG, RESULTS_DIR)
display(pd.read_csv(RESULTS_DIR / "cv_metrics.csv").round(3))
cv_summary_table(RESULTS_DIR).round(3)
```

## 5. Exploratory data analysis

We first summarise each cycle: sample size, unweighted and survey-weighted prevalence, and a few means.
The figure functions read the saved CSV files, so every plot shows the stored results.

<!-- source: drift.weighted_mean drift.cycle_summary -->

```python
summary = cycle_summary(dataset)
summary.to_csv(RESULTS_DIR / "cycle_summary.csv", index=False)
display(summary.round(4))
dataset[FEATURES].describe().T.round(2)
```

Plot style and helpers (colours follow the course presentation theme; model colours were checked for
colour-blind separation):

<!-- source: plotting.MODEL_LABELS plotting.MODEL_COLORS plotting.CYCLE_COLORS plotting.INK plotting.apply_style plotting.figure_size plotting.bottom_legend plotting.models_in plotting.save_figure -->

<!-- source: plotting.plot_dataset_characteristics plotting.plot_prevalence -->

```python
apply_style("paper")
for name, plot in [("fig01_dataset_characteristics", plot_dataset_characteristics),
                   ("fig02_prevalence", plot_prevalence)]:
    fig = plot(RESULTS_DIR, "paper")
    save_figure(fig, FIGURES_DIR, name, ["png", "pdf"])
    plt.show()
```

Category shares by cycle (gender, race/Hispanic origin, education):

```python
for column in CONFIG["data"]["categorical_features"]:
    shares = pd.crosstab(dataset[column].astype("Float64").fillna(-1), dataset["cycle"],
                         normalize="columns") * 100
    shares.index = [CONFIG["data"]["category_labels"][column].get(int(c), "missing")
                    for c in shares.index]
    display(shares.round(1).rename_axis(column))
```

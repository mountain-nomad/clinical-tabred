## 11. Results and comparison

### Random vs in-period vs chronological evaluation
Mean ± std over the 5 random-split seeds (test set, tuned threshold):

```python
pd.read_csv(RESULTS_DIR / "random_seeds_summary.csv").round(3)
```

Differences between test sets with 95% bootstrap CIs (A2: the same fitted model on the in-period holdout
minus the 2015-2016 cycle; B minus A: random-split test minus chronological test):

```python
uncertainty["period_gaps"].round(4)
```

<!-- source: plotting.split_scores plotting.plot_protocol_comparison -->

```python
fig = plot_protocol_comparison(RESULTS_DIR, "paper")
save_figure(fig, FIGURES_DIR, "fig04_random_vs_temporal", ["png", "pdf"])
plt.show()
```

### Key findings (computed from the saved result files)

<!-- source: summaries.lookup summaries.key_number_specs summaries.key_numbers summaries.ci_text summaries.key_findings summaries.describe_interval summaries.interpretation -->

```python
display(Markdown(key_findings(RESULTS_DIR)))
display(Markdown(interpretation(RESULTS_DIR)))
```

### Reproducibility record
The metrics of this run are saved to `./notebook_outputs/metrics.csv`, together with package versions
and hardware details.

<!-- source: benchmark.package_versions benchmark.hardware_info benchmark.write_run_metadata -->

```python
metadata = write_run_metadata(CONFIG, RESULTS_DIR, {"source": "standalone notebook", "run_mode": RUN_MODE})
print(json.dumps(metadata, indent=2))
print(sorted(p.name for p in RESULTS_DIR.glob("*.csv")))
```

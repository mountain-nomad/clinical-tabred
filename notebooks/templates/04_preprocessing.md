## 4. Data preprocessing

Steps for each cycle:
1. **Merge** DEMO, BMX and DIQ on `SEQN`, starting from DEMO (left join), so nobody is lost silently.
2. **Build the label** from `DIQ010` (1 → 1, 2 → 0, everything else excluded).
3. **Clean codes:** categorical codes become pandas `Int64`; refused/don't-know codes become missing.
4. **Filter** eligible respondents and record every exclusion in a data-flow (attrition) table.
5. **Sanity check:** if the positive rate of a cycle falls outside 5-25%, stop, because the label mapping is probably wrong.

<!-- source: prepare.merge_cycle prepare.map_target prepare.clean_special_codes prepare.flow_row prepare.filter_cycle prepare.check_prevalence prepare.build_dataset prepare.missingness_table prepare.save_processed prepare.subsample_per_cycle prepare.feature_matrix -->

```python
dataset, data_flow, missing_components = build_dataset(CONFIG, RAW_DIR)
save_processed(dataset, CONFIG, PROCESSED_DIR)
FEATURES = CONFIG["data"]["numeric_features"] + CONFIG["data"]["categorical_features"]
data_flow.to_csv(RESULTS_DIR / "data_flow.csv", index=False)
missing_components.to_csv(RESULTS_DIR / "missing_components.csv", index=False)
missingness_table(dataset, FEATURES).to_csv(RESULTS_DIR / "missingness.csv", index=False)
if RUN_MODE == "quick":
    dataset = subsample_per_cycle(dataset, CONFIG["quick"]["rows_per_cycle"], CONFIG["seed"])
print("Final dataset:", dataset.shape)
display(data_flow)
display(missing_components)
dataset.head()
```

Missing values per feature and cycle (in %). They are imputed inside the model pipeline (below), never here:

```python
missingness_table(dataset, FEATURES).round(2)
```

### The preprocessing pipeline
The same `ColumnTransformer` is used by all five models:
numeric features → median imputation → standardisation; categorical codes → most-frequent imputation →
one-hot encoding. It is **fitted on the training split only** and then applied unchanged to validation
and test data, so no statistics from the evaluation data leak into training.

<!-- source: models.make_preprocessor -->

```python
make_preprocessor(CONFIG["data"]["numeric_features"], CONFIG["data"]["categorical_features"])
```

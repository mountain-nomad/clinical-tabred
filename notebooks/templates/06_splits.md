## 6. Train/test splitting

| Protocol | Train | Validation | Test |
|---|---|---|---|
| A chronological | 2011-2012 | 2013-2014 | 2015-2016 |
| A2 in-period reference | 80% of 2011-2012 (stratified) | 2013-2014 | 20% of 2011-2012 **and** 2015-2016 |
| B random | 70% of all cycles | 15% | 15% (stratified, 5 seeds) |

Validation data is used for hyperparameter selection, the MLP's early stopping and the decision threshold.
Test data is used only once, for the final evaluation. We check that no respondent appears in two splits.
Note that random and chronological test sets contain different people, so a difference between A and B
cannot be attributed entirely to time; protocol A2 isolates the time effect better because it scores the
**same fitted model** on two test sets.

<!-- source: splits.temporal_split splits.in_period_split splits.random_split splits.check_no_overlap splits.describe_splits benchmark.make_all_splits -->

```python
all_splits = make_all_splits(dataset, CONFIG)
split_table = pd.concat([describe_splits(s, f"{p}_seed{seed}") for p, seed, s in all_splits])
split_table.round(4)
```

# Algorithm 1: Clinical TabReD benchmark for one dataset

1: Input: wide table D with period labels t; grids G_m; seed s
2: Output: test metrics with 95% CIs per model and split; random-vs-time gaps
3: S ← {default, sliding-window-0..2, random-0..2}        ▷ random-k = rows of window k, shuffled, same sizes
4: for each split (train, val, test) in S do
5:     for each model m in {LR, RF, XGBoost, LightGBM, MLP} do
6:         for each configuration θ in G_m do
7:             fit ColumnTransformer + m_θ on train only   ▷ MLP: epoch loop, early stopping on val loss
8:             score val PR-AUC (ROC-AUC breaks ties)
9:         θ* ← best configuration;  τ* ← threshold with max F1 on val
10:        score test once at τ = 0.5 and τ = τ*; store test predictions
11: default split: bootstrap CIs; paired bootstrap of the val-selected model vs the others
12: for k = 0, 1, 2: Δ_k = AUC(random-k) − AUC(sliding-window-k) with unpaired bootstrap CI
13:     same-period Δ_k: random-k scored only on window k's test period
14: drift: domain classifier per split; PSI / TVD per feature; documented events per period

## Grouped explanations

- **Lines 1–3** — Inputs and the TabReD split set: one default time split, three sliding windows and three size-matched random splits.
- **Lines 4–9** — Tuning: every configuration is trained on the training periods and selected on the validation period; preprocessing never sees validation or test data.
- **Line 10** — The test split is scored once, with the threshold fixed on validation.
- **Lines 11–13** — Uncertainty and the random-vs-time comparison, including the same-period version.
- **Line 14** — Drift measured and linked to documented clinical causes.

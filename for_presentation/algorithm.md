# Algorithm 1: Temporal benchmark for one protocol

1: Input: train, validation and test splits; grid G_m for each model m; seed s
2: Output: tuned models, test metrics with 95% CIs, ablation table
3: for each model m in {LR, RF, XGBoost, LightGBM, MLP} do
4:     for each configuration θ in G_m do
5:         fit ColumnTransformer on the training split only
6:         if m = MLP then
7:             for epoch e = 1 … 200: partial_fit; record train/val loss and accuracy
8:             keep weights of the epoch with lowest validation loss; stop after 10 epochs without improvement
9:         else fit m_θ on the training split
10:         score validation PR-AUC (ROC-AUC breaks ties)
11:     θ* ← best configuration of m
12:     τ* ← threshold that maximises F1 on the validation split
13:     evaluate m_θ* on the test split at τ = 0.5 and τ = τ*
14:     95% bootstrap CIs of ROC-AUC and PR-AUC (1000 resamples)
15: select the best model on validation; paired bootstrap against the others
16: ablation: retrain the best model with one change at a time (same split and seed)

## Grouped explanations

- **Lines 1–2** — Inputs and outputs: three time-ordered splits, small grids and a fixed seed.
- **Lines 3–5** — Loop over models and grid configurations; preprocessing is fitted on training data only to avoid leakage.
- **Lines 6–9** — The MLP is trained one epoch at a time with early stopping; other models are fitted in one call.
- **Lines 10–12** — Model and threshold selection use validation data only, never the test set.
- **Lines 13–16** — Single test evaluation with bootstrap CIs, paired comparison and ablation.

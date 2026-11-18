# Equations

Shown on slide 11; (6)–(8) are used on the results slides.

## (1) Logistic model
Unicode: `P(y = 1 | x) = σ(wᵀx + b) = 1 / (1 + e^−(wᵀx + b))`
LaTeX: `P(y=1\mid\mathbf{x})=\sigma(\mathbf{w}^\top\mathbf{x}+b)`
Symbols: w: weights, b: bias, σ: sigmoid

## (2) Precision, recall and F1
Unicode: `P = TP / (TP + FP),   R = TP / (TP + FN),   F1 = 2PR / (P + R)`
LaTeX: `P=\frac{TP}{TP+FP},\; R=\frac{TP}{TP+FN},\; F_1=\frac{2PR}{P+R}`
Symbols: TP/FP/FN at threshold τ* = argmax F1 on the validation split

## (3) ROC-AUC
Unicode: `AUC = P( s(x⁺) > s(x⁻) )`
LaTeX: `\mathrm{AUC}=\Pr\left(s(\mathbf{x}^{+})>s(\mathbf{x}^{-})\right)`
Symbols: s: predicted probability; x⁺, x⁻: a random positive and a random negative row. Independent of the threshold and of the prevalence.

## (4) Population Stability Index (numeric drift)
Unicode: `PSI = Σ_b (q_b − p_b) · ln(q_b / p_b)`
LaTeX: `\mathrm{PSI}=\sum_b (q_b-p_b)\ln\frac{q_b}{p_b}`
Symbols: p_b, q_b: share of rows in bin b (10 quantile bins of the earlier period) in the earlier and later period

## (5) Total variation distance (categorical drift)
Unicode: `TVD = ½ Σ_c | p_c − q_c |`
LaTeX: `\mathrm{TVD}=\tfrac12\sum_c\lvert p_c-q_c\rvert`
Symbols: p_c, q_c: share of category c (missing counted as a category) in the two periods

## (6) Bootstrap 95% CI
Unicode: `CI = [ θ*₍₀.₀₂₅₎ , θ*₍₀.₉₇₅₎ ] over B = 1000 resamples`
LaTeX: `\mathrm{CI}_{95\%}=[\hat\theta^*_{(0.025)},\hat\theta^*_{(0.975)}]`
Symbols: θ*: metric recomputed on each resample of the test rows (paired: both models on the same resamples)

## (7) Random-minus-time gap (TabReD comparison)
Unicode: `Δ_k = AUC(random-k) − AUC(sliding-window-k)`
LaTeX: `\Delta_k=\mathrm{AUC}_{\text{random-}k}-\mathrm{AUC}_{\text{window-}k}`
Symbols: k = 0, 1, 2 sliding windows; Δ > 0: a random split overestimates future performance. Same-period version: the random-split model is scored only on its rows from window k's test period.

## (8) Relative improvement over the MLP (TabReD Table 4)
Unicode: `I_m = (1/3) Σ_k (AUC_m,k − AUC_MLP,k) / AUC_MLP,k × 100%`
LaTeX: `I_m=\frac13\sum_k\frac{\mathrm{AUC}_{m,k}-\mathrm{AUC}_{\mathrm{MLP},k}}{\mathrm{AUC}_{\mathrm{MLP},k}}\cdot100\%`
Symbols: m: model; computed separately for the time and the random splits

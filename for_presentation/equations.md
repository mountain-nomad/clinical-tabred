# Equations

Slide 11 shows (1)–(6); (7) is referenced on the results slides. Same numbering as the report.

## (1) Logistic model

Unicode: `P(y = 1 | x) = σ(wᵀx + b) = 1 / (1 + e^−(wᵀx + b))`

LaTeX: `P(y=1\mid\mathbf{x})=\sigma(\mathbf{w}^\top\mathbf{x}+b)`

Symbols: w: weights, b: bias, σ: sigmoid

## (2) Precision and recall

Unicode: `P = TP / (TP + FP),   R = TP / (TP + FN)`

LaTeX: `P=\frac{TP}{TP+FP},\; R=\frac{TP}{TP+FN}`

Symbols: TP/FP/FN: true positives, false positives, false negatives at threshold τ

## (3) F1-score

Unicode: `F1 = 2PR / (P + R)`

LaTeX: `F_1=\frac{2PR}{P+R}`

Symbols: P: precision, R: recall

## (4) KS statistic

Unicode: `D = supₓ | F_ref(x) − F_cmp(x) |`

LaTeX: `D=\sup_x\lvert F_{ref}(x)-F_{cmp}(x)\rvert`

Symbols: F: empirical CDF of a feature in the reference (2011-12) and compared cycle

## (5) Population Stability Index

Unicode: `PSI = Σ_b (q_b − p_b) · ln(q_b / p_b)`

LaTeX: `\mathrm{PSI}=\sum_b (q_b-p_b)\ln\frac{q_b}{p_b}`

Symbols: p_b, q_b: share of rows in bin b (10 quantile bins fixed on 2011-12) in the reference and compared cycle

## (6) CV mean and std

Unicode: `m̄ = (1/K) Σₖ mₖ,   s = sqrt( Σₖ (mₖ − m̄)² / (K − 1) )`

LaTeX: `\bar m=\frac1K\sum_k m_k,\; s=\sqrt{\frac{\sum_k (m_k-\bar m)^2}{K-1}}`

Symbols: m_k: metric on fold k, K = 5 folds

## (7) Bootstrap 95% CI

Unicode: `CI = [ θ*₍₀.₀₂₅₎ , θ*₍₀.₉₇₅₎ ] over 1000 resamples`

LaTeX: `\mathrm{CI}_{95\%}=[\hat\theta^*_{(0.025)},\hat\theta^*_{(0.975)}]`

Symbols: θ*: metric recomputed on each bootstrap resample of the test set

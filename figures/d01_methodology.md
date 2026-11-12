# Proposed methodology

## Boxes
1. **1 Download NHANES XPT (validated)** (position x=0.0, y=0)
2. **2 Prepare merge, label, filter adults** (position x=1.85, y=0)
3. **3 Features 9 inputs, special codes** (position x=3.7, y=0)
4. **4 Split A / A2 / B** (position x=5.550000000000001, y=0)
5. **5 Cross- validation 5-fold** (position x=7.4, y=0)
6. **6 Train 5 models, small grids** (position x=9.25, y=0)
7. **7 Evaluate metrics, CIs, ablation** (position x=11.100000000000001, y=0)
8. **8 Drift KS, PSI, domain clf.** (position x=12.950000000000001, y=0)

## Arrows
- 1 Download NHANES XPT (validated) → 2 Prepare merge, label, filter adults
- 2 Prepare merge, label, filter adults → 3 Features 9 inputs, special codes
- 3 Features 9 inputs, special codes → 4 Split A / A2 / B
- 4 Split A / A2 / B → 5 Cross- validation 5-fold
- 5 Cross- validation 5-fold → 6 Train 5 models, small grids
- 6 Train 5 models, small grids → 7 Evaluate metrics, CIs, ablation
- 7 Evaluate metrics, CIs, ablation → 8 Drift KS, PSI, domain clf.

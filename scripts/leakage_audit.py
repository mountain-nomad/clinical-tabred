"""Leakage screen of a wide table: how well does each feature ALONE predict the label?

A feature that is a consequence or a restatement of the target (e.g. "takes insulin" for diabetes,
"NICU admission" for preterm birth) predicts the label almost perfectly on its own. The build
scripts already drop such variables by name (text filter and documented exclusion lists); this
screen checks the result numerically.

For every feature:
  auc          univariate ROC-AUC (numeric: the value itself; categorical: the category's positive
               rate, i.e. in-sample target encoding). Folded so that 0.5 = useless, 1.0 = perfect.
  missing_auc  ROC-AUC of the "value is missing" indicator (skip patterns can leak the label:
               questions asked only of people with the condition).
A feature is flagged when either value exceeds FLAG (0.80); flagged features are reviewed by hand.

Usage: python scripts/leakage_audit.py --dataset {nhanes,brfss,natality}
Output: results_<dataset>/leakage_audit.csv (sorted by auc)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_benchmark import DATASETS, load_dataset  # noqa: E402

FLAG = 0.80


def folded_auc(y, score) -> float:
    if len(np.unique(y)) < 2 or len(np.unique(score)) < 2:
        return 0.5
    auc = roc_auc_score(y, score)
    return max(auc, 1 - auc)


def leakage_audit(data: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    y = data["label"].astype(int)
    rows = []
    for f in numeric + categorical:
        x = data[f]
        seen = x.notna()
        if f in numeric:
            score = x[seen].astype(float)
        else:   # positive rate of the row's category
            score = y[seen].groupby(x[seen]).transform("mean")
        rows.append({"feature": f, "type": "numeric" if f in numeric else "categorical",
                     "auc": folded_auc(y[seen], score) if seen.sum() > 100 else np.nan,
                     "missing_share": 1 - seen.mean(),
                     "missing_auc": folded_auc(y, (~seen).astype(int))})
    out = pd.DataFrame(rows).sort_values("auc", ascending=False)
    out["flagged"] = (out["auc"] > FLAG) | (out["missing_auc"] > FLAG)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nhanes", choices=list(DATASETS))
    ds = ap.parse_args().dataset
    data, numeric, categorical = load_dataset(ds)
    audit = leakage_audit(data, numeric, categorical)
    audit.to_csv(ROOT / DATASETS[ds]["out"] / "leakage_audit.csv", index=False)
    print(audit.head(10).round(3).to_string(index=False))
    print(f"\n{int(audit['flagged'].sum())} feature(s) above {FLAG}")


if __name__ == "__main__":
    main()

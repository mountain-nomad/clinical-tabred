"""Describe how the data changes between survey cycles (temporal drift).

These are descriptive statistics only: they show WHAT changed, not WHY.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from clinical_tabred.models import make_preprocessor
from clinical_tabred.prepare import feature_matrix


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Survey-weighted mean (ignores rows where the value or weight is missing)."""
    ok = values.notna() & weights.notna() & (weights > 0)
    return float(np.average(values[ok].astype(float), weights=weights[ok]))


def cycle_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    """Sample size, prevalence (unweighted and survey-weighted) and key means per cycle.

    The weighted prevalence uses the interview weight WTINT2YR (the diabetes
    question is part of the interview). It is a descriptive check only; all
    models are trained without weights.
    """
    rows = []
    for cycle, part in dataset.groupby("cycle"):
        rows.append({
            "cycle": cycle, "n": len(part), "n_pos": int(part["label"].sum()),
            "prevalence": part["label"].mean(),
            "weighted_prevalence": weighted_mean(part["label"], part["WTINT2YR"]),
            "mean_age": part["RIDAGEYR"].mean(), "mean_bmi": part["BMXBMI"].mean(),
            "mean_waist": part["BMXWAIST"].mean(), "female_share": (part["RIAGENDR"] == 2).mean(),
            "any_missing_share": part.drop(columns=["WTINT2YR"]).isna().any(axis=1).mean(),
        })
    return pd.DataFrame(rows)


def psi(reference: np.ndarray, current: np.ndarray, n_bins: int) -> float:
    """Population Stability Index with quantile bins fixed on the reference cycle.

    PSI = sum over bins of (p_cur - p_ref) * ln(p_cur / p_ref). A common rule of
    thumb: < 0.1 small change, 0.1-0.25 moderate, > 0.25 large.
    """
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    p_ref = np.histogram(reference, edges)[0] / len(reference)
    p_cur = np.histogram(current, edges)[0] / len(current)
    p_ref, p_cur = np.clip(p_ref, 1e-6, None), np.clip(p_cur, 1e-6, None)  # avoid log(0)
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def numeric_drift(dataset: pd.DataFrame, features: list[str], reference: str,
                  n_bins: int) -> pd.DataFrame:
    """Summary statistics, KS statistic and PSI of each numeric feature vs the reference.

    The KS statistic is the largest gap between the two cumulative distributions
    (0 = identical, 1 = no overlap). With thousands of rows almost any difference
    gives a tiny p-value, so the statistic itself is read as an effect size.
    """
    ref = dataset[dataset["cycle"] == reference]
    rows = []
    for cycle in sorted(c for c in dataset["cycle"].unique() if c != reference):
        cur = dataset[dataset["cycle"] == cycle]
        for feature in features:
            a, b = ref[feature].dropna().to_numpy(), cur[feature].dropna().to_numpy()
            test = ks_2samp(a, b)
            rows.append({"feature": feature, "reference": reference, "compared": cycle,
                         "mean_ref": a.mean(), "mean_cmp": b.mean(),
                         "median_ref": np.median(a), "median_cmp": np.median(b),
                         "std_ref": a.std(), "std_cmp": b.std(),
                         "missing_pct_ref": ref[feature].isna().mean() * 100,
                         "missing_pct_cmp": cur[feature].isna().mean() * 100,
                         "ks_statistic": test.statistic, "ks_pvalue": test.pvalue,
                         "psi": psi(a, b, n_bins)})
    return pd.DataFrame(rows)


def categorical_drift(dataset: pd.DataFrame, features: list[str], reference: str,
                      labels: dict) -> pd.DataFrame:
    """Share of each category per cycle and its change (percentage points) vs the reference."""
    rows = []
    for feature in features:
        shares = (dataset[feature].astype("Float64").fillna(-1)  # -1 marks missing values
                  .groupby(dataset["cycle"]).value_counts(normalize=True)
                  .unstack(level=0, fill_value=0.0) * 100)
        for code, row in shares.iterrows():
            name = "missing" if code == -1 else labels.get(feature, {}).get(int(code), str(code))
            for cycle in [c for c in shares.columns if c != reference]:
                rows.append({"feature": feature, "code": int(code), "category": name,
                             "reference": reference, "compared": cycle,
                             "pct_ref": row[reference], "pct_cmp": row[cycle],
                             "diff_pp": row[cycle] - row[reference]})
    return pd.DataFrame(rows)


def domain_classifier_auc(dataset: pd.DataFrame, reference: str, compared: str,
                          config: dict) -> dict:
    """Can a model tell the two cycles apart from the features alone?

    A logistic regression is trained to predict "is this row from `compared`?".
    Cross-validated ROC-AUC near 0.5 means the feature distributions are very
    similar (little covariate shift); values near 1 mean strong shift.
    """
    data_cfg = config["data"]
    features = data_cfg["numeric_features"] + data_cfg["categorical_features"]
    both = dataset[dataset["cycle"].isin([reference, compared])]
    X, y = feature_matrix(both, features), (both["cycle"] == compared).astype(int)
    model = Pipeline([("prep", make_preprocessor(data_cfg["numeric_features"],
                                                 data_cfg["categorical_features"])),
                      ("model", LogisticRegression(max_iter=2000))])
    folds = StratifiedKFold(n_splits=config["cv"]["n_splits"], shuffle=True,
                            random_state=config["seed"])
    scores = cross_val_score(model, X, y, cv=folds, scoring="roc_auc")
    return {"reference": reference, "compared": compared, "roc_auc_mean": scores.mean(),
            "roc_auc_std": scores.std(), "n_rows": len(both)}


def run_drift(dataset: pd.DataFrame, config: dict, results_dir) -> dict[str, pd.DataFrame]:
    """Compute every drift table and save them as CSV files in `results_dir`."""
    data_cfg, reference = config["data"], config["drift"]["reference_cycle"]
    others = sorted(c for c in dataset["cycle"].unique() if c != reference)
    tables = {
        "cycle_summary": cycle_summary(dataset),
        "drift_numeric": numeric_drift(dataset, data_cfg["numeric_features"], reference,
                                       config["drift"]["psi_bins"]),
        "drift_categorical": categorical_drift(dataset, data_cfg["categorical_features"],
                                               reference, data_cfg["category_labels"]),
        "domain_classifier": pd.DataFrame([domain_classifier_auc(dataset, reference, c, config)
                                           for c in others]),
    }
    for name, table in tables.items():
        table.to_csv(f"{results_dir}/{name}.csv", index=False)
    return tables

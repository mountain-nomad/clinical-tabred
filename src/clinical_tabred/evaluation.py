"""Metrics, threshold selection and bootstrap confidence intervals."""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

logger = logging.getLogger("clinical_tabred")

METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]


def safe_ratio(numerator: float, denominator: float, name: str) -> float:
    """Divide, but return NaN (and log a warning) instead of crashing or writing 0."""
    if denominator == 0:
        logger.warning("%s is undefined (division by zero); stored as NaN", name)
        return float("nan")
    return numerator / denominator


def ranking_metrics(y: np.ndarray, prob: np.ndarray) -> dict:
    """ROC-AUC and PR-AUC (average precision). Both need both classes to be present."""
    if len(np.unique(y)) < 2:
        logger.warning("Only one class present: ROC-AUC and PR-AUC stored as NaN")
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    return {"roc_auc": roc_auc_score(y, prob), "pr_auc": average_precision_score(y, prob)}


def compute_metrics(y, prob, threshold: float) -> dict:
    """All metrics for one set of predictions.

    Probabilities >= threshold are called positive. Returns accuracy, precision,
    recall, F1, ROC-AUC, PR-AUC, Brier score and the confusion-matrix counts.
    """
    y, prob = np.asarray(y).astype(int), np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = safe_ratio(tp, tp + fp, "precision")
    recall = safe_ratio(tp, tp + fn, "recall")
    f1 = safe_ratio(2 * tp, 2 * tp + fp + fn, "f1")
    return {"accuracy": (tp + tn) / len(y), "precision": precision, "recall": recall,
            "f1": f1, **ranking_metrics(y, prob), "brier": brier_score_loss(y, prob),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "threshold": threshold, "n": len(y), "n_pos": int(y.sum())}


def best_f1_threshold(y_val, prob_val) -> float:
    """Return the probability threshold that maximises F1 on the validation set.

    The threshold is chosen on validation data only; using the test set here would
    make the reported test scores optimistic.
    """
    precision, recall, thresholds = precision_recall_curve(y_val, prob_val)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])  # the last P/R pair has no threshold


def bootstrap_indices(n: int, n_resamples: int, seed: int) -> np.ndarray:
    """Matrix of resampled row indices (one row per bootstrap resample)."""
    return np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))


def bootstrap_metric(y, prob, metric, indices: np.ndarray) -> np.ndarray:
    """Value of `metric(y, prob)` on every bootstrap resample (NaN if one class is missing)."""
    y, prob = np.asarray(y), np.asarray(prob)
    values = np.full(len(indices), np.nan)
    for i, idx in enumerate(indices):
        if y[idx].min() != y[idx].max():
            values[i] = metric(y[idx], prob[idx])
    return values


def percentile_ci(values: np.ndarray, confidence: float) -> tuple[float, float]:
    """Percentile confidence interval, e.g. the 2.5th and 97.5th percentiles for 95%."""
    alpha = (1 - confidence) / 2
    return float(np.nanquantile(values, alpha)), float(np.nanquantile(values, 1 - alpha))


def metric_ci(y, prob, n_resamples: int, confidence: float, seed: int) -> dict:
    """Point estimates and bootstrap CIs for ROC-AUC and PR-AUC on one test set."""
    indices = bootstrap_indices(len(y), n_resamples, seed)
    result = {}
    for name, metric in [("roc_auc", roc_auc_score), ("pr_auc", average_precision_score)]:
        low, high = percentile_ci(bootstrap_metric(y, prob, metric, indices), confidence)
        result.update({name: metric(y, prob), f"{name}_low": low, f"{name}_high": high})
    return result


def paired_difference(y, prob_a, prob_b, metric, n_resamples: int, confidence: float,
                      seed: int) -> dict:
    """Paired bootstrap of metric(A) - metric(B) on the SAME test set.

    Both models are scored on identical resamples, so the CI reflects only the
    difference between models. If the CI contains 0 the difference is inconclusive.
    """
    indices = bootstrap_indices(len(y), n_resamples, seed)
    diffs = (bootstrap_metric(y, prob_a, metric, indices)
             - bootstrap_metric(y, prob_b, metric, indices))
    low, high = percentile_ci(diffs, confidence)
    return {"diff": metric(y, prob_a) - metric(y, prob_b), "diff_low": low, "diff_high": high}


def unpaired_difference(y_a, prob_a, y_b, prob_b, metric, n_resamples: int,
                        confidence: float, seed: int) -> dict:
    """Bootstrap of metric(set A) - metric(set B) for two DIFFERENT test sets.

    Each set is resampled independently (used e.g. for in-period vs future test).
    """
    values_a = bootstrap_metric(y_a, prob_a, metric, bootstrap_indices(len(y_a), n_resamples, seed))
    values_b = bootstrap_metric(y_b, prob_b, metric,
                                bootstrap_indices(len(y_b), n_resamples, seed + 1))
    low, high = percentile_ci(values_a - values_b, confidence)
    return {"diff": metric(y_a, prob_a) - metric(y_b, prob_b), "diff_low": low, "diff_high": high}


def ci_table(predictions: pd.DataFrame, n_resamples: int, confidence: float,
             seed: int) -> pd.DataFrame:
    """Bootstrap CIs of ROC-AUC and PR-AUC for every (protocol, seed, model, split) group.

    `predictions` has columns protocol, seed, model, split, label, probability.
    """
    rows = []
    for keys, group in predictions.groupby(["protocol", "seed", "model", "split"], sort=False):
        y, prob = group["label"].to_numpy(), group["probability"].to_numpy()
        rows.append({**dict(zip(["protocol", "seed", "model", "split"], keys)),
                     **metric_ci(y, prob, n_resamples, confidence, seed),
                     "n": len(y), "n_pos": int(y.sum())})
    return pd.DataFrame(rows)


def paired_comparisons(test: pd.DataFrame, reference: str, n_resamples: int,
                       confidence: float, seed: int) -> pd.DataFrame:
    """Paired bootstrap: the reference model minus every other model on one test set."""
    wide = test.pivot(index="SEQN", columns="model", values="probability")
    y = test.drop_duplicates("SEQN").set_index("SEQN").loc[wide.index, "label"].to_numpy()
    rows = []
    for other in [m for m in wide.columns if m != reference]:
        for name, metric in [("roc_auc", roc_auc_score), ("pr_auc", average_precision_score)]:
            result = paired_difference(y, wide[reference].to_numpy(), wide[other].to_numpy(),
                                       metric, n_resamples, confidence, seed)
            conclusive = result["diff_low"] > 0 or result["diff_high"] < 0
            rows.append({"reference": reference, "other": other, "metric": name, **result,
                         "ci_excludes_zero": conclusive})
    return pd.DataFrame(rows)


def gap_table(first: pd.DataFrame, second: pd.DataFrame, label: str, n_resamples: int,
              confidence: float, seed: int) -> pd.DataFrame:
    """Per model: metric(first test set) - metric(second test set) with a bootstrap CI.

    Used for "in-period holdout minus future cycle" and "random minus temporal".
    """
    rows = []
    for model in first["model"].unique():
        a, b = first[first["model"] == model], second[second["model"] == model]
        for name, metric in [("roc_auc", roc_auc_score), ("pr_auc", average_precision_score)]:
            result = unpaired_difference(a["label"].to_numpy(), a["probability"].to_numpy(),
                                         b["label"].to_numpy(), b["probability"].to_numpy(),
                                         metric, n_resamples, confidence, seed)
            rows.append({"comparison": label, "model": model, "metric": name, **result,
                         "ci_excludes_zero": result["diff_low"] > 0 or result["diff_high"] < 0})
    return pd.DataFrame(rows)

"""Tuning on validation, evaluation of one split, and cross-validation."""

import json
import logging

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from clinical_tabred.evaluation import best_f1_threshold, compute_metrics
from clinical_tabred.models import MODEL_NAMES, expand_grid, fit_model, predict_proba
from clinical_tabred.prepare import feature_matrix

logger = logging.getLogger("clinical_tabred")


def fit_on_splits(name: str, params: dict, splits: dict, settings: dict) -> dict:
    """Fit one model on splits["train"], using splits["val"] for MLP early stopping."""
    features = settings["numeric"] + settings["categorical"]
    return fit_model(name, params, feature_matrix(splits["train"], features),
                     splits["train"]["label"].astype(int),
                     feature_matrix(splits["val"], features), splits["val"]["label"].astype(int),
                     settings)


def validation_scores(fitted: dict, val: pd.DataFrame, settings: dict) -> tuple:
    """Validation PR-AUC and ROC-AUC of a fitted model (used for model selection)."""
    prob, _ = predict_proba(fitted["pipeline"],
                            feature_matrix(val, settings["numeric"] + settings["categorical"]))
    y = val["label"].astype(int)
    return average_precision_score(y, prob), roc_auc_score(y, prob)


def tune_model(name: str, grid: dict, splits: dict, settings: dict, tag: dict):
    """Try every configuration in the grid and keep the best by validation PR-AUC.

    ROC-AUC breaks ties. Returns (best fitted model, best params, one row per config).
    """
    rows, best = [], None
    for params in expand_grid(grid):
        fitted = fit_on_splits(name, params, splits, settings)
        pr_auc, roc_auc = validation_scores(fitted, splits["val"], settings)
        rows.append({**tag, "model": name, "params": json.dumps(params), "val_pr_auc": pr_auc,
                     "val_roc_auc": roc_auc, "train_time_s": fitted["train_time_s"]})
        if best is None or (pr_auc, roc_auc) > best[0]:
            best = ((pr_auc, roc_auc), fitted, params)
    for row in rows:
        row["selected"] = row["params"] == json.dumps(best[2])
    return best[1], best[2], rows


def evaluate_model(name: str, fitted: dict, splits: dict, settings: dict, tag: dict):
    """Score a fitted model on every split at threshold 0.5 and at the tuned threshold.

    The tuned threshold maximises F1 on the validation split. Returns
    (metric rows, prediction table for the test splits, chosen threshold).
    """
    features = settings["numeric"] + settings["categorical"]
    probs, timings = {}, {}
    for split_name, part in splits.items():
        probs[split_name], timings[split_name] = predict_proba(
            fitted["pipeline"], feature_matrix(part, features))
    threshold = best_f1_threshold(splits["val"]["label"].astype(int), probs["val"])
    rows, predictions = [], []
    for split_name, part in splits.items():
        for rule, value in [("0.5", 0.5), ("val_f1", threshold)]:
            metrics = compute_metrics(part["label"].astype(int), probs[split_name], value)
            rows.append({**tag, "model": name, "split": split_name, "threshold_rule": rule,
                         **metrics, "train_time_s": fitted["train_time_s"],
                         "infer_ms_per_1k": timings[split_name]})
        if split_name.startswith("test"):
            predictions.append(pd.DataFrame({
                **tag, "model": name, "split": split_name, "SEQN": part["SEQN"].to_numpy(),
                "cycle": part["cycle"].to_numpy(), "label": part["label"].astype(int).to_numpy(),
                "probability": probs[split_name]}))
    return rows, pd.concat(predictions, ignore_index=True), threshold


def cross_validate(protocol: str, dev: pd.DataFrame, params: dict, best_epoch: dict,
                   config: dict, settings: dict) -> list[dict]:
    """Stratified K-fold CV on development data only (the test set is never touched).

    Each model uses the hyperparameters selected for this protocol. The MLP has no
    validation set inside a fold, so it trains for the number of epochs at which
    early stopping halted in the main run. Threshold-based metrics use 0.5.
    """
    folds = StratifiedKFold(n_splits=config["cv"]["n_splits"], shuffle=True,
                            random_state=config["seed"])
    features = settings["numeric"] + settings["categorical"]
    X, y = feature_matrix(dev, features), dev["label"].astype(int)
    rows = []
    for name in MODEL_NAMES:
        fold_settings = {**settings, "max_epochs": best_epoch[name] or settings["max_epochs"]}
        for fold, (train_idx, test_idx) in enumerate(folds.split(X, y), start=1):
            fitted = fit_model(name, params[name], X.iloc[train_idx], y.iloc[train_idx],
                               None, None, fold_settings)
            prob, _ = predict_proba(fitted["pipeline"], X.iloc[test_idx])
            metrics = compute_metrics(y.iloc[test_idx], prob, 0.5)
            rows.append({"protocol": protocol, "model": name, "fold": fold,
                         **{k: metrics[k] for k in ["accuracy", "precision", "recall", "f1",
                                                    "roc_auc", "pr_auc"]}})
    return rows

"""Run the experiments: tuning, evaluation per protocol, cross-validation and ablation."""

import json
import logging

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from clinical_tabred.evaluation import best_f1_threshold, compute_metrics
from clinical_tabred.models import MODEL_NAMES, expand_grid, fit_model, predict_proba
from clinical_tabred.prepare import feature_matrix

logger = logging.getLogger("clinical_tabred")


def base_settings(config: dict) -> dict:
    """Settings shared by every model fit (features, seed, n_jobs, MLP loop length)."""
    data_cfg = config["data"]
    return {"numeric": list(data_cfg["numeric_features"]),
            "categorical": list(data_cfg["categorical_features"]),
            "seed": config["seed"], "n_jobs": config["n_jobs"], "balanced": True,
            "max_epochs": config["mlp_training"]["max_epochs"],
            "patience": config["mlp_training"]["patience"]}


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


def run_protocol(protocol: str, seed: int, splits: dict, config: dict, settings: dict,
                 fixed_params: dict | None = None) -> dict:
    """Tune (unless `fixed_params` is given), fit and evaluate all five models.

    Returns a dict of lists: metrics, tuning, predictions (per model), history,
    plus the selected params and best MLP epoch for each model.
    """
    tag = {"protocol": protocol, "seed": seed}
    out = {"metrics": [], "tuning": [], "predictions": {}, "history": [], "params": {},
           "best_epoch": {}, "thresholds": {}}
    for name in MODEL_NAMES:
        if fixed_params is None:
            fitted, params, rows = tune_model(name, config["models"][name], splits, settings, tag)
            out["tuning"].extend(rows)
        else:
            params = fixed_params[name]
            fitted = fit_on_splits(name, params, splits, settings)
        metrics, predictions, threshold = evaluate_model(name, fitted, splits, settings, tag)
        out["metrics"].extend(metrics)
        out["predictions"][name] = predictions
        out["params"][name], out["thresholds"][name] = params, threshold
        out["best_epoch"][name] = fitted["best_epoch"]
        if fitted["history"] is not None:
            out["history"].append(fitted["history"].assign(**tag))
        logger.info("%s seed=%s %s params=%s threshold=%.3f", protocol, seed, name, params,
                    threshold)
    return out


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


def ablation_variants(config: dict, settings: dict) -> list[tuple[str, dict, bool]]:
    """(name, settings, use_tuned_threshold) for every ablation configuration."""
    variants = [("full model", settings, True)]
    for group, columns in config["data"]["feature_groups"].items():
        reduced = {**settings,
                   "numeric": [c for c in settings["numeric"] if c not in columns],
                   "categorical": [c for c in settings["categorical"] if c not in columns]}
        variants.append((f"without {group} features", reduced, True))
    variants.append(("no class-imbalance handling", {**settings, "balanced": False}, True))
    variants.append(("threshold 0.5 instead of tuned", settings, False))
    return variants


def run_ablation(name: str, params: dict, splits: dict, config: dict,
                 settings: dict) -> pd.DataFrame:
    """Retrain the selected model with one component changed at a time.

    Same split, seed and hyperparameters as the full model; scores are on the
    temporal test set and deltas are relative to the full model.
    """
    rows = []
    for variant, variant_settings, tuned in ablation_variants(config, settings):
        fitted = fit_on_splits(name, params, splits, variant_settings)
        metrics, _, _ = evaluate_model(name, fitted, splits, variant_settings, {})
        rule = "val_f1" if tuned else "0.5"
        test = next(m for m in metrics if m["split"] == "test" and m["threshold_rule"] == rule)
        rows.append({"model": name, "configuration": variant, "threshold": test["threshold"],
                     "roc_auc": test["roc_auc"], "pr_auc": test["pr_auc"], "f1": test["f1"],
                     "accuracy": test["accuracy"]})
    ablation = pd.DataFrame(rows)
    for metric in ["roc_auc", "pr_auc", "f1"]:
        ablation[f"delta_{metric}"] = ablation[metric] - ablation.loc[0, metric]
    return ablation

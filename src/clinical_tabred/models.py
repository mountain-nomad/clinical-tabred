"""Preprocessing pipeline, the five models, and the MLP training loop."""

import copy
import itertools
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

# XGBoost and LightGBM are imported lazily so the other models still work where they
# are not installed; using one of them without the package raises a clear error.
try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None

MODEL_NAMES = ["logreg", "random_forest", "xgboost", "lightgbm", "mlp"]


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Imputation + scaling for numbers, imputation + one-hot encoding for categories.

    The same preprocessor is used for every model. It is always fitted on the
    training split only: if it saw validation or test rows, their medians and
    category frequencies would leak information about data the model must not see.
    """
    numeric_steps = Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())])
    categorical_steps = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # Category codes (e.g. race 1..7) are labels, not quantities, so they are one-hot encoded.
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", numeric_steps, numeric),
                              ("cat", categorical_steps, categorical)])


def expand_grid(grid: dict) -> list[dict]:
    """Turn {"a": [1, 2], "b": [3]} into [{"a": 1, "b": 3}, {"a": 2, "b": 3}]."""
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*grid.values())]


def make_estimator(name: str, params: dict, y_train: pd.Series, seed: int, n_jobs: int,
                   balanced: bool = True):
    """Create one (unfitted) classifier.

    `balanced=True` turns on class-imbalance handling: only ~13% of respondents are
    positive, so without it the models would mostly learn to predict "no diabetes".
    """
    ratio = float((y_train == 0).sum() / (y_train == 1).sum()) if balanced else 1.0
    if name == "logreg":
        return LogisticRegression(C=params["C"], max_iter=2000,
                                  class_weight="balanced" if balanced else None)
    if name == "random_forest":
        return RandomForestClassifier(
            **params, random_state=seed, n_jobs=n_jobs,
            class_weight="balanced_subsample" if balanced else None)
    if name in ("xgboost", "lightgbm") and {"xgboost": XGBClassifier,
                                            "lightgbm": LGBMClassifier}[name] is None:
        raise ImportError(f"{name} is not installed: run `uv pip install -r requirements.txt`")
    if name == "xgboost":
        return XGBClassifier(**params, tree_method="hist", scale_pos_weight=ratio,
                             eval_metric="logloss", random_state=seed, n_jobs=n_jobs)
    if name == "lightgbm":
        return LGBMClassifier(**params, scale_pos_weight=ratio, random_state=seed,
                              n_jobs=n_jobs, deterministic=True, force_row_wise=True,
                              verbose=-1)
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
                             alpha=params["alpha"], learning_rate_init=1e-3, batch_size=128,
                             random_state=seed)
    raise ValueError(f"Unknown model name '{name}'. Choose one of {MODEL_NAMES}.")


def epoch_scores(model, X, y, weights) -> tuple[float, float]:
    """Log loss and accuracy (threshold 0.5) of the current MLP on one data split."""
    prob = model.predict_proba(X)[:, 1]
    loss = log_loss(y, prob, sample_weight=weights, labels=[0, 1])
    accuracy = float(np.mean((prob >= 0.5) == y))
    return loss, accuracy


def train_mlp(model: MLPClassifier, X_train, y_train, X_val, y_val, max_epochs: int,
              patience: int, balanced: bool) -> tuple[MLPClassifier, pd.DataFrame, int]:
    """Train the MLP one epoch at a time with early stopping on validation loss.

    Each `partial_fit` call is one pass (epoch) over the training data. sklearn's
    MLP has no `class_weight`, so imbalance is handled with "balanced" sample
    weights; the same weighting is used for the losses we record. If no validation
    data is given, the loop simply runs `max_epochs` epochs.

    Returns (model with the best epoch's weights, per-epoch history, best epoch).
    """
    y_train, y_val = np.asarray(y_train), (None if y_val is None else np.asarray(y_val))
    w_train = compute_sample_weight("balanced", y_train) if balanced else None
    w_val = compute_sample_weight("balanced", y_val) if balanced and y_val is not None else None
    best_model, best_loss, best_epoch, history = model, np.inf, 0, []
    for epoch in range(1, max_epochs + 1):
        model.partial_fit(X_train, y_train, classes=[0, 1], sample_weight=w_train)
        train_loss, train_acc = epoch_scores(model, X_train, y_train, w_train)
        row = {"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc}
        if X_val is None:
            history.append(row)
            best_model, best_epoch = model, epoch
            continue
        row["val_loss"], row["val_accuracy"] = epoch_scores(model, X_val, y_val, w_val)
        history.append(row)
        if row["val_loss"] < best_loss - 1e-4:  # a real improvement: remember these weights
            best_loss, best_epoch, best_model = row["val_loss"], epoch, copy.deepcopy(model)
        elif epoch - best_epoch >= patience:  # no improvement for `patience` epochs: stop
            break
    history = pd.DataFrame(history)
    history["best_epoch"] = best_epoch
    return best_model, history, best_epoch


def fit_model(name: str, params: dict, X_train: pd.DataFrame, y_train: pd.Series,
              X_val, y_val, settings: dict) -> dict:
    """Fit preprocessing + model on the training split and return a dict with the fitted
    pipeline, the training time in seconds and (MLP only) the training history.

    `settings` holds numeric/categorical feature lists, seed, n_jobs, balanced,
    max_epochs and patience.
    """
    start = time.perf_counter()
    preprocessor = make_preprocessor(settings["numeric"], settings["categorical"])
    estimator = make_estimator(name, params, y_train, settings["seed"], settings["n_jobs"],
                               settings["balanced"])
    history, best_epoch = None, None
    if name == "mlp":
        Xt_train = preprocessor.fit_transform(X_train)
        Xt_val = None if X_val is None else preprocessor.transform(X_val)
        estimator, history, best_epoch = train_mlp(
            estimator, Xt_train, y_train, Xt_val, y_val, settings["max_epochs"],
            settings["patience"], settings["balanced"])
        pipeline = Pipeline([("prep", preprocessor), ("model", estimator)])
    else:
        pipeline = Pipeline([("prep", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
    return {"pipeline": pipeline, "train_time_s": time.perf_counter() - start,
            "history": history, "best_epoch": best_epoch}


def predict_proba(pipeline: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Positive-class probabilities and inference time in milliseconds per 1,000 rows."""
    start = time.perf_counter()
    prob = pipeline.predict_proba(X)[:, 1]
    ms_per_1k = (time.perf_counter() - start) * 1000 / max(len(X), 1) * 1000
    return prob, ms_per_1k

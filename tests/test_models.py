import numpy as np
import pytest
from conftest import SPEC
from run_benchmark import make_splits
from sklearn.metrics import roc_auc_score

from clinical_tabred.evaluation import best_f1_threshold, compute_metrics, paired_difference
from clinical_tabred.experiment import evaluate_model, tune_model
from clinical_tabred.models import MODEL_NAMES

GRIDS = {"logreg": {"C": [0.1, 1.0]}, "random_forest": {"n_estimators": [20], "max_depth": [4]},
         "xgboost": {"n_estimators": [20], "max_depth": [3]}, "lightgbm": {"n_estimators": [20], "num_leaves": [7]},
         "mlp": {"hidden_layer_sizes": [[8]], "alpha": [0.01]}}


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_tunes_and_evaluates(synthetic, settings, name):
    parts = make_splits(synthetic, 42, SPEC)["default"]
    fitted, _params, rows = tune_model(name, GRIDS[name], parts, settings, {"protocol": "default"})
    assert sum(r["selected"] for r in rows) == 1
    metrics, preds, threshold = evaluate_model(name, fitted, parts, settings, {"protocol": "default"})
    assert {m["split"] for m in metrics} == {"train", "val", "test"}
    assert preds["probability"].between(0, 1).all() and len(preds) == len(parts["test"])
    test = next(m for m in metrics if m["split"] == "test" and m["threshold_rule"] == "val_f1")
    assert test["roc_auc"] > 0.6          # the synthetic signal is learnable
    assert 0 < threshold < 1


def test_metrics_on_known_example():
    y, p = np.array([0, 0, 1, 1]), np.array([0.1, 0.6, 0.4, 0.9])
    m = compute_metrics(y, p, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == m["recall"] == m["f1"] == 0.5
    assert best_f1_threshold(y, p) == pytest.approx(0.4)


def test_paired_difference_of_identical_models_is_zero():
    rng = np.random.default_rng(1)
    y, p = rng.integers(0, 2, 200), rng.random(200)
    r = paired_difference(y, p, p, roc_auc_score, 100, 0.95, 0)
    assert r["diff"] == r["diff_low"] == r["diff_high"] == 0

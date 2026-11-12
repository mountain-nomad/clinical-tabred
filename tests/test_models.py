"""Tests for the preprocessing pipeline and metrics."""

import numpy as np

from clinical_tabred.evaluation import best_f1_threshold, compute_metrics
from clinical_tabred.experiment import base_settings
from clinical_tabred.models import fit_model
from clinical_tabred.prepare import build_dataset, feature_matrix
from clinical_tabred.splits import temporal_split


def test_preprocessing_is_fitted_on_training_data_only(test_config):
    dataset = build_dataset(test_config, test_config["paths"]["raw_dir"])[0]
    splits = temporal_split(dataset, test_config["protocols"]["temporal"])
    settings = base_settings(test_config)
    features = settings["numeric"] + settings["categorical"]
    X_train = feature_matrix(splits["train"], features)
    fitted = fit_model("logreg", {"C": 1.0}, X_train, splits["train"]["label"].astype(int),
                       None, None, settings)
    imputer = fitted["pipeline"].named_steps["prep"].named_transformers_["num"]["impute"]
    expected = X_train[settings["numeric"]].median().to_numpy()
    np.testing.assert_allclose(imputer.statistics_, expected)
    all_rows = feature_matrix(dataset, features)[settings["numeric"]].median().to_numpy()
    assert not np.allclose(imputer.statistics_, all_rows)  # not fitted on all cycles


def test_mlp_training_loop_records_history(test_config):
    dataset = build_dataset(test_config, test_config["paths"]["raw_dir"])[0]
    splits = temporal_split(dataset, test_config["protocols"]["temporal"])
    settings = {**base_settings(test_config), "max_epochs": 8, "patience": 3}
    features = settings["numeric"] + settings["categorical"]
    fitted = fit_model("mlp", {"hidden_layer_sizes": [8], "alpha": 0.01},
                       feature_matrix(splits["train"], features),
                       splits["train"]["label"].astype(int),
                       feature_matrix(splits["val"], features), splits["val"]["label"].astype(int),
                       settings)
    history = fitted["history"]
    assert {"epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy"} <= set(history)
    assert 1 <= fitted["best_epoch"] <= len(history) <= 8


def test_metrics_with_no_positive_predictions_are_nan_not_zero():
    metrics = compute_metrics([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4], threshold=0.9)
    assert np.isnan(metrics["precision"])
    assert metrics["recall"] == 0.0
    assert metrics["tp"] == 0 and metrics["fn"] == 2


def test_best_f1_threshold_separates_perfect_scores():
    threshold = best_f1_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert 0.2 < threshold <= 0.8

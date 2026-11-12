"""End-to-end smoke test of the quick pipeline on synthetic data (no network)."""

import pandas as pd
import yaml

from clinical_tabred.run import load_config, stage_prepare, stage_train

EXPECTED_METRIC_COLUMNS = {"protocol", "seed", "model", "split", "threshold_rule", "accuracy",
                           "precision", "recall", "f1", "roc_auc", "pr_auc", "brier", "tn", "fp",
                           "fn", "tp", "threshold", "n", "n_pos", "train_time_s",
                           "infer_ms_per_1k"}


def test_quick_run_produces_metrics(test_config, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(test_config))
    config = load_config(config_path, quick=True)
    stage_prepare(config)
    stage_train(config, quick=True)
    results = tmp_path / "results_quick"
    metrics = pd.read_csv(results / "metrics.csv")
    assert EXPECTED_METRIC_COLUMNS <= set(metrics.columns)
    assert set(metrics["model"]) == {"logreg", "random_forest", "xgboost", "lightgbm", "mlp"}
    assert set(metrics["protocol"]) == {"temporal", "in_period", "random"}
    for name in ["cv_folds", "cv_metrics", "ablation", "metrics_ci", "tuning", "mlp_history",
                 "confusion_matrices", "paired_comparisons", "period_gaps"]:
        assert (results / f"{name}.csv").exists(), name
    folds = pd.read_csv(results / "cv_folds.csv")
    assert set(folds["fold"]) == {1, 2, 3, 4, 5}

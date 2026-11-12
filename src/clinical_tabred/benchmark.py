"""The training stage: run every protocol and save all result tables to disk."""

import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd

from clinical_tabred.evaluation import ci_table, gap_table, paired_comparisons
from clinical_tabred.experiment import base_settings, cross_validate, run_protocol
from clinical_tabred.splits import (
    check_no_overlap,
    describe_splits,
    in_period_split,
    random_split,
    temporal_split,
)

logger = logging.getLogger("clinical_tabred")


def quick_config(config: dict) -> dict:
    """A cheaper copy of the config for smoke tests: one value per grid, few epochs."""
    quick = json.loads(json.dumps(config))  # deep copy
    quick["models"] = {name: {k: v[:1] for k, v in grid.items()}
                       for name, grid in config["models"].items()}
    quick["bootstrap"]["n_resamples"] = config["quick"]["n_resamples"]
    quick["protocols"]["random"]["seeds"] = config["quick"]["random_seeds"]
    quick["mlp_training"]["max_epochs"] = config["quick"]["max_epochs"]
    return quick


def make_all_splits(dataset: pd.DataFrame, config: dict) -> list[tuple[str, int, dict]]:
    """Build (protocol, seed, splits) for A, A2 and every seed of B; check for overlap."""
    protocols, seed = config["protocols"], config["seed"]
    runs = [("temporal", seed, temporal_split(dataset, protocols["temporal"])),
            ("in_period", seed, in_period_split(dataset, protocols["temporal"],
                                                protocols["in_period"]["holdout_fraction"], seed))]
    runs += [("random", s, random_split(dataset, protocols["random"], s))
             for s in protocols["random"]["seeds"]]
    for _, _, splits in runs:
        check_no_overlap(splits)
    return runs


def run_all_protocols(dataset: pd.DataFrame, config: dict) -> dict:
    """Run protocol A (tuned), A2 (A's hyperparameters) and B (tuned per seed).

    Returns {(protocol, seed): run output} plus the split description table.
    """
    settings = base_settings(config)
    runs = make_all_splits(dataset, config)
    outputs = {}
    split_info = pd.concat([describe_splits(s, f"{p}_seed{seed}") for p, seed, s in runs])
    for protocol, seed, splits in runs:
        fixed = outputs[("temporal", config["seed"])]["params"] if protocol == "in_period" else None
        outputs[(protocol, seed)] = run_protocol(protocol, seed, splits, config,
                                                 {**settings, "seed": seed}, fixed)
        outputs[(protocol, seed)]["splits"] = splits
    return {"outputs": outputs, "split_info": split_info}


def save_protocol_results(runs: dict, results_dir: Path) -> dict[str, pd.DataFrame]:
    """Write metrics, tuning, predictions, MLP history and confusion matrices to CSV."""
    results_dir = Path(results_dir)
    (results_dir / "predictions").mkdir(parents=True, exist_ok=True)
    outputs = runs["outputs"].values()
    metrics = pd.DataFrame([row for out in outputs for row in out["metrics"]])
    tables = {
        "metrics": metrics,
        "tuning": pd.DataFrame([row for out in outputs for row in out["tuning"]]),
        "mlp_history": pd.concat([h for out in outputs for h in out["history"]]),
        "confusion_matrices": metrics[["protocol", "seed", "model", "split", "threshold_rule",
                                       "threshold", "tn", "fp", "fn", "tp"]],
        "splits": runs["split_info"],
    }
    for name, table in tables.items():
        table.to_csv(results_dir / f"{name}.csv", index=False)
    predictions = pd.concat([p for out in outputs for p in out["predictions"].values()])
    for (protocol, model), part in predictions.groupby(["protocol", "model"]):
        part.to_csv(results_dir / "predictions" / f"{protocol}_{model}.csv", index=False)
    tables["predictions"] = predictions
    return tables


def summarize_random_seeds(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean and std over the random-split seeds (test split, tuned threshold)."""
    test = metrics[(metrics["protocol"] == "random") & (metrics["split"] == "test")
                   & (metrics["threshold_rule"] == "val_f1")]
    columns = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]
    summary = test.groupby("model")[columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index().assign(n_seeds=test["seed"].nunique())


def select_best_model(metrics: pd.DataFrame) -> str:
    """Best protocol-A model by validation PR-AUC (ROC-AUC as tiebreaker). Never uses test."""
    val = metrics[(metrics["protocol"] == "temporal") & (metrics["split"] == "val")
                  & (metrics["threshold_rule"] == "val_f1")]
    return val.sort_values(["pr_auc", "roc_auc"], ascending=False).iloc[0]["model"]


def save_selected_models(runs: dict, best: str, config: dict, results_dir: Path) -> dict:
    """Save the best model name, chosen hyperparameters and thresholds to JSON."""
    outputs = runs["outputs"].items()
    selected = {"best_model": best, "seed": config["seed"],
                "params": {f"{p}_seed{s}": out["params"] for (p, s), out in outputs},
                "thresholds": {f"{p}_seed{s}": out["thresholds"] for (p, s), out in outputs}}
    (Path(results_dir) / "selected_models.json").write_text(json.dumps(selected, indent=2))
    return selected


def run_cv_all(runs: dict, config: dict, results_dir: Path) -> pd.DataFrame:
    """5-fold CV on protocol A's training cycle and on protocol B's train+validation."""
    settings = base_settings(config)
    rows = []
    for protocol, parts in [("temporal", ["train"]), ("random", ["train", "val"])]:
        out = runs["outputs"][(protocol, config["seed"])]
        dev = pd.concat([out["splits"][p] for p in parts])
        rows += cross_validate(protocol, dev, out["params"], out["best_epoch"], config, settings)
    folds = pd.DataFrame(rows)
    folds.to_csv(Path(results_dir) / "cv_folds.csv", index=False)
    summary = folds.groupby(["protocol", "model"]).agg(["mean", "std"]).drop(columns="fold")
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(Path(results_dir) / "cv_metrics.csv", index=False)
    return folds


def run_uncertainty(predictions: pd.DataFrame, best_model: str, config: dict,
                    results_dir: Path) -> dict[str, pd.DataFrame]:
    """Bootstrap CIs, paired model comparisons and period-gap CIs; saved to CSV."""
    boot, seed = config["bootstrap"], config["seed"]
    n, conf = boot["n_resamples"], boot["confidence"]
    main = predictions[predictions["seed"] == seed]
    temporal_test = main[(main["protocol"] == "temporal") & (main["split"] == "test")]
    in_period = main[main["protocol"] == "in_period"]
    random_test = main[(main["protocol"] == "random") & (main["split"] == "test")]
    tables = {
        "metrics_ci": ci_table(main, n, conf, seed),
        "paired_comparisons": paired_comparisons(temporal_test, best_model, n, conf, seed),
        "period_gaps": pd.concat([
            gap_table(in_period[in_period["split"] == "test_in_period"],
                      in_period[in_period["split"] == "test"],
                      "A2: in-period holdout minus 2015-2016", n, conf, seed),
            gap_table(random_test, temporal_test, "B random test minus A temporal test",
                      n, conf, seed)]),
    }
    for name, table in tables.items():
        table.to_csv(Path(results_dir) / f"{name}.csv", index=False)
    return tables


def package_versions(packages: list[str]) -> dict[str, str]:
    """Installed version of each package (or 'not installed')."""
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def hardware_info() -> dict:
    """CPU model, core count, RAM and operating system of this machine."""
    info = {"os": platform.platform(), "machine": platform.machine(),
            "cpu_count": os.cpu_count(), "cpu_model": platform.processor()}
    try:
        if sys.platform == "darwin":
            info["cpu_model"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                               capture_output=True, text=True, check=False).stdout.strip()
            ram = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, check=False,
                                     text=True).stdout)
        else:
            ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        info["ram_gb"] = round(ram / 1024**3, 1)
    except (OSError, ValueError) as error:
        logger.warning("Could not read hardware details: %s", error)
    return info


def write_run_metadata(config: dict, results_dir: Path, extra: dict) -> dict:
    """Save Python/package versions, seed, timestamp and hardware to run_metadata.json.

    Stage runtimes from earlier runs are kept, so running stages one by one still
    produces a complete record.
    """
    path = Path(results_dir, "run_metadata.json")
    previous = json.loads(path.read_text()) if path.exists() else {}
    runtimes = {**previous.get("stage_runtimes_s", {}), **extra.get("stage_runtimes_s", {})}
    packages = ["pandas", "numpy", "scikit-learn", "xgboost", "lightgbm", "scipy",
                "matplotlib", "pyyaml", "requests"]
    meta = {"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(), "seed": config["seed"],
            "n_jobs": config["n_jobs"], "packages": package_versions(packages),
            "hardware": hardware_info(), **extra, "stage_runtimes_s": runtimes}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2))
    return meta

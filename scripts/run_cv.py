"""5-fold stratified cross-validation for the wide benchmark (course requirement).

Same rule as the original project (DECISIONS.md #17, `clinical_tabred.experiment.cross_validate`):
CV runs on the development data of the default time split only (train + validation periods), so
the test periods are never touched. Each model uses the hyperparameters selected on validation in
the main run (results_*/selected_models.json); the MLP trains for its best epoch from that run;
threshold-based metrics use 0.5 (no validation set exists inside a fold).

Usage: python scripts/run_cv.py --dataset {nhanes,brfss,natality}
Output: <results dir>/cv_folds.csv (one row per model x fold) and cv_summary.csv (mean, std)
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from clinical_tabred.experiment import cross_validate  # noqa: E402
from run_benchmark import DATASETS, load_config, load_dataset, make_splits, model_settings  # noqa: E402

logger = logging.getLogger("clinical_tabred")
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


def run_cv(dataset: str, n_jobs: int = -1) -> pd.DataFrame:
    """5-fold CV of all models on the default split's development periods; writes cv_*.csv."""
    config = load_config()
    spec = DATASETS[dataset]
    out_dir = ROOT / spec["out"]
    selected = json.loads((out_dir / "selected_models.json").read_text())["default"]

    data, numeric, categorical = load_dataset(dataset)
    split = make_splits(data, config["seed"], spec)["default"]
    dev = pd.concat([split["train"], split["val"]], ignore_index=True)
    settings = model_settings(config, numeric, categorical, n_jobs)
    params = {m: v["params"] for m, v in selected.items()}
    best_epoch = {m: v["best_epoch"] for m, v in selected.items()}
    logger.info("%s: 5-fold CV on %d development rows (periods %s)", dataset, len(dev),
                ",".join(sorted(dev["cycle"].unique())))

    rows = cross_validate("default", dev, params, best_epoch, config, settings)
    folds = pd.DataFrame(rows).assign(dataset=dataset)
    folds.to_csv(out_dir / "cv_folds.csv", index=False)
    summary = folds.groupby("model")[METRICS].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index().assign(dataset=dataset)
    summary.to_csv(out_dir / "cv_summary.csv", index=False)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nhanes", choices=list(DATASETS))
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    print(run_cv(args.dataset, args.n_jobs).round(4).to_string(index=False))


if __name__ == "__main__":
    main()

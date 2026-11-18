"""TabReD-style benchmark on the wide clinical tables (NHANES, BRFSS, natality).

Splits (same design as TabReD's preprocessing scripts), per dataset (see DATASETS):
  nhanes   default train 2005-2012 | val 2013-2014 | test 2015-2018   (7 two-year cycles)
  brfss    default train 2017-2020 | val 2021      | test 2022-2023   (7 years)
  natality default train 2016-2021 | val 2022      | test 2023-2024   (9 years)
  sliding-window-k   train 3 periods | val next period | test next period   (3 windows)
  random-k           the rows of sliding-window-k, shuffled and re-split into the SAME sizes

Every model is tuned on the validation split (PR-AUC, ROC-AUC tiebreak), the threshold is
chosen on validation (max F1), and the test split is scored once. Preprocessing (imputation,
scaling, one-hot) is fitted on the training split only.

Usage (from the repository root, inside the .venv):
  python scripts/run_benchmark.py                       # NHANES, all five models, all splits
  python scripts/run_benchmark.py --dataset brfss       # BRFSS 2017-2023 (results_brfss/)
  python scripts/run_benchmark.py --dataset natality    # natality 2016-2024 (results_natality/)
  python scripts/run_benchmark.py --models logreg mlp   # a subset
  python scripts/run_benchmark.py --quick               # smoke test (small sample)
Outputs go to results_<dataset>/ (suffix _quick with --quick).
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinical_tabred.evaluation import metric_ci, paired_comparisons, unpaired_difference  # noqa: E402
from clinical_tabred.experiment import evaluate_model, tune_model  # noqa: E402
from clinical_tabred.models import MODEL_NAMES  # noqa: E402

CYCLES = ["2005-2006", "2007-2008", "2009-2010", "2011-2012", "2013-2014", "2015-2016", "2017-2018"]
# periods: chronological period labels (column "cycle" after loading); default: (n_train, n_val) periods,
# the remaining periods are the test set; windows: start index of each 3+1+1 sliding window.
DATASETS = {
    "nhanes": {"table": "nhanes_wide_2005_2018.csv", "columns": "nhanes_wide_columns.csv",
               "periods": CYCLES, "default": (4, 1), "windows": [0, 1, 2], "out": "results_nhanes"},
    "brfss": {"table": "brfss_wide_2017_2023.csv.gz", "columns": "brfss_wide_columns.csv",
              "periods": [str(y) for y in range(2017, 2024)], "default": (4, 1), "windows": [0, 1, 2],
              "out": "results_brfss"},
    # 9 years: default is ~78/22 (course: 80/20); windows test 2020 (COVID), 2022 and 2024.
    "natality": {"table": "natality_wide_2016_2024.csv.gz", "columns": "natality_wide_columns.csv",
                 "periods": [str(y) for y in range(2016, 2025)], "default": (6, 1), "windows": [0, 2, 4],
                 "out": "results_natality"},
}
# Laboratory and examination values are quantities.
NUMERIC_COMPONENTS = {"BIOPRO", "CBC", "BMX", "TCHOL", "HDL", "TRIGLY", "ALB_CR", "BPX"}
CATEGORICAL_EXCEPTIONS = {"BPXPTY", "BPXPULS"}                    # pulse type / regularity codes
NUMERIC_DEMO = {"RIDAGEYR", "INDFMPIR", "DMDHHSIZ", "DMDFMSIZ"}
MAX_CATEGORY_LEVELS = 12   # questionnaire answers with <= 12 distinct codes are categories

logger = logging.getLogger("clinical_tabred")


def feature_types(columns: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split features into numeric and categorical (codes are labels, not quantities)."""
    numeric, categorical = [], []
    for row in columns.itertuples():
        f = row.feature
        if f in CATEGORICAL_EXCEPTIONS:
            categorical.append(f)
        elif row.component in NUMERIC_COMPONENTS or f in NUMERIC_DEMO:
            numeric.append(f)
        elif row.n_unique <= MAX_CATEGORY_LEVELS:
            categorical.append(f)
        else:
            numeric.append(f)   # e.g. weights in pounds, ages, hours worked, drinks per day
    return numeric, categorical


def load_dataset(name: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Read a wide table; return (data with SEQN/cycle/label columns, numeric, categorical)."""
    spec = DATASETS[name]
    data = pd.read_csv(ROOT / "data" / "processed" / spec["table"], dtype={"period": str})
    columns = pd.read_csv(ROOT / "data" / "processed" / spec["columns"])
    columns = columns[columns["feature"].isin(data.columns)]
    if name != "nhanes":   # shared pipeline code identifies rows by SEQN and periods by cycle
        data = data.rename(columns={"row_id": "SEQN"}).assign(cycle=data["period"].astype(str))
    if "type" in columns:  # BRFSS / natality: feature types decided in the build scripts
        numeric = columns.loc[columns["type"] == "numeric", "feature"].tolist()
        categorical = columns.loc[columns["type"] == "categorical", "feature"].tolist()
    else:
        numeric, categorical = feature_types(columns)
    return data, numeric, categorical


def make_splits(data: pd.DataFrame, seed: int, spec: dict = DATASETS["nhanes"]) -> dict[str, dict[str, pd.DataFrame]]:
    periods, (n_tr, n_va) = spec["periods"], spec["default"]
    by = {c: data[data["cycle"] == c] for c in periods}
    cat = lambda cs: pd.concat([by[c] for c in cs], ignore_index=True)  # noqa: E731
    splits = {"default": {"train": cat(periods[:n_tr]), "val": cat(periods[n_tr:n_tr + n_va]),
                          "test": cat(periods[n_tr + n_va:])}}
    for k, s in enumerate(spec["windows"]):
        window = {"train": cat(periods[s:s + 3]), "val": cat([periods[s + 3]]), "test": cat([periods[s + 4]])}
        splits[f"sliding-window-{k}"] = window
        # TabReD: random splits shuffle all rows of the window and re-split into the same sizes.
        pooled = pd.concat(window.values(), ignore_index=True)
        order = np.random.default_rng(seed + k).permutation(len(pooled))
        n_tr, n_va = len(window["train"]), len(window["val"])
        splits[f"random-{k}"] = {"train": pooled.iloc[order[:n_tr]].reset_index(drop=True),
                                 "val": pooled.iloc[order[n_tr:n_tr + n_va]].reset_index(drop=True),
                                 "test": pooled.iloc[order[n_tr + n_va:]].reset_index(drop=True)}
    for name, parts in splits.items():
        ids = [set(p["SEQN"]) for p in parts.values()]
        assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), f"{name}: splits overlap"
    return splits


def describe(splits) -> pd.DataFrame:
    rows = []
    for name, parts in splits.items():
        for part, df in parts.items():
            rows.append({"split": name, "part": part, "n": len(df), "n_pos": int(df["label"].sum()),
                         "prevalence": df["label"].mean(),
                         "cycles": ",".join(sorted(df["cycle"].unique()))})
    return pd.DataFrame(rows)


def load_config() -> dict:
    return yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text())


def model_settings(config: dict, numeric: list[str], categorical: list[str], n_jobs: int = -1,
                   quick: bool = False) -> dict:
    """Settings shared by every model fit: features, seed, threads, MLP epoch loop."""
    return {"numeric": numeric, "categorical": categorical, "seed": config["seed"], "n_jobs": n_jobs,
            "balanced": True, "max_epochs": 15 if quick else config["mlp_training"]["max_epochs"],
            "patience": config["mlp_training"]["patience"]}


def run_benchmark(dataset: str, models=MODEL_NAMES, only_splits=None, quick: bool = False,
                  n_jobs: int = -1) -> Path:
    """Tune, fit and evaluate every model on every split of one dataset; write all result files."""
    config = load_config()
    seed = config["seed"]
    spec = DATASETS[dataset]
    out_dir = ROOT / (spec["out"] + ("_quick" if quick else ""))
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)

    data, numeric, categorical = load_dataset(dataset)
    if quick:
        data = data.groupby("cycle").sample(n=600, random_state=seed).reset_index(drop=True)
    settings = model_settings(config, numeric, categorical, n_jobs, quick)
    logger.info("%s: rows %d | numeric %d | categorical %d | models %s", dataset, len(data), len(numeric),
                len(categorical), models)
    pd.DataFrame({"feature": numeric + categorical,
                  "type": ["numeric"] * len(numeric) + ["categorical"] * len(categorical)}
                 ).to_csv(out_dir / "feature_types.csv", index=False)

    splits = make_splits(data, seed, spec)
    if only_splits:
        splits = {k: v for k, v in splits.items() if k in only_splits}
    describe(splits).to_csv(out_dir / "splits.csv", index=False)

    metrics, tuning, preds, history, selected = [], [], [], [], {}
    started = time.perf_counter()
    for split_name, parts in splits.items():
        for model in models:
            t0 = time.perf_counter()
            tag = {"protocol": split_name, "seed": seed}
            fitted, params, rows = tune_model(model, config["models"][model], parts, settings, tag)
            m, p, threshold = evaluate_model(model, fitted, parts, settings, tag)
            metrics += m
            tuning += rows
            preds.append(p)
            selected.setdefault(split_name, {})[model] = {"params": params, "threshold": threshold,
                                                          "best_epoch": fitted["best_epoch"]}
            if fitted["history"] is not None:
                history.append(fitted["history"].assign(protocol=split_name))
            test_auc = next(r for r in m if r["split"] == "test" and r["threshold_rule"] == "val_f1")["roc_auc"]
            logger.info("%-17s %-14s test ROC-AUC %.3f  (%.0fs, total %.0fs)", split_name, model,
                        test_auc, time.perf_counter() - t0, time.perf_counter() - started)
        # Save after every split so a long run can be inspected (or resumed) early.
        pd.DataFrame(metrics).to_csv(out_dir / "metrics.csv", index=False)
        pd.DataFrame(tuning).to_csv(out_dir / "tuning.csv", index=False)
        pd.concat(preds, ignore_index=True).to_csv(out_dir / "predictions" / "test_predictions.csv", index=False)
        (out_dir / "selected_models.json").write_text(json.dumps(selected, indent=2))
        if history:
            pd.concat(history, ignore_index=True).to_csv(out_dir / "mlp_history.csv", index=False)

    summarize(pd.DataFrame(metrics), pd.concat(preds, ignore_index=True), config, out_dir, models,
              n_windows=len(spec["windows"]))
    logger.info("Done in %.1f min. Results in %s", (time.perf_counter() - started) / 60, out_dir)
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nhanes", choices=list(DATASETS))
    ap.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    ap.add_argument("--splits", nargs="+", default=None, help="subset of split names")
    ap.add_argument("--quick", action="store_true", help="smoke test on 600 rows per period")
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    run_benchmark(args.dataset, args.models, args.splits, args.quick, args.n_jobs)


def summarize(metrics: pd.DataFrame, preds: pd.DataFrame, config: dict, out_dir: Path, models, n_windows=3):
    """Tables that answer the TabReD questions."""
    boot = config["bootstrap"]
    n_res, conf, seed = boot["n_resamples"], boot["confidence"], config["seed"]
    test = metrics[(metrics["split"] == "test") & (metrics["threshold_rule"] == "val_f1")]

    # 1) Test ROC-AUC / PR-AUC per split and model (wide view).
    table = test.pivot_table(index="model", columns="protocol", values="roc_auc").reindex(models)
    table.round(4).to_csv(out_dir / "test_roc_auc_by_split.csv")

    # 2) Time-based vs random split, per window and averaged (TabReD Table 4 idea).
    rows = []
    for k in range(n_windows):
        tw, rw = f"sliding-window-{k}", f"random-{k}"
        if tw not in set(preds["protocol"]) or rw not in set(preds["protocol"]):
            continue
        for model in models:
            a = preds[(preds["protocol"] == rw) & (preds["model"] == model)]
            b = preds[(preds["protocol"] == tw) & (preds["model"] == model)]
            if a.empty or b.empty:
                continue
            for name, metric in [("roc_auc", roc_auc_score), ("pr_auc", average_precision_score)]:
                r = unpaired_difference(a["label"].to_numpy(), a["probability"].to_numpy(),
                                        b["label"].to_numpy(), b["probability"].to_numpy(),
                                        metric, n_res, conf, seed)
                rows.append({"window": k, "model": model, "metric": name,
                             "random": metric(a["label"], a["probability"]),
                             "time": metric(b["label"], b["probability"]), **r,
                             "ci_excludes_zero": r["diff_low"] > 0 or r["diff_high"] < 0})
    gaps = pd.DataFrame(rows)
    if not gaps.empty:
        gaps.to_csv(out_dir / "random_minus_time_by_window.csv", index=False)
        mean = gaps.groupby(["model", "metric"])[["random", "time", "diff"]].mean().reset_index()
        mean.to_csv(out_dir / "random_minus_time_mean.csv", index=False)
        # 3) Model ranking under time vs random splits (does the ranking change?).
        roc = mean[mean["metric"] == "roc_auc"].set_index("model")
        ranks = pd.DataFrame({"mean_roc_auc_time": roc["time"], "rank_time": roc["time"].rank(ascending=False),
                              "mean_roc_auc_random": roc["random"],
                              "rank_random": roc["random"].rank(ascending=False)})
        ranks.sort_values("rank_time").to_csv(out_dir / "ranking_time_vs_random.csv")

    # 4) Default split: bootstrap CIs and paired comparisons against the validation-selected model.
    default = preds[preds["protocol"] == "default"]
    if not default.empty:
        cis = [{"model": m, **metric_ci(g["label"].to_numpy(), g["probability"].to_numpy(), n_res, conf, seed)}
               for m, g in default.groupby("model")]
        pd.DataFrame(cis).to_csv(out_dir / "default_test_ci.csv", index=False)
        val = metrics[(metrics["protocol"] == "default") & (metrics["split"] == "val")
                      & (metrics["threshold_rule"] == "val_f1")]
        best = val.sort_values(["pr_auc", "roc_auc"], ascending=False)["model"].iloc[0]
        if default["model"].nunique() > 1:
            paired_comparisons(default, best, n_res, conf, seed).to_csv(
                out_dir / "default_paired_comparisons.csv", index=False)
        (out_dir / "default_selected_model.txt").write_text(best + "\n")


if __name__ == "__main__":
    main()

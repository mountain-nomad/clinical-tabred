"""Summary tables and headline numbers, always computed from the saved result files.

Every headline number is looked up with `lookup`, which records the source file,
column and row filter, so any number in the notebook, report or slides can be
traced back to (and re-checked against) a file in the results folder.
"""

import json
from pathlib import Path

import pandas as pd

from clinical_tabred.plotting import MODEL_LABELS


def lookup(results_dir: Path, source: str, column: str, row_filter: dict | None = None,
           agg: str = "single"):
    """Read one value from a result file.

    `source` is a CSV (filtered by `row_filter`, then reduced with agg = single,
    sum or max) or a JSON file (then `column` is a dot-separated key path).
    """
    path = Path(results_dir) / source
    if source.endswith(".json"):
        value = json.loads(path.read_text())
        for key in column.split("."):
            value = value[key]
        return value
    table = pd.read_csv(path)
    for key, wanted in (row_filter or {}).items():
        table = table[table[key] == wanted]
    if agg == "sum":
        return float(table[column].sum())
    if agg == "max":
        return float(table[column].max())
    if len(table) != 1:
        raise ValueError(f"{source} with filter {row_filter} matched {len(table)} rows, not 1")
    value = table[column].iloc[0]
    return value.item() if hasattr(value, "item") else value


def model_comparison_table(results_dir: Path, protocol: str = "temporal",
                           split: str = "test") -> pd.DataFrame:
    """All metrics of every model on one test set (tuned threshold) with bootstrap CIs."""
    metrics = pd.read_csv(Path(results_dir) / "metrics.csv")
    ci = pd.read_csv(Path(results_dir) / "metrics_ci.csv")
    seed = int(metrics["seed"].min())
    rows = metrics[(metrics["protocol"] == protocol) & (metrics["split"] == split)
                   & (metrics["threshold_rule"] == "val_f1") & (metrics["seed"] == seed)]
    ci = ci[(ci["protocol"] == protocol) & (ci["split"] == split) & (ci["seed"] == seed)]
    table = rows.merge(ci[["model", "roc_auc_low", "roc_auc_high", "pr_auc_low", "pr_auc_high"]],
                       on="model")
    columns = ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "roc_auc_low",
               "roc_auc_high", "pr_auc", "pr_auc_low", "pr_auc_high", "brier", "threshold",
               "train_time_s", "infer_ms_per_1k", "n", "n_pos"]
    table = table[columns].copy()
    table["model"] = table["model"].map(MODEL_LABELS)
    return table.reset_index(drop=True)


def cv_summary_table(results_dir: Path) -> pd.DataFrame:
    """Per-fold ROC-AUC plus mean ± std for each protocol and model."""
    folds = pd.read_csv(Path(results_dir) / "cv_folds.csv")
    wide = folds.pivot_table(index=["protocol", "model"], columns="fold", values="roc_auc")
    wide.columns = [f"fold_{c}" for c in wide.columns]
    stats = folds.groupby(["protocol", "model"])["roc_auc"].agg(["mean", "std"])
    return wide.join(stats).reset_index()


def key_number_specs(results_dir: Path) -> list[dict]:
    """Definitions (source file, column, filter) of the headline numbers."""
    best = lookup(results_dir, "selected_models.json", "best_model")
    seed = lookup(results_dir, "selected_models.json", "seed")
    test = {"protocol": "temporal", "split": "test", "model": best}
    specs = [{"id": "best_model", "label": "Model selected on 2013-2014 validation PR-AUC",
              "source": "selected_models.json", "column": "best_model"},
             {"id": "seed", "label": "Main random seed", "source": "selected_models.json",
              "column": "seed"}]
    for cycle in ["2011-2012", "2013-2014", "2015-2016"]:
        tag = cycle[:4]
        specs += [{"id": f"n_{tag}", "label": f"Eligible adults {cycle}", "source":
                   "cycle_summary.csv", "column": "n", "filter": {"cycle": cycle}},
                  {"id": f"prevalence_{tag}", "label": f"Unweighted prevalence {cycle}",
                   "source": "cycle_summary.csv", "column": "prevalence",
                   "filter": {"cycle": cycle}},
                  {"id": f"weighted_prevalence_{tag}", "label": f"Weighted prevalence {cycle}",
                   "source": "cycle_summary.csv", "column": "weighted_prevalence",
                   "filter": {"cycle": cycle}}]
    specs.append({"id": "n_total", "label": "Eligible adults, all cycles",
                  "source": "cycle_summary.csv", "column": "n", "agg": "sum"})
    for column in ["roc_auc", "roc_auc_low", "roc_auc_high", "pr_auc", "pr_auc_low",
                   "pr_auc_high"]:
        specs.append({"id": f"best_temporal_{column}", "label": f"Best model, 2015-2016 {column}",
                      "source": "metrics_ci.csv", "column": column, "filter": test})
    for column in ["f1", "recall", "precision", "accuracy", "threshold"]:
        specs.append({"id": f"best_temporal_{column}", "label": f"Best model, 2015-2016 {column}"
                      " (validation-tuned threshold)", "source": "metrics.csv", "column": column,
                      "filter": {**test, "threshold_rule": "val_f1", "seed": seed}})
    gap_a2 = {"comparison": "A2: in-period holdout minus 2015-2016", "model": best,
              "metric": "roc_auc"}
    gap_b = {"comparison": "B random test minus A temporal test", "model": best,
             "metric": "roc_auc"}
    for name, row_filter in [("a2_gap", gap_a2), ("random_minus_temporal", gap_b)]:
        for column in ["diff", "diff_low", "diff_high"]:
            specs.append({"id": f"{name}_roc_auc_{column}", "label": f"{name} ROC-AUC {column}",
                          "source": "period_gaps.csv", "column": column, "filter": row_filter})
    specs += [
        {"id": "random_roc_auc_mean", "label": "Best model ROC-AUC, random split (mean of seeds)",
         "source": "random_seeds_summary.csv", "column": "roc_auc_mean", "filter": {"model": best}},
        {"id": "random_n_seeds", "label": "Number of random-split seeds",
         "source": "random_seeds_summary.csv", "column": "n_seeds", "filter": {"model": best}},
        {"id": "random_roc_auc_std", "label": "Best model ROC-AUC, random split (std of seeds)",
         "source": "random_seeds_summary.csv", "column": "roc_auc_std", "filter": {"model": best}},
        {"id": "cv_temporal_roc_auc_mean", "label": "Best model 5-fold CV ROC-AUC (2011-2012)",
         "source": "cv_metrics.csv", "column": "roc_auc_mean",
         "filter": {"protocol": "temporal", "model": best}},
        {"id": "cv_temporal_roc_auc_std", "label": "Best model 5-fold CV ROC-AUC std",
         "source": "cv_metrics.csv", "column": "roc_auc_std",
         "filter": {"protocol": "temporal", "model": best}},
        {"id": "domain_auc_2015", "label": "Domain classifier ROC-AUC, 2011-2012 vs 2015-2016",
         "source": "domain_classifier.csv", "column": "roc_auc_mean",
         "filter": {"compared": "2015-2016"}},
        {"id": "max_ks_2015", "label": "Largest KS statistic, 2015-2016 vs 2011-2012",
         "source": "drift_numeric.csv", "column": "ks_statistic",
         "filter": {"compared": "2015-2016"}, "agg": "max"},
        {"id": "ablation_demographic_delta_roc_auc", "label": "ROC-AUC change without demographics",
         "source": "ablation.csv", "column": "delta_roc_auc",
         "filter": {"configuration": "without demographic features"}},
        {"id": "ablation_anthropometric_delta_roc_auc",
         "label": "ROC-AUC change without body measurements", "source": "ablation.csv",
         "column": "delta_roc_auc", "filter": {"configuration": "without anthropometric features"}},
        {"id": "mlp_best_epoch", "label": "MLP early-stopping epoch (protocol A)",
         "source": "mlp_history.csv", "column": "best_epoch",
         "filter": {"protocol": "temporal"}, "agg": "max"},
    ]
    return specs


def key_numbers(results_dir: Path) -> list[dict]:
    """Evaluate every key-number spec; each entry keeps its source for traceability."""
    entries = []
    for spec in key_number_specs(results_dir):
        value = lookup(results_dir, spec["source"], spec["column"], spec.get("filter"),
                       spec.get("agg", "single"))
        entries.append({**spec, "value": value})
    return entries


def ci_text(numbers: dict, prefix: str) -> str:
    """Format the 95% CI stored under `<prefix>_low` / `<prefix>_high` as "[low, high]"."""
    return f"[{numbers[prefix + '_low']:.3f}, {numbers[prefix + '_high']:.3f}]"


def key_findings(results_dir: Path) -> str:
    """Markdown bullet list of the main findings, with every number read from the results."""
    k = {e["id"]: e["value"] for e in key_numbers(results_dir)}
    name = MODEL_LABELS[k["best_model"]]
    lines = [
        (f"- Eligible adults: {k['n_total']:,.0f}; unweighted prevalence "
         f"{k['prevalence_2011']:.1%} (2011-2012), {k['prevalence_2013']:.1%} (2013-2014), "
         f"{k['prevalence_2015']:.1%} (2015-2016)."),
        (f"- Model selected on validation: **{name}**. Chronological test (2015-2016) ROC-AUC "
         f"{k['best_temporal_roc_auc']:.3f} {ci_text(k, 'best_temporal_roc_auc')}, PR-AUC "
         f"{k['best_temporal_pr_auc']:.3f} {ci_text(k, 'best_temporal_pr_auc')}, F1 "
         f"{k['best_temporal_f1']:.3f} at threshold {k['best_temporal_threshold']:.2f}."),
        (f"- Same model, random split: ROC-AUC {k['random_roc_auc_mean']:.3f} ± "
         f"{k['random_roc_auc_std']:.3f} ({k['random_n_seeds']} seeds); seed-{k['seed']} random split "
         f"minus chronological test {k['random_minus_temporal_roc_auc_diff']:+.3f} "
         f"{ci_text(k, 'random_minus_temporal_roc_auc_diff')}."),
        (f"- Protocol A2 (same fitted model): in-period holdout minus 2015-2016 ROC-AUC "
         f"{k['a2_gap_roc_auc_diff']:+.3f} {ci_text(k, 'a2_gap_roc_auc_diff')}."),
        (f"- Covariate shift 2011-2012 vs 2015-2016: largest numeric KS statistic "
         f"{k['max_ks_2015']:.3f}; domain-classifier ROC-AUC {k['domain_auc_2015']:.3f}."),
        (f"- Ablation: dropping demographic features changes ROC-AUC by "
         f"{k['ablation_demographic_delta_roc_auc']:+.3f}; dropping body measurements by "
         f"{k['ablation_anthropometric_delta_roc_auc']:+.3f}."),
    ]
    return "\n".join(lines)


def describe_interval(low: float, high: float) -> str:
    """Plain-language reading of a confidence interval for a difference."""
    if low > 0:
        return "positive (the 95% CI excludes zero)"
    if high < 0:
        return "negative (the 95% CI excludes zero)"
    return "inconclusive (the 95% CI includes zero)"


def interpretation(results_dir: Path) -> str:
    """Rule-based reading of the uncertainty results, so wording always matches the numbers."""
    paired = pd.read_csv(Path(results_dir) / "paired_comparisons.csv")
    roc = paired[paired["metric"] == "roc_auc"]
    k = {e["id"]: e["value"] for e in key_numbers(results_dir)}
    best = MODEL_LABELS[k["best_model"]]
    clear = roc[roc["ci_excludes_zero"]]
    beaten = ", ".join(MODEL_LABELS[m] for m in clear[clear["diff"] > 0]["other"]) or "none"
    worse = ", ".join(MODEL_LABELS[m] for m in clear[clear["diff"] < 0]["other"]) or "none"
    lines = [
        (f"- Paired bootstrap on 2015-2016: {best} has a clearly higher ROC-AUC than: {beaten}; "
         f"clearly lower than: {worse}; all other pairwise differences are inconclusive."),
        ("- Random-split minus chronological ROC-AUC (selected model) is "
         + describe_interval(k["random_minus_temporal_roc_auc_diff_low"],
                             k["random_minus_temporal_roc_auc_diff_high"]) + "."),
        ("- In-period holdout minus future-cycle ROC-AUC (protocol A2, selected model) is "
         + describe_interval(k["a2_gap_roc_auc_diff_low"], k["a2_gap_roc_auc_diff_high"]) + "."),
    ]
    return "\n".join(lines)

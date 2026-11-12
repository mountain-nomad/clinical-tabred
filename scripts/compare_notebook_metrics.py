"""Compare the standalone notebook's outputs with the repository results.

Usage: python scripts/compare_notebook_metrics.py <notebook_outputs_dir> [results_dir]
Writes results/notebook_consistency.json and exits with an error if a difference exceeds 1e-9.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TOLERANCE = 1e-9
KEYS = ["protocol", "seed", "model", "split", "threshold_rule"]
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier", "threshold",
           "tn", "fp", "fn", "tp"]
TABLES = ["ablation", "cv_metrics", "metrics_ci", "drift_numeric", "domain_classifier",
          "cycle_summary", "paired_comparisons", "period_gaps"]


def max_difference(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Largest absolute difference between the numeric columns of two equally shaped tables."""
    numeric = a.select_dtypes("number").columns
    return float(np.nanmax(np.abs(a[numeric].to_numpy() - b[numeric].to_numpy())))


def main() -> None:
    notebook_dir = Path(sys.argv[1])
    results_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results")
    repo = pd.read_csv(results_dir / "metrics.csv")
    notebook = pd.read_csv(notebook_dir / "metrics.csv")
    merged = repo.merge(notebook, on=KEYS, suffixes=("_repo", "_nb"))
    report = {"metric_rows_repo": len(repo), "metric_rows_notebook": len(notebook),
              "metric_rows_matched": len(merged), "tolerance": TOLERANCE,
              "max_abs_diff_metrics": max(float(np.nanmax(np.abs(merged[f"{m}_repo"]
                                                                 - merged[f"{m}_nb"])))
                                          for m in METRICS)}
    for name in TABLES:
        a, b = pd.read_csv(results_dir / f"{name}.csv"), pd.read_csv(notebook_dir / f"{name}.csv")
        report[f"max_abs_diff_{name}"] = max_difference(a, b) if a.shape == b.shape else None
    report["all_within_tolerance"] = (
        len(merged) == len(repo) == len(notebook)
        and all(v is not None and v <= TOLERANCE for k, v in report.items()
                if k.startswith("max_abs_diff")))
    (results_dir / "notebook_consistency.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["all_within_tolerance"]:
        sys.exit("Notebook and repository results differ beyond the tolerance.")


if __name__ == "__main__":
    main()

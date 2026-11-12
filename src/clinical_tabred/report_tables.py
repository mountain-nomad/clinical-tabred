"""Generate report tables (LaTeX), the model-comparison Markdown and figure captions.

Every number comes from the CSV/JSON files in the results folder; nothing is typed by hand.
"""

import json
import re
from pathlib import Path

import pandas as pd

from clinical_tabred.plotting import FIGURES, MODEL_LABELS
from clinical_tabred.summaries import (
    cv_summary_table,
    interpretation,
    key_findings,
    key_numbers,
    lookup,
    model_comparison_table,
)

NUMERIC_CELL = re.compile(r"^[-+0-9.,%± \[\]]+$")  # cells that should be right-aligned


def latex_escape(text) -> str:
    """Escape the characters that LaTeX treats specially (citations are kept as they are)."""
    text = str(text)
    if text.startswith(r"\cite{"):
        return text
    for char, replacement in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                              ("_", r"\_"), ("#", r"\#"), ("±", r"$\pm$"),
                              ("Δ", r"$\Delta$")]:
        text = text.replace(char, replacement)
    return text


def markdown_table(df: pd.DataFrame) -> str:
    """A GitHub-style Markdown table (no extra dependency needed)."""
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    divider = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.to_numpy()]
    return "\n".join([header, divider, *rows])


def latex_table(df: pd.DataFrame, caption: str, label: str, wide: bool = False) -> str:
    """A booktabs table for an already formatted DataFrame (`wide` spans both columns)."""
    env = "table*" if wide else "table"
    align = "".join("r" if NUMERIC_CELL.match(" ".join(df[c].astype(str))) else "l"
                    for c in df.columns)
    header = " & ".join(latex_escape(c) for c in df.columns) + r" \\"
    rows = [" & ".join(latex_escape(v) for v in row) + r" \\" for row in df.to_numpy()]
    return "\n".join([rf"\begin{{{env}}}[t]", r"\centering", r"\footnotesize",
                      rf"\caption{{{caption}}}", rf"\label{{{label}}}",
                      rf"\begin{{tabular}}{{{align}}}", r"\toprule", header, r"\midrule",
                      *rows, r"\bottomrule", r"\end{tabular}", rf"\end{{{env}}}", ""])


def dataset_table(results_dir: Path) -> pd.DataFrame:
    """Dataset characteristics per cycle."""
    s = pd.read_csv(Path(results_dir) / "cycle_summary.csv")
    return pd.DataFrame({
        "Cycle": s["cycle"], "Adults (n)": s["n"].map("{:,}".format),
        "Positives": s["n_pos"].map("{:,}".format),
        "Prevalence": s["prevalence"].map("{:.1%}".format),
        "Weighted prev.": s["weighted_prevalence"].map("{:.1%}".format),
        "Mean age": s["mean_age"].map("{:.1f}".format),
        "Mean BMI": s["mean_bmi"].map("{:.1f}".format),
        "Female": s["female_share"].map("{:.1%}".format)})


def comparison_table(results_dir: Path) -> pd.DataFrame:
    """Formatted chronological-test comparison table (tuned threshold, 95% CIs)."""
    t = model_comparison_table(results_dir)
    return pd.DataFrame({
        "Model": t["model"], "Acc.": t["accuracy"].map("{:.3f}".format),
        "Prec.": t["precision"].map("{:.3f}".format), "Rec.": t["recall"].map("{:.3f}".format),
        "F1": t["f1"].map("{:.3f}".format),
        "ROC-AUC [95% CI]": [f"{v:.3f} [{lo:.3f}, {hi:.3f}]" for v, lo, hi in
                             zip(t["roc_auc"], t["roc_auc_low"], t["roc_auc_high"])],
        "PR-AUC [95% CI]": [f"{v:.3f} [{lo:.3f}, {hi:.3f}]" for v, lo, hi in
                            zip(t["pr_auc"], t["pr_auc_low"], t["pr_auc_high"])],
        "Brier": t["brier"].map("{:.3f}".format), "Train (s)": t["train_time_s"].map(
            "{:.2f}".format)})


def protocol_table(results_dir: Path) -> pd.DataFrame:
    """Test ROC-AUC per model under A, A2 (both test sets) and B (mean ± std)."""
    m = pd.read_csv(Path(results_dir) / "metrics.csv")
    m = m[m["threshold_rule"] == "val_f1"]

    def pick(protocol, split):
        return m[(m["protocol"] == protocol) & (m["split"] == split)].groupby("model")["roc_auc"]

    random = pick("random", "test").agg(["mean", "std"])
    rows = []
    for model in [k for k in MODEL_LABELS if k in random.index]:
        rows.append({"Model": MODEL_LABELS[model],
                     "A2 in-period": f"{pick('in_period', 'test_in_period').first()[model]:.3f}",
                     "A2 future": f"{pick('in_period', 'test').first()[model]:.3f}",
                     "A chronological": f"{pick('temporal', 'test').first()[model]:.3f}",
                     "B random (mean ± std)": f"{random.loc[model, 'mean']:.3f} ± "
                                              f"{random.loc[model, 'std']:.3f}"})
    return pd.DataFrame(rows)


def cv_table(results_dir: Path) -> pd.DataFrame:
    """5-fold CV ROC-AUC per fold and mean ± std."""
    t = cv_summary_table(results_dir)
    out = pd.DataFrame({"Protocol": t["protocol"].map({"temporal": "A", "random": "B"}),
                        "Model": t["model"].map(MODEL_LABELS)})
    for column in [c for c in t.columns if c.startswith("fold_")]:
        out[column.replace("fold_", "Fold ")] = t[column].map("{:.3f}".format)
    out["Mean ± std"] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(t["mean"], t["std"])]
    return out


def ablation_table(results_dir: Path) -> pd.DataFrame:
    """Ablation results with deltas relative to the full model."""
    a = pd.read_csv(Path(results_dir) / "ablation.csv")
    return pd.DataFrame({"Configuration": a["configuration"],
                         "ROC-AUC": a["roc_auc"].map("{:.3f}".format),
                         "PR-AUC": a["pr_auc"].map("{:.3f}".format), "F1": a["f1"].map(
                             "{:.3f}".format),
                         "ΔROC-AUC": a["delta_roc_auc"].map("{:+.3f}".format),
                         "ΔPR-AUC": a["delta_pr_auc"].map("{:+.3f}".format),
                         "ΔF1": a["delta_f1"].map("{:+.3f}".format)})


def drift_table(results_dir: Path) -> pd.DataFrame:
    """Numeric drift 2015-2016 vs 2011-2012."""
    d = pd.read_csv(Path(results_dir) / "drift_numeric.csv")
    d = d[d["compared"] == d["compared"].max()]
    return pd.DataFrame({"Feature": d["feature"], "Mean 2011-12": d["mean_ref"].map(
        "{:.2f}".format), "Mean 2015-16": d["mean_cmp"].map("{:.2f}".format),
        "KS": d["ks_statistic"].map("{:.3f}".format), "PSI": d["psi"].map("{:.3f}".format),
        "Missing 2015-16": d["missing_pct_cmp"].map("{:.1f}%".format)})


def stack_table(results_dir: Path) -> pd.DataFrame:
    """Technology stack with the exact versions recorded at run time."""
    meta = json.loads((Path(results_dir) / "run_metadata.json").read_text())
    purposes = {"pandas": "tables, XPT reading", "numpy": "arrays", "scikit-learn":
                "pipeline, LR, RF, MLP, metrics", "xgboost": "gradient boosting",
                "lightgbm": "gradient boosting", "scipy": "KS test", "matplotlib": "figures",
                "pyyaml": "configuration", "requests": "data download"}
    rows = [{"Tool": "Python", "Version": meta["python"], "Purpose": "language"}]
    rows += [{"Tool": k, "Version": v, "Purpose": purposes.get(k, "")}
             for k, v in meta["packages"].items()]
    return pd.DataFrame(rows)


REPORT_TABLES = [
    ("dataset", dataset_table, "Dataset characteristics per NHANES cycle (adults with a valid label)."),
    ("comparison", comparison_table, "Chronological test set (2015-2016), validation-tuned thresholds; 95\\% bootstrap CIs."),
    ("protocols", protocol_table, "Test ROC-AUC under the three evaluation protocols."),
    ("cv", cv_table, "5-fold stratified cross-validation ROC-AUC on development data."),
    ("ablation", ablation_table, "Ablation of the validation-selected model on the 2015-2016 test set."),
    ("drift", drift_table, "Numeric feature drift, 2015-2016 vs 2011-2012."),
    ("stack", stack_table, "Technology stack (versions recorded at run time)."),
]


def write_report_tables(results_dir: Path, out_dir: Path) -> list[Path]:
    """Write every report table to `out_dir/<name>.tex`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, build, caption in REPORT_TABLES:
        path = out_dir / f"{name}.tex"
        wide = name != "stack"  # all but the narrow stack table span both columns
        path.write_text(latex_table(build(results_dir), caption, f"tab:{name}", wide))
        paths.append(path)
    return paths


def write_model_comparison_md(results_dir: Path) -> Path:
    """results/model_comparison.md: comparison tables plus the computed findings."""
    parts = ["# Model comparison", "", "Generated from the CSV files in this folder.", "",
             "## Chronological test set (2015-2016)", "",
             markdown_table(comparison_table(results_dir)), "",
             "## Test ROC-AUC by protocol", "", markdown_table(protocol_table(results_dir)),
             "", "## Key findings", "", key_findings(results_dir), "", interpretation(results_dir), ""]
    path = Path(results_dir) / "model_comparison.md"
    path.write_text("\n".join(parts))
    return path


def write_captions(figures_dir: Path) -> Path:
    """figures/captions.md built from the first docstring line of each plot function."""
    lines = ["# Figure captions", ""]
    for number, (name, plot) in enumerate(FIGURES, start=1):
        text = plot.__doc__.strip().splitlines()[0].split(":", 1)[1].strip()
        text = text[0].upper() + text[1:]
        lines.append(f"- **Fig. {number}** (`fig{name}.png` / `.pdf`): {text}")
    path = Path(figures_dir) / "captions.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def format_key_number(key: str, value) -> str:
    """Format a headline number for text: percentages, counts, signed differences, AUCs."""
    if key == "best_model":
        return MODEL_LABELS[value]
    if "prevalence" in key:
        return f"{value * 100:.1f}\\%"
    if key.startswith("n_") or key in {"mlp_best_epoch", "random_n_seeds", "seed"}:
        return f"{value:,.0f}".replace(",", "{,}")
    if "diff" in key or "delta" in key:
        return f"{value:+.3f}"
    return f"{value:.3f}" if key != "best_temporal_threshold" else f"{value:.2f}"


def write_number_macros(results_dir: Path, out_path: Path) -> Path:
    """LaTeX macros so the report text uses \\kn{id} instead of typed numbers."""
    lines = ["% Generated from the results folder by report_tables.py. Do not edit by hand.",
             r"\makeatletter", r"\newcommand{\kn}[1]{\@nameuse{kn@#1}}"]
    for entry in key_numbers(results_dir):
        lines.append(rf"\@namedef{{kn@{entry['id']}}}"
                     f"{{{format_key_number(entry['id'], entry['value'])}}}")
    lines.append(r"\makeatother")
    Path(out_path).write_text("\n".join(lines) + "\n")
    return Path(out_path)


def write_literature_tables(results_dir: Path, curated_dir: Path) -> dict[str, pd.DataFrame]:
    """results/related_work.csv and results/literature_comparison.csv.

    Literature rows are copied from curated files (values typed exactly from the
    opened sources); the rows for this work are read from metrics.csv.
    """
    related = pd.read_csv(Path(curated_dir) / "related_work_curated.csv")
    literature = pd.read_csv(Path(curated_dir) / "literature_values_curated.csv")
    metrics = pd.read_csv(Path(results_dir) / "metrics.csv")
    seed = int(lookup(results_dir, "selected_models.json", "seed"))
    rows = []
    for model in ["xgboost", "lightgbm", "mlp"]:
        for split, label in [("test_in_period", "A2 in-period holdout (2011-2012)"),
                             ("test", "A2 future cycle (2015-2016)")]:
            row = metrics[(metrics["protocol"] == "in_period") & (metrics["model"] == model)
                          & (metrics["split"] == split) & (metrics["threshold_rule"] == "val_f1")
                          & (metrics["seed"] == seed)].iloc[0]
            for metric in ["accuracy", "roc_auc"]:
                rows.append({"method": MODEL_LABELS[model], "ref": "this work",
                             "dataset": "NHANES adults (DIQ010)", "protocol": label,
                             "metric": metric, "value": round(float(row[metric]), 3),
                             "notes": "From results/metrics.csv; accuracy depends on the "
                                      "validation-tuned threshold."})
    literature = pd.concat([literature, pd.DataFrame(rows)], ignore_index=True)
    related.to_csv(Path(results_dir) / "related_work.csv", index=False)
    literature.to_csv(Path(results_dir) / "literature_comparison.csv", index=False)
    return {"related_work": related, "literature_comparison": literature}


def timeline_table(timeline_csv: Path) -> pd.DataFrame:
    """Planned project timeline (a plan, not a result) for the report and slides."""
    t = pd.read_csv(timeline_csv)
    return pd.DataFrame({"Task": t["task"], "Weeks": [f"{a}-{b}" if a != b else f"{a}"
                                                      for a, b in zip(t["start_week"],
                                                                      t["end_week"])],
                         "Deliverable": t["deliverable"]})


def literature_table(results_dir: Path) -> pd.DataFrame:
    """Published values next to ours (different datasets: see notes before comparing)."""
    lit = pd.read_csv(Path(results_dir) / "literature_comparison.csv")
    lit = lit[lit["metric"] == "accuracy"]
    return pd.DataFrame({"Method": lit["method"], "Source": lit["ref"].map(
        lambda r: "This work" if r == "this work" else r"\cite{" + r + "}"),
        "Dataset": lit["dataset"], "Test set": lit["protocol"],
        "Accuracy": lit["value"].map("{:.3f}".format)})


def write_extra_tables(results_dir: Path, timeline_csv: Path, out_dir: Path) -> list[Path]:
    """Timeline and literature tables (written separately because of their inputs)."""
    timeline = latex_table(timeline_table(timeline_csv), "Planned project timeline.",
                           "tab:timeline", wide=True)
    literature = latex_table(literature_table(results_dir), "Accuracy reported by TableShift "
                             "for its BRFSS Diabetes task next to ours. Datasets, shifts and "
                             "thresholds differ, so the values are not directly comparable.",
                             "tab:literature", wide=True)
    paths = [Path(out_dir) / "timeline.tex", Path(out_dir) / "literature.tex"]
    paths[0].write_text(timeline)
    paths[1].write_text(literature)
    return paths

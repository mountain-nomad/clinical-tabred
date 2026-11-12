"""Build the self-contained for_presentation/ folder used to design the slides.

Everything is generated from the results folder, the figures code and the config, so
no number in the pack is typed by hand. Run with `--stage presentation`.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve

from clinical_tabred.diagrams import make_all_diagrams
from clinical_tabred.plotting import (
    MODEL_LABELS,
    apply_style,
    plot_ablation,
    plot_auc_with_ci,
    plot_confusion_matrix,
    plot_feature_drift,
    plot_mlp_history,
    plot_protocol_comparison,
    plot_roc_curves,
    save_figure,
)
from clinical_tabred.presentation_docs import (
    render_sample_slides,
    write_algorithm,
    write_pack_readme,
    write_placeholders,
    write_references,
    write_style_guide,
)
from clinical_tabred.report_tables import (
    ablation_table,
    comparison_table,
    cv_table,
    dataset_table,
    markdown_table,
    stack_table,
    timeline_table,
)
from clinical_tabred.slides import write_slides_md
from clinical_tabred.summaries import key_numbers

# Reference keys in order of first citation in the deck (numbers [1], [2], ...).
REFERENCE_ORDER = ["rubachev2024tabred", "gardner2023tableshift", "grinsztajn2022why",
                   "rabanser2019failing", "nhanes", "nhanesdiq", "pedregosa2011scikit",
                   "chen2016xgboost", "ke2017lightgbm", "tabredrepo"]
SLIDE_FIGURES = [("f05_auc_ci", plot_auc_with_ci), ("f06_mlp_curves", plot_mlp_history),
                 ("f07_confusion_matrix", plot_confusion_matrix),
                 ("f08_roc_curves", plot_roc_curves), ("f09_ablation", plot_ablation),
                 ("f10_feature_drift_ks", plot_feature_drift),
                 ("f11_random_vs_temporal", plot_protocol_comparison)]


def cite_number(key: str) -> str:
    """IEEE citation label such as [2] for a bibliography key."""
    return f"[{REFERENCE_ORDER.index(key) + 1}]"


def save_table(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    """Write one table as tNN_name.csv and tNN_name.md."""
    df.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / f"{name}.md").write_text(markdown_table(df) + "\n")


def mark_best(df: pd.DataFrame, higher_better: list[str], lower_better: list[str]) -> pd.DataFrame:
    """Wrap the best value of each listed column in ** ** (bold in Markdown)."""
    df = df.copy()
    for column, pick in [(c, max) for c in higher_better] + [(c, min) for c in lower_better]:
        values = df[column].str.split(" ").str[0].astype(float)
        best = values == pick(values)
        df.loc[best, column] = "**" + df.loc[best, column] + "**"
    return df


def features_table(results_dir: Path, config: dict) -> pd.DataFrame:
    """Feature, type, source file, missing % in 2015-2016 and preprocessing."""
    missing = pd.read_csv(Path(results_dir) / "missingness.csv").set_index("feature")
    data_cfg = config["data"]
    rows = []
    for feature in data_cfg["numeric_features"] + data_cfg["categorical_features"]:
        numeric = feature in data_cfg["numeric_features"]
        rows.append({"Feature": feature, "Type": "numeric" if numeric else "categorical",
                     "Source": "BMX" if feature.startswith("BMX") else "DEMO",
                     "Missing 2015-16": f"{missing.loc[feature, '2015-2016']:.1f}%",
                     "Preprocessing": "median impute → scale" if numeric
                     else "mode impute → one-hot"})
    return pd.DataFrame(rows)


def gaps_table(results_dir: Path) -> pd.DataFrame:
    """Period gaps (ROC-AUC) with 95% CIs for every model."""
    gaps = pd.read_csv(Path(results_dir) / "period_gaps.csv")
    gaps = gaps[gaps["metric"] == "roc_auc"]
    return pd.DataFrame({"Comparison": gaps["comparison"], "Model": gaps["model"].map(MODEL_LABELS),
                         "ΔROC-AUC": gaps["diff"].map("{:+.3f}".format),
                         "95% CI": [f"[{a:+.3f}, {b:+.3f}]" for a, b in
                                    zip(gaps["diff_low"], gaps["diff_high"])]})


def write_tables(results_dir: Path, config: dict, out_dir: Path) -> None:
    """All tables of the deck, numbered in order of appearance."""
    out_dir.mkdir(parents=True, exist_ok=True)
    related = pd.read_csv(Path(results_dir) / "related_work.csv")
    related["ref"] = related["ref"].map(cite_number)
    literature = pd.read_csv(Path(results_dir) / "literature_comparison.csv")
    literature["ref"] = literature["ref"].map(lambda r: r if r == "this work" else cite_number(r))
    comparison = mark_best(comparison_table(results_dir),
                           ["Acc.", "Prec.", "Rec.", "F1", "ROC-AUC [95% CI]", "PR-AUC [95% CI]"],
                           ["Brier"])
    cv = cv_table(results_dir)
    tables = {"t01_cycle_context": dataset_table(results_dir),
              "t02_timeline": timeline_table(Path("report/timeline.csv")),
              "t03_related_work": related.drop(columns="verification"),
              "t04_dataset": dataset_table(results_dir)[["Cycle", "Adults (n)", "Positives",
                                                         "Prevalence", "Weighted prev."]],
              "t05_features": features_table(results_dir, config),
              "t06_tech_stack": stack_table(results_dir),
              "t07_temporal_test_results": comparison,
              "t08_cv_results": cv[cv["Protocol"] == "A"].drop(columns="Protocol"),
              "t09_ablation": ablation_table(results_dir),
              "t10_period_gaps": gaps_table(results_dir),
              "t11_literature_comparison": literature}
    for name, table in tables.items():
        save_table(table, out_dir, name)


def roc_points(results_dir: Path, n_points: int = 101) -> pd.DataFrame:
    """ROC curve of every model on 2015-2016, thinned to about n_points rows per model."""
    parts = []
    for model in MODEL_LABELS:
        pred = pd.read_csv(Path(results_dir) / "predictions" / f"temporal_{model}.csv")
        pred = pred[pred["split"] == "test"]
        fpr, tpr, thr = roc_curve(pred["label"], pred["probability"])
        step = max(1, len(fpr) // n_points)
        keep = list(range(0, len(fpr), step)) + [len(fpr) - 1]
        parts.append(pd.DataFrame({"model": model, "fpr": fpr[keep], "tpr": tpr[keep],
                                   "threshold": thr[keep]}).drop_duplicates())
    return pd.concat(parts, ignore_index=True)


def write_chart_data(results_dir: Path, out_dir: Path) -> None:
    """Raw numbers behind every chart, so charts can be redrawn natively in the slides."""
    out_dir.mkdir(parents=True, exist_ok=True)
    r = Path(results_dir)
    ci = pd.read_csv(r / "metrics_ci.csv")
    metrics = pd.read_csv(r / "metrics.csv")
    best = json.loads((r / "selected_models.json").read_text())["best_model"]
    cm = pd.read_csv(r / "confusion_matrices.csv")
    numeric = pd.read_csv(r / "drift_numeric.csv")
    rule = metrics[metrics["threshold_rule"] == "val_f1"]
    protocol = rule[rule["split"].isin(["test", "test_in_period"])][
        ["protocol", "seed", "model", "split", "roc_auc", "pr_auc"]]
    timeline = pd.read_csv("report/timeline.csv")
    charts = {
        "c01_auc_ci": ci[(ci["protocol"] == "temporal") & (ci["split"] == "test")],
        "c02_mlp_history": pd.read_csv(r / "mlp_history.csv").query("protocol == 'temporal'"),
        "c03_roc_points": roc_points(results_dir),
        "c04_confusion_matrix": cm[(cm["protocol"] == "temporal") & (cm["model"] == best)
                                   & (cm["split"] == "test")],
        "c05_ablation_deltas": pd.read_csv(r / "ablation.csv"),
        "c06_ks_values": numeric[["feature", "compared", "ks_statistic", "psi"]],
        "c07_category_shifts": pd.read_csv(r / "drift_categorical.csv"),
        "c08_protocol_comparison": protocol,
        "c09_prevalence": pd.read_csv(r / "cycle_summary.csv")[["cycle", "n", "n_pos", "prevalence",
                                                               "weighted_prevalence"]],
        "c10_gantt": timeline,
        "c11_cv_folds": pd.read_csv(r / "cv_folds.csv"),
    }
    for name, table in charts.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)


def write_figures(results_dir: Path, out_dir: Path) -> None:
    """Slide-ready figures (large fonts, deck palette, white background) as PNG and SVG."""
    apply_style("slide")
    for name, plot in SLIDE_FIGURES:
        fig = plot(results_dir, "slide")
        save_figure(fig, out_dir, name, ["png", "svg"])
        plt.close(fig)


def write_key_numbers(results_dir: Path, out_path: Path) -> list[dict]:
    """key_numbers.json: each headline value with its source file, column and row filter."""
    entries = key_numbers(results_dir)
    for entry in entries:
        entry["source"] = f"results/{entry['source']}"
    Path(out_path).write_text(json.dumps(entries, indent=2, default=float))
    return entries


EQUATIONS = [
    ("Logistic model", "P(y = 1 | x) = σ(wᵀx + b) = 1 / (1 + e^−(wᵀx + b))",
     r"P(y=1\mid\mathbf{x})=\sigma(\mathbf{w}^\top\mathbf{x}+b)", "w: weights, b: bias, σ: sigmoid"),
    ("Precision and recall", "P = TP / (TP + FP),   R = TP / (TP + FN)",
     r"P=\frac{TP}{TP+FP},\; R=\frac{TP}{TP+FN}", ("TP/FP/FN: true positives, false positives, "
     "false negatives at threshold τ")),
    ("F1-score", "F1 = 2PR / (P + R)", r"F_1=\frac{2PR}{P+R}", "P: precision, R: recall"),
    ("KS statistic", "D = supₓ | F_ref(x) − F_cmp(x) |",
     r"D=\sup_x\lvert F_{ref}(x)-F_{cmp}(x)\rvert", ("F: empirical CDF of a feature in the "
     "reference (2011-12) and compared cycle")),
    ("Population Stability Index", "PSI = Σ_b (q_b − p_b) · ln(q_b / p_b)",
     r"\mathrm{PSI}=\sum_b (q_b-p_b)\ln\frac{q_b}{p_b}", ("p_b, q_b: share of rows in bin b "
     "(10 quantile bins fixed on 2011-12) in the reference and compared cycle")),
    ("CV mean and std", "m̄ = (1/K) Σₖ mₖ,   s = sqrt( Σₖ (mₖ − m̄)² / (K − 1) )",
     r"\bar m=\frac1K\sum_k m_k,\; s=\sqrt{\frac{\sum_k (m_k-\bar m)^2}{K-1}}",
     "m_k: metric on fold k, K = 5 folds"),
    ("Bootstrap 95% CI", "CI = [ θ*₍₀.₀₂₅₎ , θ*₍₀.₉₇₅₎ ] over 1000 resamples",
     r"\mathrm{CI}_{95\%}=[\hat\theta^*_{(0.025)},\hat\theta^*_{(0.975)}]",
     "θ*: metric recomputed on each bootstrap resample of the test set"),
]


def write_equations(out_path: Path) -> None:
    """equations.md: numbered Unicode equations (as on the sample slide) plus LaTeX."""
    lines = ["# Equations", "", ("Slide 11 shows (1)–(6); (7) is referenced on the results "
             "slides. Same numbering as the report."), ""]
    for number, (name, unicode, latex, symbols) in enumerate(EQUATIONS, start=1):
        lines += [f"## ({number}) {name}", "", f"Unicode: `{unicode}`", "",
                  f"LaTeX: `{latex}`", "", f"Symbols: {symbols}", ""]
    Path(out_path).write_text("\n".join(lines))


def build_pack(results_dir: Path, config: dict, pack_dir: Path) -> None:
    """Generate the whole for_presentation/ folder (the sample .pptx must already be there)."""
    pack_dir = Path(pack_dir)
    pptx = pack_dir / "reference" / "CSCI447_Final_Project_Sample_Presentation.pptx"
    if not pptx.exists():
        raise FileNotFoundError(f"Sample deck missing: copy it to {pptx}")
    render_note = render_sample_slides(pptx, pack_dir / "reference" / "sample_slides")
    write_tables(results_dir, config, pack_dir / "tables")
    write_chart_data(results_dir, pack_dir / "charts_data")
    write_figures(results_dir, pack_dir / "figures")
    make_all_diagrams(pack_dir / "diagrams", ["png", "svg"])
    numbers = {e["id"]: e["value"] for e in write_key_numbers(results_dir,
                                                              pack_dir / "key_numbers.json")}
    write_slides_md(numbers, config["project"], pack_dir / "slides.md")
    write_equations(pack_dir / "equations.md")
    write_algorithm(pack_dir / "algorithm.md")
    write_references(Path("report/references.bib"), REFERENCE_ORDER, pack_dir / "references.md")
    write_style_guide(pptx, pack_dir / "style_guide.md", render_note)
    write_placeholders(pack_dir / "slides.md", render_note, pack_dir / "PLACEHOLDERS.md")
    write_pack_readme(pack_dir, render_note)

"""Figures. Every function reads the saved CSV files, so plots always show real results.

Colours and fonts follow the course sample deck (navy/gold/teal/rust theme,
Calibri body font); model colours were checked for colour-blind separation.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.metrics import roc_curve

MODEL_LABELS = {"logreg": "Logistic Regression", "random_forest": "Random Forest",
                "xgboost": "XGBoost", "lightgbm": "LightGBM", "mlp": "MLP"}
MODEL_COLORS = {"logreg": "#1F5FA8", "random_forest": "#A8822C", "xgboost": "#109A92",
                "lightgbm": "#B4432E", "mlp": "#6B4E9B"}
CYCLE_COLORS = ["#9FB0CC", "#4F6A99", "#14213D"]  # light -> dark navy: older -> newer
INK, MUTED, GRID = "#1F2937", "#5B6472", "#D5DAE3"


def apply_style(style: str) -> None:
    """Set matplotlib defaults. 'paper' for the report/notebook, 'slide' for 16:9 slides."""
    fonts = [f for f in ["Calibri", "Carlito", "Arial", "DejaVu Sans"]
             if f in {entry.name for entry in font_manager.fontManager.ttflist}]
    base = 15 if style == "slide" else 10
    plt.rcParams.update({
        "font.family": fonts[0], "font.size": base, "axes.titlesize": base + 2,
        "axes.labelsize": base, "xtick.labelsize": base - 1, "ytick.labelsize": base - 1,
        "legend.fontsize": base - 1, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "figure.facecolor": "white", "savefig.facecolor": "white", "legend.frameon": False,
    })


def figure_size(style: str, height_ratio: float = 0.45) -> tuple[float, float]:
    """Width x height in inches: wide 16:9-friendly for slides, column width for paper."""
    width = 12.0 if style == "slide" else 7.0
    return width, width * height_ratio


def bottom_legend(fig, ax, ncol: int, height: float = 0.14) -> None:
    """Place the legend of `ax` below the plot so it never covers data."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=ncol)
    fig.tight_layout(rect=(0, height, 1, 1))


def models_in(table: pd.DataFrame) -> list[str]:
    """Model names in the fixed display order (so colours never change between plots)."""
    return [m for m in MODEL_LABELS if m in set(table["model"])]


def plot_dataset_characteristics(results_dir: Path, style: str):
    """Fig. 1: respondents per cycle and the share of missing values per feature."""
    summary = pd.read_csv(Path(results_dir) / "cycle_summary.csv")
    missing = pd.read_csv(Path(results_dir) / "missingness.csv").set_index("feature")
    missing = missing[(missing > 0).any(axis=1)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figure_size(style),
                                   gridspec_kw={"width_ratios": [1, 2]})
    bars = ax1.bar(summary["cycle"], summary["n"], color=CYCLE_COLORS, width=0.6)
    ax1.bar_label(bars, fmt="{:,.0f}", padding=2)
    ax1.set(title="Adults with a valid label", ylabel="Respondents")
    ax1.set_ylim(0, summary["n"].max() * 1.15)
    width = 0.8 / len(missing.columns)
    for i, cycle in enumerate(missing.columns):
        ax2.bar(np.arange(len(missing)) + i * width, missing[cycle], width * 0.92,
                color=CYCLE_COLORS[i], label=cycle)
    ax2.set_xticks(np.arange(len(missing)) + width, missing.index, rotation=30, ha="right")
    ax2.set(title="Missing values per feature", ylabel="Missing (%)")
    ax2.legend(title="Cycle")
    fig.tight_layout()
    return fig


def plot_prevalence(results_dir: Path, style: str):
    """Fig. 2: share of adults reporting diagnosed diabetes, unweighted and survey-weighted."""
    summary = pd.read_csv(Path(results_dir) / "cycle_summary.csv")
    fig, ax = plt.subplots(figsize=figure_size(style, 0.5 if style == "paper" else 0.42))
    x = np.arange(len(summary))
    for offset, column, color, label in [(-0.18, "prevalence", "#14213D", "Unweighted (sample)"),
                                         (0.18, "weighted_prevalence", "#A8822C",
                                          "Survey-weighted (WTINT2YR)")]:
        bars = ax.bar(x + offset, summary[column] * 100, 0.34, color=color, label=label)
        ax.bar_label(bars, fmt="{:.1f}%", padding=2)
    ax.set_xticks(x, summary["cycle"])
    ax.set(ylabel="Self-reported diagnosed diabetes (%)", title="Target prevalence by cycle")
    ax.set_ylim(0, summary["prevalence"].max() * 100 * 1.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_auc_with_ci(results_dir: Path, style: str):
    """Fig. 3: temporal-test ROC-AUC and PR-AUC per model with 95% bootstrap CIs."""
    ci = pd.read_csv(Path(results_dir) / "metrics_ci.csv")
    ci = ci[(ci["protocol"] == "temporal") & (ci["split"] == "test")].set_index("model")
    order = models_in(ci.reset_index())
    fig, axes = plt.subplots(1, 2, figsize=figure_size(style))
    for ax, metric, title in [(axes[0], "roc_auc", "ROC-AUC"), (axes[1], "pr_auc", "PR-AUC")]:
        for y, model in enumerate(order):
            row = ci.loc[model]
            ax.errorbar(row[metric], y, xerr=[[row[metric] - row[f"{metric}_low"]],
                                              [row[f"{metric}_high"] - row[metric]]],
                        fmt="o", ms=8, capsize=4, color=MODEL_COLORS[model], lw=2)
            ax.annotate(f"{row[metric]:.3f}", (row[f"{metric}_high"], y), xytext=(6, 0),
                        textcoords="offset points", va="center", color=INK)
        ax.set_yticks(range(len(order)), [MODEL_LABELS[m] for m in order])
        ax.invert_yaxis()
        ax.set(title=f"{title} on 2015-2016 (95% CI)", xlabel=title)
        ax.margins(x=0.25)
    axes[1].set_yticklabels([])
    fig.tight_layout()
    return fig


def split_scores(metrics: pd.DataFrame, protocol: str, split: str) -> pd.DataFrame:
    """Rows of one protocol and split, indexed by model."""
    return metrics[(metrics["protocol"] == protocol) & (metrics["split"] == split)].set_index("model")


def plot_protocol_comparison(results_dir: Path, style: str):
    """Fig. 4: test ROC-AUC under the random split, the in-period holdout and the future cycle."""
    metrics = pd.read_csv(Path(results_dir) / "metrics.csv")
    rule = metrics[(metrics["threshold_rule"] == "val_f1")]
    random = (rule[(rule["protocol"] == "random") & (rule["split"] == "test")]
              .groupby("model")["roc_auc"].agg(["mean", "std"]))
    n_seeds = rule.loc[rule["protocol"] == "random", "seed"].nunique()
    series = [(f"B: random split (mean ± std, {n_seeds} seeds)", random["mean"], random["std"],
               "#9FB0CC", "s"),
              ("A2: in-period holdout (2011-2012)",
               split_scores(rule, "in_period", "test_in_period")["roc_auc"], None, "#4F6A99", "^"),
              ("A: chronological test (2015-2016)", split_scores(rule, "temporal", "test")["roc_auc"],
               None, "#14213D", "o")]
    order = models_in(rule)
    fig, ax = plt.subplots(figsize=figure_size(style, 0.5))
    for i, (label, values, std, color, marker) in enumerate(series):
        y = np.arange(len(order)) + (i - 1) * 0.22
        ax.errorbar(values[order], y, xerr=None if std is None else std[order], fmt=marker,
                    ms=8, capsize=3, color=color, label=label, lw=1.5)
    ax.set_yticks(range(len(order)), [MODEL_LABELS[m] for m in order])
    ax.invert_yaxis()
    ax.set(xlabel="Test ROC-AUC", title="Random vs in-period vs chronological evaluation")
    bottom_legend(fig, ax, ncol=3 if style == "slide" else 2,
                  height=0.12 if style == "slide" else 0.15)
    return fig


def plot_roc_curves(results_dir: Path, style: str):
    """Fig. 5: ROC curves of all models on the chronological test cycle (2015-2016)."""
    fig, ax = plt.subplots(figsize=figure_size(style, 0.62 if style == "paper" else 0.5))
    for model, label in MODEL_LABELS.items():
        path = Path(results_dir) / "predictions" / f"temporal_{model}.csv"
        pred = pd.read_csv(path)
        pred = pred[pred["split"] == "test"]
        fpr, tpr, _ = roc_curve(pred["label"], pred["probability"])
        ax.plot(fpr, tpr, color=MODEL_COLORS[model], lw=2, label=label)
    ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1, label="Chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="ROC curves, chronological test set (2015-2016)", aspect="equal")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(results_dir: Path, style: str):
    """Fig. 6: confusion matrix of the validation-selected model on the 2015-2016 test set."""
    best = json.loads((Path(results_dir) / "selected_models.json").read_text())["best_model"]
    cm = pd.read_csv(Path(results_dir) / "confusion_matrices.csv")
    row = cm[(cm["protocol"] == "temporal") & (cm["model"] == best) & (cm["split"] == "test")
             & (cm["threshold_rule"] == "val_f1")].iloc[0]
    matrix = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
    fig, ax = plt.subplots(figsize=(6.5, 5.2) if style == "slide" else (4.2, 3.6))
    ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.grid(False)
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value:,}\n({value / matrix.sum():.1%})", ha="center", va="center",
                color="white" if value > matrix.max() / 2 else INK)
    ax.set_xticks([0, 1], ["No diabetes", "Diabetes"])
    ax.set_yticks([0, 1], ["No diabetes", "Diabetes"])
    ax.set(xlabel="Predicted", ylabel="Actual (self-reported)",
           title=f"{MODEL_LABELS[best]}, 2015-2016 test\n(threshold {row['threshold']:.2f})")
    fig.tight_layout()
    return fig


def plot_feature_drift(results_dir: Path, style: str):
    """Fig. 7: KS statistic of numeric features and category-share changes vs 2011-2012."""
    numeric = pd.read_csv(Path(results_dir) / "drift_numeric.csv")
    categorical = pd.read_csv(Path(results_dir) / "drift_categorical.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figure_size(style, 0.5))
    features = list(dict.fromkeys(numeric["feature"]))
    for i, (cycle, part) in enumerate(numeric.groupby("compared")):
        ax1.barh(np.arange(len(features)) + i * 0.4, part.set_index("feature").loc[features,
                 "ks_statistic"], 0.38, color=CYCLE_COLORS[i + 1], label=f"{cycle} vs 2011-2012")
    ax1.set_yticks(np.arange(len(features)) + 0.2, features)
    ax1.invert_yaxis()
    ax1.set(xlabel="KS statistic (0 = identical)", title="Numeric features")
    last = categorical[(categorical["compared"] == categorical["compared"].max())
                       & (categorical["category"] != "missing")]
    last = last.reindex(last["diff_pp"].abs().sort_values(ascending=False).index).head(8)
    colors = ["#1F5FA8" if d > 0 else "#B4432E" for d in last["diff_pp"]]
    ax2.barh(last["feature"] + ": " + last["category"], last["diff_pp"], color=colors)
    ax2.axvline(0, color=MUTED, lw=1)
    ax2.invert_yaxis()
    ax2.set(xlabel="Change in share (percentage points)",
            title=f"Largest category shifts, {last['compared'].iloc[0]} vs 2011-2012")
    bottom_legend(fig, ax1, ncol=2, height=0.1)
    return fig


def plot_mlp_history(results_dir: Path, style: str):
    """Fig. 8: MLP loss and accuracy per epoch (protocol A), with the early-stopping epoch."""
    history = pd.read_csv(Path(results_dir) / "mlp_history.csv")
    history = history[history["protocol"] == "temporal"]
    best = int(history["best_epoch"].iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=figure_size(style))
    for ax, metric, title in [(axes[0], "loss", "Log loss (class-weighted)"),
                              (axes[1], "accuracy", "Accuracy (threshold 0.5)")]:
        ax.plot(history["epoch"], history[f"train_{metric}"], color="#1F5FA8", lw=2,
                label="Train (2011-2012)")
        ax.plot(history["epoch"], history[f"val_{metric}"], color="#A8822C", lw=2,
                label="Validation (2013-2014)")
        ax.axvline(best, color=MUTED, ls="--", lw=1.2,
                   label=f"Early stopping: best epoch {best}")
        ax.set(xlabel="Epoch", title=title)
    bottom_legend(fig, axes[0], ncol=3 if style == "slide" else 2,
                  height=0.1 if style == "slide" else 0.16)
    return fig


def plot_ablation(results_dir: Path, style: str):
    """Fig. 9: change in test ROC-AUC, PR-AUC and F1 when one component is removed."""
    ablation = pd.read_csv(Path(results_dir) / "ablation.csv").iloc[1:]  # row 0 = full model
    fig, ax = plt.subplots(figsize=figure_size(style, 0.5))
    y = np.arange(len(ablation))
    for i, (metric, color) in enumerate([("roc_auc", "#1F5FA8"), ("pr_auc", "#A8822C"),
                                         ("f1", "#109A92")]):
        ax.barh(y + (i - 1) * 0.26, ablation[f"delta_{metric}"], 0.24, color=color,
                label=f"Δ {metric.replace('_', '-').upper()}")
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y, ablation["configuration"])
    ax.invert_yaxis()
    model = MODEL_LABELS[ablation["model"].iloc[0]]
    ax.set(xlabel="Change vs full model on 2015-2016 test set", title=f"Ablation study ({model})")
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


FIGURES = [("01_dataset_characteristics", plot_dataset_characteristics),
           ("02_prevalence", plot_prevalence), ("03_auc_ci", plot_auc_with_ci),
           ("04_random_vs_temporal", plot_protocol_comparison),
           ("05_roc_curves", plot_roc_curves), ("06_confusion_matrix", plot_confusion_matrix),
           ("07_feature_drift_ks", plot_feature_drift), ("08_mlp_curves", plot_mlp_history),
           ("09_ablation", plot_ablation)]


def save_figure(fig, out_dir: Path, name: str, formats: list[str]) -> list[Path]:
    """Save one figure as e.g. PNG (300 DPI) and PDF; returns the written paths."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = [Path(out_dir) / f"{name}.{fmt}" for fmt in formats]
    for path in paths:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return paths


def make_all_figures(results_dir: Path, out_dir: Path, style: str, formats: list[str],
                     prefix: str = "fig") -> list[Path]:
    """Draw every figure from the CSVs in `results_dir` and save it in each format."""
    apply_style(style)
    saved = []
    for name, plot in FIGURES:
        fig = plot(results_dir, style)
        saved += save_figure(fig, out_dir, f"{prefix}{name}", formats)
        plt.close(fig)
    return saved

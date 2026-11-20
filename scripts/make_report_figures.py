"""Extra figures for the IEEE report (for_presentation/for_report/P034_report/figures/).

- fig11_mlp_loss_accuracy.png: MLP training curves, loss AND accuracy per epoch (default split),
  from for_presentation/charts_data/c06_mlp_history.csv.
- fig00_timeline.png: project timeline (phases in order; no calendar dates are recorded).

Run from the repository root:  python scripts/make_report_figures.py
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from clinical_tabred.plotting import INK, MODEL_COLORS, MUTED, apply_style  # noqa: E402

OUT = ROOT / "for_presentation" / "for_report" / "P034_report" / "figures"
NAMES = {"nhanes": "NHANES (diabetes)", "brfss": "BRFSS (diabetes)", "natality": "Natality (preterm birth)"}


def fig_mlp_loss_accuracy() -> None:
    h = pd.read_csv(ROOT / "for_presentation" / "charts_data" / "c06_mlp_history.csv")
    h = h[h["protocol"] == "default"]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 5.6))
    for j, ds in enumerate(["nhanes", "brfss", "natality"]):
        d = h[h["dataset"] == ds]
        best = d["best_epoch"].iloc[0]
        for i, (col, label) in enumerate([("loss", "balanced log loss"), ("accuracy", "accuracy (threshold 0.5)")]):
            ax = axes[i, j]
            ax.plot(d["epoch"], d[f"train_{col}"], color=MODEL_COLORS["logreg"], lw=1.8, marker="o", ms=3,
                    label="training")
            ax.plot(d["epoch"], d[f"val_{col}"], color=MODEL_COLORS["lightgbm"], lw=1.8, marker="s", ms=3,
                    label="validation (later period)")
            ax.axvline(best, color=MUTED, ls="--", lw=0.9, label="early-stopping epoch")
            ax.set_ylabel(label)
            if i == 0:
                ax.set_title(NAMES[ds])
            else:
                ax.set_xlabel("epoch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "fig11_mlp_loss_accuracy.png", dpi=300)
    plt.close(fig)


PHASES = [
    ("1", "Literature\nreview", "TabReD, TableShift,\nshift detection"),
    ("2", "NHANES\nprototype", "9 features,\n3 cycles"),
    ("3", "Wide NHANES\ntable", "2005-2018,\n280 features"),
    ("4", "BRFSS +\nnatality", "tables and\nleakage audits"),
    ("5", "Benchmark", "5 models x 7 splits\nx 3 datasets"),
    ("6", "CV + drift\nanalysis", "5-fold CV, domain\nclassifier, causes"),
    ("7", "Deliverables", "notebook, report,\npresentation"),
]


def fig_timeline() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 2.5))
    ax.set_xlim(0, len(PHASES))
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.annotate("", xy=(len(PHASES) - 0.02, 0.86), xytext=(0.02, 0.86),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    shades = ["#9FB0CC", "#9FB0CC", "#4F6A99", "#4F6A99", "#14213D", "#14213D", "#109A92"]
    for k, (num, title, sub) in enumerate(PHASES):
        x = k + 0.06
        ax.add_patch(FancyBboxPatch((x, 0.04), 0.88, 0.67, boxstyle="round,pad=0.0,rounding_size=0.06",
                                    fc=shades[k], ec="white", lw=2))
        txt = "white" if k >= 2 else INK
        ax.text(k + 0.5, 0.635, f"Phase {num}", ha="center", va="center", fontsize=9, color=txt)
        ax.text(k + 0.5, 0.47, title, ha="center", va="center", fontsize=10.5, weight="bold", color=txt,
                linespacing=1.1)
        ax.text(k + 0.5, 0.2, sub, ha="center", va="center", fontsize=8.8, color=txt, linespacing=1.15)
        ax.plot(k + 0.5, 0.86, "o", ms=8, color=shades[k], mec="white", mew=1.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig00_timeline.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    apply_style("paper")
    plt.rcParams["axes.grid"] = True
    fig_mlp_loss_accuracy()
    fig_timeline()
    print("wrote", OUT)

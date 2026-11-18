"""Diagrams of the three-dataset benchmark: methodology, model architecture and the split design.

Usage: python scripts/make_diagrams.py
Output: figures/d01_methodology, d02_architecture, d03_splits (.png 300 DPI, .pdf, .svg) plus a
.md text description of each (boxes and arrows), so the diagrams can be redrawn natively on slides.
The split diagram is drawn from the DATASETS definition in run_benchmark.py, so it always shows the
splits the benchmark really used.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_benchmark import DATASETS  # noqa: E402

OUT = ROOT / "figures"
NAVY, GOLD, TEAL, RUST, LIGHT, INK, MUTED = "#24375F", "#A8822C", "#109A92", "#B4432E", "#F3F5F9", "#1F2937", "#5B6472"
TRAIN, VAL, TEST = "#9FB0CC", "#C9A55C", "#B4432E"

METHODOLOGY = {
    "title": "Methodology: a clinical TabReD benchmark", "box": (2.0, 1.25),
    "boxes": [
        (0.0, 0, "1 Public data\nNHANES 2005-18\nBRFSS 2017-23\nNatality 2016-24"),
        (2.4, 0, "2 Build tables\nharmonise codes,\nmissing codes,\nevery-period rule"),
        (4.8, 0, "3 Leakage audit\ntext filter,\nunivariate AUC,\nmissingness AUC"),
        (7.2, 0, "4 TabReD splits\ndefault time split\n3 sliding windows\n3 random (same size)"),
        (9.6, 0, "5 Train 5 models\nLR, RF, XGBoost,\nLightGBM, MLP\ntuned on validation"),
        (12.0, 0, "6 Evaluate\nmetrics + bootstrap CI\nrandom vs time\nsame-period, 5-fold CV"),
        (14.4, 0, "7 Drift by cause\ndomain classifier\nPSI / TVD per feature\nclinical-cause events"),
    ],
    "arrows": [(i, i + 1, "") for i in range(6)],
}
ARCHITECTURE = {
    "title": "Model pipeline (identical for every dataset and split)", "box": (2.4, 1.1),
    "boxes": [
        (0.0, 2.4, "Wide table\nNHANES 280 / BRFSS 93 /\nNatality 73 features"),
        (3.0, 2.4, "ColumnTransformer\nnumeric: median + scale\ncategorical: mode + one-hot\n(fit on train only)"),
        (6.0, 4.8, "Logistic Regression\nC grid, balanced"),
        (6.0, 3.6, "Random Forest\ndepth / leaf grid"),
        (6.0, 2.4, "XGBoost\nscale_pos_weight"),
        (6.0, 1.2, "LightGBM\nscale_pos_weight"),
        (6.0, 0.0, "MLP\nepoch loop,\nearly stopping"),
        (9.0, 2.4, "Model selection\nvalidation PR-AUC\n(ROC-AUC tiebreak)"),
        (12.0, 2.4, "Threshold\nmax F1 on\nvalidation"),
        (15.0, 2.4, "Test, scored once\nmetrics, confusion,\nbootstrap 95% CI"),
    ],
    "arrows": [(0, 1, "")] + [(1, i, "") for i in range(2, 7)] + [(i, 7, "") for i in range(2, 7)]
              + [(7, 8, ""), (8, 9, "")],
}


def draw_boxes(spec: dict, font: int = 11):
    width, height = spec["box"]
    xs = [x for x, _, _ in spec["boxes"]]
    ys = [y for _, y, _ in spec["boxes"]]
    fig, ax = plt.subplots(figsize=(max(xs) - min(xs) + width + 0.8, max(ys) - min(ys) + height + 1.0))
    last = len(spec["boxes"]) - 1
    for i, (x, y, text) in enumerate(spec["boxes"]):
        color = NAVY if i == 0 else (GOLD if i == last else TEAL)
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.04",
                                    facecolor=LIGHT, edgecolor=color, linewidth=2))
        title, *rest = text.split("\n")
        ax.text(x + width / 2, y + height - 0.2, title, ha="center", va="center", fontsize=font,
                color=INK, fontweight="bold")
        ax.text(x + width / 2, y + (height - 0.35) / 2, "\n".join(rest), ha="center", va="center",
                fontsize=font - 2, color=INK)
    for start, end, label in spec["arrows"]:
        x0, y0, _ = spec["boxes"][start]
        x1, y1, _ = spec["boxes"][end]
        ax.add_patch(FancyArrowPatch((x0 + width + 0.04, y0 + height / 2), (x1 - 0.04, y1 + height / 2),
                                     arrowstyle="-|>", mutation_scale=13, color=MUTED, linewidth=1.3))
    ax.set_xlim(min(xs) - 0.2, max(xs) + width + 0.2)
    ax.set_ylim(min(ys) - 0.2, max(ys) + height + 0.4)
    ax.set_title(spec["title"], fontsize=font + 4, color=NAVY)
    ax.axis("off")
    return fig


def describe_boxes(spec: dict) -> str:
    lines = [f"# {spec['title']}", "", "## Boxes (left to right, top to bottom)"]
    for i, (_, _, text) in enumerate(spec["boxes"]):
        title, *rest = text.split("\n")
        lines.append(f"{i + 1}. **{title}**: {' '.join(rest)}")
    lines += ["", "## Arrows"]
    for start, end, _ in spec["arrows"]:
        lines.append(f"- {spec['boxes'][start][2].splitlines()[0]} -> {spec['boxes'][end][2].splitlines()[0]}")
    return "\n".join(lines) + "\n"


def split_rows():
    """(dataset, split name, {period: role}) for the default split and the sliding windows."""
    rows = []
    for ds, spec in DATASETS.items():
        periods, (n_tr, n_va) = spec["periods"], spec["default"]
        roles = {p: "train" for p in periods[:n_tr]} | {p: "val" for p in periods[n_tr:n_tr + n_va]}
        roles |= {p: "test" for p in periods[n_tr + n_va:]}
        rows.append((ds, "default", roles))
        for k, s in enumerate(spec["windows"]):
            roles = {p: "train" for p in periods[s:s + 3]} | {periods[s + 3]: "val", periods[s + 4]: "test"}
            rows.append((ds, f"sliding-window-{k}", roles))
    return rows


def draw_splits():
    names = {"nhanes": "NHANES (2-year cycles)", "brfss": "BRFSS (years)", "natality": "Natality (years)"}
    colors = {"train": TRAIN, "val": VAL, "test": TEST}
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.6), gridspec_kw={"width_ratios": [7, 7, 9]})
    for ax, ds in zip(axes, DATASETS):
        periods = DATASETS[ds]["periods"]
        rows = [(n, r) for d, n, r in split_rows() if d == ds]
        for i, (name, roles) in enumerate(rows):
            y = len(rows) - 1 - i
            for j, p in enumerate(periods):
                role = roles.get(p)
                ax.add_patch(Rectangle((j + 0.05, y + 0.1), 0.9, 0.8, facecolor=colors.get(role, "white"),
                                       edgecolor="#D5DAE3", linewidth=1))
            ax.text(-0.15, y + 0.5, name, ha="right", va="center", fontsize=9, color=INK)
        ax.set_xlim(-0.1, len(periods))
        ax.set_ylim(-0.1, len(rows))
        ax.set_xticks([j + 0.5 for j in range(len(periods))], [p[:4] if "-" not in p else p[2:4] + "-" + p[7:9]
                                                               for p in periods], fontsize=8)
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.set_title(names[ds], fontsize=11, color=NAVY)
    handles = [Rectangle((0, 0), 1, 1, facecolor=c) for c in colors.values()]
    fig.legend(handles, ["train", "validation", "test"], loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Time-based splits (TabReD construction). Each random-k split pools the rows of "
                 "sliding-window-k and re-splits them into the same sizes.", fontsize=10, color=MUTED, y=0.02 + 0.98)
    fig.tight_layout(rect=(0.04, 0.08, 1, 0.94))
    return fig


def describe_splits() -> str:
    lines = ["# Time-based splits (TabReD construction)", "",
             "Each random-k split pools all rows of sliding-window-k and re-splits them at random into the",
             "same train / validation / test sizes (no temporal order).", "",
             "| Dataset | Split | Train | Validation | Test |", "|---|---|---|---|---|"]
    for ds, name, roles in split_rows():
        part = {r: ", ".join(p for p, v in roles.items() if v == r) for r in ("train", "val", "test")}
        lines.append(f"| {ds} | {name} | {part['train']} | {part['val']} | {part['test']} |")
    return "\n".join(lines) + "\n"


def save(fig, name: str, description: str):
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight",
                    metadata={"CreationDate": None} if ext == "pdf" else {"Date": None} if ext == "svg" else None)
    plt.close(fig)
    (OUT / f"{name}.md").write_text(description)
    print("saved", name)


def main():
    save(draw_boxes(METHODOLOGY), "d01_methodology", describe_boxes(METHODOLOGY))
    save(draw_boxes(ARCHITECTURE), "d02_architecture", describe_boxes(ARCHITECTURE))
    save(draw_splits(), "d03_splits", describe_splits())


if __name__ == "__main__":
    main()

"""Box-and-arrow diagrams (methodology, architecture, protocols, features) drawn with matplotlib.

Each diagram is defined as plain data (boxes and arrows), so the same definition is
used to draw the figure and to write its text description for the slides.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

NAVY, GOLD, TEAL, LIGHT, INK_COLOR = "#24375F", "#A8822C", "#109A92", "#F3F5F9", "#1F2937"

METHODOLOGY = {
    "title": "Proposed methodology", "box": (1.5, 1.0),
    "boxes": [(i * 1.85, 0, text) for i, text in enumerate([
        "1 Download\nNHANES XPT\n(validated)", "2 Prepare\nmerge, label,\nfilter adults",
        "3 Features\n9 inputs,\nspecial codes", "4 Split\nA / A2 / B",
        "5 Cross-\nvalidation\n5-fold", "6 Train\n5 models,\nsmall grids",
        "7 Evaluate\nmetrics, CIs,\nablation", "8 Drift\nKS, PSI,\ndomain clf."])],
    "arrows": [(i, i + 1, "") for i in range(7)],
}
ARCHITECTURE = {
    "title": "System architecture", "box": (2.3, 1.0),
    "boxes": [(0, 2.4, "NHANES table\n6 numeric +\n3 categorical"),
              (2.9, 2.4, "ColumnTransformer\nimpute + scale /\nimpute + one-hot\n(fit on train only)"),
              (5.8, 4.8, "Logistic\nRegression"), (5.8, 3.6, "Random Forest"),
              (5.8, 2.4, "XGBoost"), (5.8, 1.2, "LightGBM"),
              (5.8, 0.0, "MLP\n(epoch loop,\nearly stopping)"),
              (8.7, 2.4, "Threshold\nmax F1 on\nvalidation"),
              (11.6, 2.4, "Metrics\nROC/PR-AUC, F1,\nBrier, bootstrap CI")],
    "arrows": [(0, 1, "")] + [(1, i, "") for i in range(2, 7)] + [(i, 7, "") for i in range(2, 7)]
              + [(7, 8, "")],
}
PROTOCOLS = {
    "title": "Evaluation protocols", "box": (2.7, 1.0),
    "boxes": [(0, 2.0, "2011-2012\n(cycle G)"), (4.8, 2.0, "2013-2014\n(cycle H)"),
              (9.6, 2.0, "2015-2016\n(cycle I)"),
              (0, 0.2, "A2: 80% train,\n20% in-period test,\nsame model on 2015-16"),
              (4.8, 0.2, "B: all cycles pooled,\n70/15/15 random\n(5 seeds)"),
              (9.6, 0.2, "5-fold CV on\ndevelopment data\nonly")],
    "arrows": [(0, 1, "A: train → validate"), (1, 2, "A: → test (once)")],
}
FEATURES = {
    "title": "Features and preprocessing", "box": (2.6, 1.0),
    "boxes": [(0, 2.8, "Demographic\nRIDAGEYR, RIAGENDR,\nRIDRETH3"),
              (0, 1.4, "Socioeconomic\nDMDEDUC2, INDFMPIR"),
              (0, 0.0, "Anthropometric\nBMXBMI, BMXWAIST,\nBMXHT, BMXWT"),
              (3.6, 2.1, "Numeric (6)\nmedian imputation\n→ standard scaling"),
              (3.6, 0.7, "Categorical (3)\nmode imputation\n→ one-hot encoding"),
              (7.2, 1.4, "Model input\n(fit on training\nsplit only)")],
    "arrows": [(0, 3, ""), (0, 4, ""), (1, 3, ""), (1, 4, ""), (2, 3, ""), (3, 5, ""), (4, 5, "")],
}
DIAGRAMS = {"d01_methodology": METHODOLOGY, "d02_architecture": ARCHITECTURE,
            "d03_protocols": PROTOCOLS, "d04_features": FEATURES}


def draw_diagram(spec: dict, font: int = 13):
    """Draw boxes (x, y, text) and arrows (from_index, to_index, label) from a spec dict."""
    width, height = spec["box"]
    xs = [x for x, _, _ in spec["boxes"]]
    ys = [y for _, y, _ in spec["boxes"]]
    fig, ax = plt.subplots(figsize=(max(xs) - min(xs) + width + 1.0,
                                    max(ys) - min(ys) + height + 1.2))
    for i, (x, y, text) in enumerate(spec["boxes"]):
        color = NAVY if i == 0 else (GOLD if i == len(spec["boxes"]) - 1 else TEAL)
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.04",
                                    facecolor=LIGHT, edgecolor=color, linewidth=2))
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
                fontsize=font, color=INK_COLOR)
    for start, end, label in spec["arrows"]:
        x0, y0, _ = spec["boxes"][start]
        x1, y1, _ = spec["boxes"][end]
        a = (x0 + width, y0 + height / 2) if x1 > x0 else (x0 + width / 2, y0)
        b = (x1, y1 + height / 2) if x1 > x0 else (x1 + width / 2, y1 + height)
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14,
                                     color="#5B6472", linewidth=1.4))
        if label:
            ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.12, label, ha="center",
                    fontsize=font - 1, color=NAVY)
    ax.set_xlim(min(xs) - 0.3, max(xs) + width + 0.3)
    ax.set_ylim(min(ys) - 0.3, max(ys) + height + 0.5)
    ax.set_title(spec["title"], fontsize=font + 3, color=NAVY)
    ax.axis("off")
    return fig


def describe_diagram(spec: dict) -> str:
    """Markdown description of every box and arrow (for rebuilding the diagram natively)."""
    lines = [f"# {spec['title']}", "", "## Boxes"]
    for i, (x, y, text) in enumerate(spec["boxes"]):
        lines.append(f"{i + 1}. **{text.replace(chr(10), ' ')}** (position x={x}, y={y})")
    lines += ["", "## Arrows"]
    for start, end, label in spec["arrows"]:
        source = spec["boxes"][start][2].replace("\n", " ")
        target = spec["boxes"][end][2].replace("\n", " ")
        lines.append(f"- {source} → {target}" + (f" (label: {label})" if label else ""))
    return "\n".join(lines) + "\n"


def make_all_diagrams(out_dir: Path, formats: list[str]) -> list[Path]:
    """Save every diagram (PNG at 300 DPI and/or SVG) plus its .md description."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for name, spec in DIAGRAMS.items():
        fig = draw_diagram(spec)
        for fmt in formats:
            fig.savefig(out_dir / f"{name}.{fmt}", dpi=300, bbox_inches="tight")
            saved.append(out_dir / f"{name}.{fmt}")
        plt.close(fig)
        (out_dir / f"{name}.md").write_text(describe_diagram(spec))
        saved.append(out_dir / f"{name}.md")
    return saved

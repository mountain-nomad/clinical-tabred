"""Plot style shared by scripts/make_figures.py and the notebook.

Colours and fonts follow the course sample deck (navy/gold/teal/rust theme,
Calibri body font); model colours were checked for colour-blind separation.
"""

import matplotlib.pyplot as plt
from matplotlib import font_manager

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
    base = 16 if style == "slide" else 10
    plt.rcParams.update({
        "font.family": fonts[0], "font.size": base, "axes.titlesize": base + 2,
        "axes.labelsize": base, "xtick.labelsize": base - 1, "ytick.labelsize": base - 1,
        "legend.fontsize": base - 1, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "figure.facecolor": "white", "savefig.facecolor": "white", "legend.frameon": False,
    })

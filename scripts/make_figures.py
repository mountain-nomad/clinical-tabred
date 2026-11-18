"""Report figures for the three-dataset benchmark (one panel per dataset), from saved results.

Usage: python scripts/make_figures.py [--style paper|slide]
Output: figures/figNN_*.png/.pdf/.svg (captions in figures/captions.md). The diagrams d01-d03 in the
same folder come from scripts/make_diagrams.py.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from clinical_tabred.plotting import INK, MODEL_COLORS, MODEL_LABELS, MUTED, apply_style  # noqa: E402
from run_benchmark import DATASETS  # noqa: E402

FIGURES_DIR = ROOT / "figures"
SUMMARY_DIR = ROOT / "results_summary"
FORMATS = ("png", "pdf", "svg")
NAMES = {"nhanes": "NHANES (diabetes)", "brfss": "BRFSS (diabetes)", "natality": "Natality (preterm birth)"}
MODELS = list(MODEL_LABELS)
CAPTIONS = {}


def res(ds: str, name: str) -> Path:
    path = ROOT / DATASETS[ds]["out"] / name
    gz = path.with_name(path.name + ".gz")   # large prediction files are stored gzipped in git
    return gz if not path.exists() and gz.exists() else path


def present(name: str) -> list[str]:
    return [ds for ds in DATASETS if res(ds, name).exists()]


def panels(n: int, height: float = 3.0, width: float = 4.2):
    fig, axes = plt.subplots(1, n, figsize=(width * n, height), squeeze=False)
    return fig, axes[0]


def model_legend(fig, models=MODELS, ncol=5):
    handles = [plt.Line2D([], [], color=MODEL_COLORS[m], lw=6) for m in models]
    fig.legend(handles, [MODEL_LABELS[m] for m in models], loc="lower center", ncol=ncol)
    fig.tight_layout(rect=(0, 0.1, 1, 1))


def save(fig, name: str, caption: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in FORMATS:
        fig.savefig(FIGURES_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight",
                    metadata={"CreationDate": None} if ext == "pdf" else {"Date": None} if ext == "svg" else None)
    plt.close(fig)
    CAPTIONS[name] = caption
    print("saved", name)


def fig_prevalence():
    dss = present("drift_periods.csv")
    fig, axes = panels(len(dss), 2.8)
    for ax, ds in zip(axes, dss):
        p = pd.read_csv(res(ds, "drift_periods.csv"))
        ax.plot(range(len(p)), p["prevalence"] * 100, marker="o", color=INK)
        ax.set_xticks(range(len(p)), [str(c)[:4] for c in p["cycle"]], rotation=45)
        ax.set_title(NAMES[ds]); ax.set_ylabel("positive rate, %")
    fig.tight_layout()
    save(fig, "fig01_prevalence", "Positive-class rate per period (label shift). NHANES cycles are labelled by start year.")


def fig_auc_ci():
    dss = present("default_test_ci.csv")
    fig, axes = panels(len(dss), 3.0)
    for ax, ds in zip(axes, dss):
        ci = pd.read_csv(res(ds, "default_test_ci.csv")).set_index("model").reindex(MODELS)
        y = np.arange(len(MODELS))
        ax.errorbar(ci["roc_auc"], y, xerr=[ci["roc_auc"] - ci["roc_auc_low"], ci["roc_auc_high"] - ci["roc_auc"]],
                    fmt="none", ecolor=MUTED, capsize=3)
        ax.scatter(ci["roc_auc"], y, c=[MODEL_COLORS[m] for m in MODELS], s=50, zorder=3)
        ax.set_yticks(y, [MODEL_LABELS[m] for m in MODELS]); ax.invert_yaxis()
        ax.set_title(NAMES[ds]); ax.set_xlabel("test ROC-AUC (95% bootstrap CI)")
    fig.tight_layout()
    save(fig, "fig02_default_auc_ci", "Default time split: test ROC-AUC per model with 95% bootstrap confidence intervals.")


def fig_random_vs_time():
    dss = present("random_minus_time_by_window.csv")
    fig, axes = panels(len(dss), 3.2)
    for ax, ds in zip(axes, dss):
        g = pd.read_csv(res(ds, "random_minus_time_by_window.csv"))
        g = g[g["metric"] == "roc_auc"]
        windows = sorted(g["window"].unique())
        width = 0.16
        test = [DATASETS[ds]["periods"][s + 4] for s in DATASETS[ds]["windows"]]
        for i, m in enumerate(MODELS):
            r = g[g["model"] == m].set_index("window").reindex(windows)
            x = np.arange(len(windows)) + (i - 2) * width
            ax.bar(x, r["diff"], width, color=MODEL_COLORS[m])
            ax.errorbar(x, r["diff"], yerr=[r["diff"] - r["diff_low"], r["diff_high"] - r["diff"]],
                        fmt="none", ecolor=INK, lw=0.8, capsize=1.5)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(range(len(windows)), [f"window {w}\ntest {str(t)[:4]}" for w, t in zip(windows, test)])
        ax.set_title(NAMES[ds]); ax.set_ylabel("ROC-AUC: random minus time")
    model_legend(fig)
    save(fig, "fig03_random_minus_time",
         "Random-split minus time-split test ROC-AUC per sliding window and model (95% bootstrap CI). "
         "Positive values: a random split overestimates performance on future data.")


def fig_roc():
    dss = present("predictions/test_predictions.csv")
    fig, axes = panels(len(dss), 3.4, 3.6)
    for ax, ds in zip(axes, dss):
        p = pd.read_csv(res(ds, "predictions/test_predictions.csv"))
        p = p[p["protocol"] == "default"]
        for m in MODELS:
            q = p[p["model"] == m]
            fpr, tpr, _ = roc_curve(q["label"], q["probability"])
            ax.plot(fpr, tpr, color=MODEL_COLORS[m], lw=1.3)
        ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=0.8)
        ax.set_title(NAMES[ds]); ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    model_legend(fig)
    save(fig, "fig04_roc_curves", "ROC curves on the default split's test periods.")


def fig_confusion():
    dss = present("predictions/test_predictions.csv")
    fig, axes = panels(len(dss), 3.2, 3.4)
    for ax, ds in zip(axes, dss):
        best = res(ds, "default_selected_model.txt").read_text().strip()
        thr = json.loads(res(ds, "selected_models.json").read_text())["default"][best]["threshold"]
        p = pd.read_csv(res(ds, "predictions/test_predictions.csv"))
        q = p[(p["protocol"] == "default") & (p["model"] == best)]
        cm = confusion_matrix(q["label"], q["probability"] >= thr)
        ax.imshow(cm, cmap="Blues"); ax.grid(False)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}\n({v / cm[i].sum():.0%})", ha="center", va="center",
                    color="white" if v > cm.max() / 2 else INK)
        ax.set_xticks([0, 1], ["pred. 0", "pred. 1"]); ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_title(f"{NAMES[ds]}\n{MODEL_LABELS[best]}, threshold {thr:.2f}")
    fig.tight_layout()
    save(fig, "fig05_confusion_matrices",
         "Confusion matrices of the validation-selected model on the default split's test periods "
         "(threshold = max F1 on validation; row percentages).")


def fig_mlp():
    dss = present("mlp_history.csv")
    fig, axes = panels(len(dss), 3.0)
    for ax, ds in zip(axes, dss):
        h = pd.read_csv(res(ds, "mlp_history.csv"))
        h = h[h["protocol"] == "default"]
        ax.plot(h["epoch"], h["train_loss"], color=MODEL_COLORS["logreg"], label="train")
        ax.plot(h["epoch"], h["val_loss"], color=MODEL_COLORS["lightgbm"], label="validation (later period)")
        ax.axvline(h["best_epoch"].iloc[0], color=MUTED, ls="--", lw=0.8)
        ax.set_title(NAMES[ds]); ax.set_xlabel("epoch"); ax.set_ylabel("balanced log loss")
    axes[0].legend()
    fig.tight_layout()
    save(fig, "fig06_mlp_training_curves",
         "MLP training curves on the default split (selected configuration); dashed line = early-stopping epoch.")


def fig_domain():
    dss = present("drift_domain.csv")
    fig, axes = panels(len(dss), 3.0)
    for ax, ds in zip(axes, dss):
        d = pd.read_csv(res(ds, "drift_domain.csv"))
        adj = d[d["kind"] == "adjacent"]
        ax.plot(range(len(adj)), adj["roc_auc"], marker="o", color=INK, label="adjacent periods")
        ax.set_xticks(range(len(adj)), [c.split(" vs ")[1][:4] for c in adj["comparison"]], rotation=45)
        tw = d[(d["kind"] == "time") & d["split"].str.startswith("sliding")]["roc_auc"].mean()
        rw = d[d["kind"] == "random"]["roc_auc"].mean()
        ax.axhline(tw, color=MODEL_COLORS["lightgbm"], ls="--", label=f"time windows, train vs test ({tw:.2f})")
        ax.axhline(rw, color=MODEL_COLORS["xgboost"], ls=":", label=f"random splits ({rw:.2f})")
        ax.set_ylim(0.45, 1.0); ax.set_title(NAMES[ds]); ax.set_ylabel("domain classifier ROC-AUC")
        ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, "fig07_domain_shift",
         "Covariate shift: ROC-AUC of a classifier telling two periods apart (0.5 = indistinguishable). "
         "Points: each period vs the previous one (x = later period).")


def fig_events():
    items = []
    for ds in present("drift_events.csv"):
        e = pd.read_csv(res(ds, "drift_events.csv"))
        cols = [c for c in e.columns if c[:4].isdigit()]
        for _, r in e.iterrows():
            items.append((ds, r, cols))
    n = len(items)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.3 * nrow), squeeze=False)
    for ax, (ds, r, cols) in zip(axes.flat, items):
        v = r[cols].astype(float).values
        scale = 100 if r["statistic"] != "mean" or r["feature"] in ("label",) or v.max() <= 1 else 1
        ax.plot(range(len(cols)), v * scale, marker="o", color=INK, ms=3)
        if str(r["change_at"]) in cols:
            ax.axvline(cols.index(str(r["change_at"])) - 0.5, color=MODEL_COLORS["lightgbm"], ls="--", lw=1)
        ax.set_xticks(range(len(cols)), [c[:4] for c in cols], rotation=60, fontsize=7)
        unit = "%" if scale == 100 else "mean"
        ax.set_title(f"{ds}: {r['feature']} ({unit})\n{r['cause']}", fontsize=8)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    save(fig, "fig08_drift_events",
         "Drift with a documented clinical cause, measured per period (dashed line = period of the change).")


def fig_cv():
    dss = present("cv_folds.csv")
    if not dss:
        return
    fig, axes = panels(len(dss), 3.0)
    for ax, ds in zip(axes, dss):
        f = pd.read_csv(res(ds, "cv_folds.csv"))
        data = [f.loc[f["model"] == m, "roc_auc"].values for m in MODELS]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6)
        for patch, m in zip(bp["boxes"], MODELS):
            patch.set_facecolor(MODEL_COLORS[m])
        ax.set_xticks(range(1, 6), [MODEL_LABELS[m].replace(" ", "\n") for m in MODELS], fontsize=7)
        ax.set_title(NAMES[ds]); ax.set_ylabel("ROC-AUC (5 folds)")
    fig.tight_layout()
    save(fig, "fig09_cross_validation",
         "5-fold stratified cross-validation ROC-AUC on the development periods of the default split.")


def fig_same_period():
    path = SUMMARY_DIR / "random_vs_time_same_period.csv"
    if not path.exists():
        return
    same = pd.read_csv(path)
    dss = [ds for ds in DATASETS if ds in set(same["dataset"])]
    fig, axes = panels(len(dss), 3.2)
    for ax, ds in zip(axes, dss):
        g = same[same["dataset"] == ds]
        windows = sorted(g["window"].unique())
        width = 0.16
        for i, m in enumerate(MODELS):
            r = g[g["model"] == m].set_index("window").reindex(windows)
            x = np.arange(len(windows)) + (i - 2) * width
            ax.bar(x, r["diff"], width, color=MODEL_COLORS[m])
            ax.errorbar(x, r["diff"], yerr=[r["diff"] - r["diff_low"], r["diff_high"] - r["diff"]],
                        fmt="none", ecolor=INK, lw=0.8, capsize=1.5)
        ax.axhline(0, color=INK, lw=0.8)
        periods = g.drop_duplicates("window").set_index("window")["test_period"].reindex(windows)
        ax.set_xticks(range(len(windows)), [f"window {w}\ntest {str(t)[:4]}" for w, t in zip(windows, periods)])
        ax.set_title(NAMES[ds]); ax.set_ylabel("ROC-AUC: random minus time\n(same test period)")
    model_legend(fig)
    save(fig, "fig10_random_minus_time_same_period",
         "Random minus time ROC-AUC when both are scored on the SAME period (the time split's test period; "
         "random split restricted to its rows from that period), 95% bootstrap CI. Removes the effect of "
         "periods that are intrinsically easier or harder.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="paper", choices=["paper", "slide"])
    apply_style(ap.parse_args().style)
    for f in [fig_prevalence, fig_auc_ci, fig_random_vs_time, fig_roc, fig_confusion, fig_mlp,
              fig_domain, fig_events, fig_cv, fig_same_period]:
        f()
    diagrams = {
        "d01_methodology": "Methodology of the clinical TabReD benchmark (scripts/make_diagrams.py).",
        "d02_architecture": "Model pipeline shared by all datasets and splits: preprocessing fitted on training data, "
                            "five models, selection and threshold on validation, single test evaluation.",
        "d03_splits": "Time-based splits per dataset (default split and three sliding windows); each random-k split "
                      "re-splits the rows of sliding-window-k at random into the same sizes.",
    }
    (FIGURES_DIR / "captions.md").write_text(
        "\n".join(f"- **{k}**: {v}" for k, v in {**diagrams, **CAPTIONS}.items()) + "\n")


if __name__ == "__main__":
    main()

"""Collect the benchmark, CV and drift results of all datasets into results_summary/.

Run after run_benchmark.py, run_cv.py and analyze_drift.py for every dataset.
Writes CSV tables and results_summary/summary.md (all tables in Markdown, for the report).
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from clinical_tabred.evaluation import unpaired_difference  # noqa: E402
from run_benchmark import DATASETS, load_dataset  # noqa: E402

SUMMARY_DIR = ROOT / "results_summary"
MODELS = ["logreg", "random_forest", "xgboost", "lightgbm", "mlp"]
LABELS = {"logreg": "Logistic Regression", "random_forest": "Random Forest", "xgboost": "XGBoost",
          "lightgbm": "LightGBM", "mlp": "MLP"}
INFO = {
    "nhanes": ("NHANES 2005-2018", "diagnosed diabetes (DIQ010)", "interview + exam + lab"),
    "brfss": ("BRFSS 2017-2023", "diagnosed diabetes (DIABETE3/4)", "telephone survey"),
    "natality": ("Natality 2016-2024", "preterm birth (< 37 weeks)", "birth certificate"),
}
SPLITS = ["default", "sliding-window-0", "sliding-window-1", "sliding-window-2",
          "random-0", "random-1", "random-2"]


def read(ds: str, name: str) -> pd.DataFrame | None:
    path = ROOT / DATASETS[ds]["out"] / name
    return pd.read_csv(path).assign(dataset=ds) if path.exists() else None


def md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def datasets_table(present) -> pd.DataFrame:
    """Rows, features and positive rate (first -> last period) of every dataset."""
    rows = []
    for ds in present:
        data, num, cat = load_dataset(ds)
        per = data.groupby("cycle")["label"].mean().reindex(DATASETS[ds]["periods"])
        rows.append({"dataset": INFO[ds][0], "source": INFO[ds][2], "target": INFO[ds][1],
                     "rows": len(data), "features": len(num) + len(cat), "numeric": len(num),
                     "categorical": len(cat), "periods": len(per),
                     "positive_rate": f"{data['label'].mean():.1%} ({per.iloc[0]:.1%} -> {per.iloc[-1]:.1%})"})
    return pd.DataFrame(rows)


def test_auc_table(present) -> pd.DataFrame:
    """Test ROC-AUC of every model on every split (one row per dataset x model)."""
    frames = []
    for ds in present:
        m = read(ds, "metrics.csv")
        m = m[(m["split"] == "test") & (m["threshold_rule"] == "val_f1")]
        piv = m.pivot_table(index="model", columns="protocol", values="roc_auc").reindex(MODELS)
        piv = piv[[s for s in SPLITS if s in piv.columns]]
        piv.insert(0, "dataset", ds)
        frames.append(piv.reset_index())
    return pd.concat(frames, ignore_index=True)


def default_metrics_table(present) -> pd.DataFrame:
    """Default time split: all course metrics on the test periods with bootstrap CIs."""
    frames = []
    for ds in present:
        m = read(ds, "metrics.csv")
        m = m[(m["protocol"] == "default") & (m["split"] == "test") & (m["threshold_rule"] == "val_f1")]
        ci = read(ds, "default_test_ci.csv")
        t = m.merge(ci.drop(columns=["roc_auc", "pr_auc", "dataset"]), on="model")
        t["ROC-AUC [95% CI]"] = t.apply(lambda r: f"{r.roc_auc:.3f} [{r.roc_auc_low:.3f}, {r.roc_auc_high:.3f}]", axis=1)
        t["PR-AUC [95% CI]"] = t.apply(lambda r: f"{r.pr_auc:.3f} [{r.pr_auc_low:.3f}, {r.pr_auc_high:.3f}]", axis=1)
        sel = (ROOT / DATASETS[ds]["out"] / "default_selected_model.txt").read_text().strip()
        t["selected_on_val"] = t["model"] == sel
        frames.append(t[["dataset", "model", "accuracy", "precision", "recall", "f1", "ROC-AUC [95% CI]",
                         "PR-AUC [95% CI]", "threshold", "selected_on_val"]])
    return pd.concat(frames, ignore_index=True)


def paired_table(present) -> pd.DataFrame:
    paired = [read(ds, "default_paired_comparisons.csv") for ds in present]
    return pd.concat([p for p in paired if p is not None], ignore_index=True)


def random_vs_time_tables(present) -> tuple[pd.DataFrame, pd.DataFrame]:
    """TabReD comparison: per window (gaps) and mean over windows with the model ranks."""
    gaps = pd.concat([read(ds, "random_minus_time_by_window.csv") for ds in present], ignore_index=True)
    gaps = gaps[gaps["metric"] == "roc_auc"]
    mean = gaps.groupby(["dataset", "model"])[["time", "random", "diff"]].mean().reset_index()
    mean["windows_significant"] = gaps.groupby(["dataset", "model"])["ci_excludes_zero"].sum().values
    mean["rank_time"] = mean.groupby("dataset")["time"].rank(ascending=False).astype(int)
    mean["rank_random"] = mean.groupby("dataset")["random"].rank(ascending=False).astype(int)
    mean = mean.rename(columns={"diff": "random_minus_time"}).sort_values(["dataset", "rank_time"])
    return gaps, mean


def load_predictions(ds: str) -> pd.DataFrame:
    path = ROOT / DATASETS[ds]["out"] / "predictions" / "test_predictions.csv"
    if not path.exists():   # large prediction files are stored gzipped in git
        path = path.with_name(path.name + ".gz")
    return pd.read_csv(path, dtype={"cycle": str})


def same_period_table(present, n_resamples: int = 1000, confidence: float = 0.95, seed: int = 42) -> pd.DataFrame:
    """Random vs time scored on the SAME period.

    TabReD's random and time test sets contain different periods, so their difference mixes
    "the model never saw this period" with "this period is easier/harder". Here the random split
    is scored only on its rows from the time split's test period.
    """
    rows = []
    for ds in present:
        p = load_predictions(ds)
        for k, start in enumerate(DATASETS[ds]["windows"]):
            period = DATASETS[ds]["periods"][start + 4]
            for m in MODELS:
                a = p[(p["protocol"] == f"random-{k}") & (p["model"] == m) & (p["cycle"] == period)]
                b = p[(p["protocol"] == f"sliding-window-{k}") & (p["model"] == m)]
                if a.empty or b.empty:
                    continue
                r = unpaired_difference(a["label"].to_numpy(), a["probability"].to_numpy(),
                                        b["label"].to_numpy(), b["probability"].to_numpy(),
                                        roc_auc_score, n_resamples, confidence, seed)
                rows.append({"dataset": ds, "window": k, "test_period": period, "model": m,
                             "random_same_period": roc_auc_score(a["label"], a["probability"]),
                             "time": roc_auc_score(b["label"], b["probability"]),
                             "n_random_rows": len(a), **r,
                             "ci_excludes_zero": r["diff_low"] > 0 or r["diff_high"] < 0})
    return pd.DataFrame(rows)


def same_period_mean(same: pd.DataFrame) -> pd.DataFrame:
    smean = same.groupby(["dataset", "model"])[["time", "random_same_period", "diff"]].mean().reset_index()
    smean["windows_significant"] = same.groupby(["dataset", "model"])["ci_excludes_zero"].sum().values
    return smean


def improvement_table(present, gaps: pd.DataFrame) -> pd.DataFrame:
    """TabReD Table 4 style: % ROC-AUC improvement over the MLP under time and random splits."""
    rows = []
    for ds in present:
        g = gaps[gaps["dataset"] == ds]
        for kind in ["time", "random"]:
            per = g.pivot_table(index="window", columns="model", values=kind)
            rel = ((per.sub(per["mlp"], axis=0)).div(per["mlp"], axis=0) * 100).mean()
            rows.append({"dataset": ds, "splits": kind, **{LABELS[m]: rel.get(m) for m in MODELS if m != "mlp"}})
    return pd.DataFrame(rows)


def cv_tables(present) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """5-fold CV summary (numbers) and a display version with mean +/- std strings."""
    cv = [c for c in (read(ds, "cv_summary.csv") for ds in present) if c is not None]
    if not cv:
        return None, None
    t = pd.concat(cv, ignore_index=True)
    show = t[["dataset", "model"]].copy()
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        show[k] = [f"{mu:.3f} ± {sd:.3f}" for mu, sd in zip(t[k + "_mean"], t[k + "_std"])]
    return t, show


def main():
    SUMMARY_DIR.mkdir(exist_ok=True)
    sections = ["# Clinical TabReD: results summary\n",
                ("Generated by `scripts/summarize_benchmarks.py` from `results_nhanes/`, "
                 "`results_brfss/`, `results_natality/`.\n")]
    present = [ds for ds in DATASETS if (ROOT / DATASETS[ds]["out"] / "metrics.csv").exists()]

    t = datasets_table(present)
    t.to_csv(SUMMARY_DIR / "datasets.csv", index=False)
    sections += ["## Datasets\n", md(t), ""]

    t = test_auc_table(present)
    t.to_csv(SUMMARY_DIR / "test_roc_auc_by_split.csv", index=False)
    sections += ["## Test ROC-AUC per split\n", md(t), ""]

    t = default_metrics_table(present)
    t.to_csv(SUMMARY_DIR / "default_test_metrics.csv", index=False)
    sections += ["## Default time split: test metrics (threshold = max F1 on validation)\n", md(t), ""]

    paired = paired_table(present)
    paired.to_csv(SUMMARY_DIR / "default_paired_comparisons.csv", index=False)
    sections += ["## Default split: paired bootstrap differences vs the validation-selected model\n",
                 md(paired, ".4f"), ""]

    gaps, mean = random_vs_time_tables(present)
    gaps.to_csv(SUMMARY_DIR / "random_minus_time_by_window.csv", index=False)
    mean.to_csv(SUMMARY_DIR / "random_vs_time.csv", index=False)
    sections += ["## Random vs time splits (ROC-AUC, mean over the 3 windows)\n",
                 "`windows_significant`: windows where the 95% CI of random minus time excludes 0.\n",
                 md(mean, ".4f"), ""]

    same = same_period_table(present)
    same.to_csv(SUMMARY_DIR / "random_vs_time_same_period.csv", index=False)
    sections += ["## Random vs time on the SAME test period (ROC-AUC, mean over windows)\n",
                 ("Random split scored only on its rows from the time split's test period; "
                  "`diff` = random minus time, positive = the random split overestimates.\n"),
                 md(same_period_mean(same), ".4f"), "", "Per window:\n",
                 md(same.drop(columns=["diff_low", "diff_high"]), ".4f"), ""]

    t = improvement_table(present, gaps)
    t.to_csv(SUMMARY_DIR / "improvement_over_mlp.csv", index=False)
    sections += ["## Relative ROC-AUC improvement over the MLP, % (TabReD Table 4 style)\n", md(t, ".2f"), ""]

    t, show = cv_tables(present)
    if t is not None:
        t.to_csv(SUMMARY_DIR / "cv_summary.csv", index=False)
        sections += ["## 5-fold cross-validation (development periods of the default split, threshold 0.5)\n",
                     md(show), ""]

    for name, title in [("drift_performance.csv", "Drift and performance per sliding window"),
                        ("drift_domain.csv", "Domain classifier ROC-AUC (0.5 = no detectable shift)"),
                        ("drift_events.csv", "Drift with a clinical cause, measured per period")]:
        found = [(ds, p) for ds in present if (p := read(ds, name)) is not None]
        if not found:
            continue
        t = pd.concat([p for _, p in found], ignore_index=True)
        t.to_csv(SUMMARY_DIR / name, index=False)
        if name == "drift_events.csv":
            body = "\n\n".join(f"**{ds}**\n\n" + md(p.drop(columns="dataset"), ".4f") for ds, p in found)
        else:
            body = md(t)
        sections += [f"## {title}\n", body, ""]
    (SUMMARY_DIR / "summary.md").write_text("\n".join(sections))
    print("\n".join(sections))


if __name__ == "__main__":
    main()

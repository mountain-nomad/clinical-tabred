"""Generate the data part of for_presentation/ from the saved results (nothing typed by hand).

Usage: python scripts/make_presentation_pack.py     (after summarize_benchmarks.py, make_figures.py
                                                     and make_diagrams.py)
Writes:
  for_presentation/figures/      figNN_*.png + .svg       (copied from figures/)
  for_presentation/diagrams/     dNN_*.png + .svg + .md   (copied from figures/)
  for_presentation/tables/       tNN_*.csv + .md          (from results_summary/ and results_<dataset>/)
  for_presentation/charts_data/  cNN_*.csv                (raw data of every chart, to redraw natively)
  for_presentation/key_numbers.json                       (headline numbers with source file and filter)
The hand-written files (README.md, slides.md, equations.md, algorithm.md, references.md,
PLACEHOLDERS.md, style_guide.md, reference/) are not touched.
"""
import json
import platform
import shutil
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_benchmark import DATASETS  # noqa: E402
from summarize_benchmarks import LABELS, MODELS, load_predictions  # noqa: E402

PACK = ROOT / "for_presentation"
SUMMARY = ROOT / "results_summary"
NAMES = {"nhanes": "NHANES", "brfss": "BRFSS", "natality": "Natality"}


def res(ds: str, name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / DATASETS[ds]["out"] / name)


def write_table(name: str, df: pd.DataFrame, floatfmt: str = ".3f"):
    df.to_csv(PACK / "tables" / f"{name}.csv", index=False)
    (PACK / "tables" / f"{name}.md").write_text(df.to_markdown(index=False, floatfmt=floatfmt) + "\n")


def copy_figures():
    for sub in ("figures", "diagrams"):
        shutil.rmtree(PACK / sub, ignore_errors=True)
        (PACK / sub).mkdir(parents=True)
    for f in sorted((ROOT / "figures").iterdir()):
        if f.name.startswith("fig") and f.suffix in (".png", ".svg"):
            shutil.copy(f, PACK / "figures" / f.name)
        elif f.name.startswith("d0") and f.suffix in (".png", ".svg", ".md"):
            shutil.copy(f, PACK / "diagrams" / f.name)
    shutil.copy(ROOT / "figures" / "captions.md", PACK / "figures" / "captions.md")


def tables():
    shutil.rmtree(PACK / "tables", ignore_errors=True)
    (PACK / "tables").mkdir(parents=True)
    write_table("t01_datasets", pd.read_csv(SUMMARY / "datasets.csv"))

    rows = []
    for ds, spec in DATASETS.items():
        p, (n_tr, n_va) = spec["periods"], spec["default"]
        rows.append({"dataset": NAMES[ds], "default train": f"{p[0]} .. {p[n_tr - 1]}", "default validation": p[n_tr],
                     "default test": f"{p[n_tr + n_va]} .. {p[-1]}",
                     "sliding-window test periods": ", ".join(p[s + 4] for s in spec["windows"])})
    write_table("t02_splits", pd.DataFrame(rows))

    m = pd.read_csv(SUMMARY / "default_test_metrics.csv")
    m["model"] = m["model"].map(LABELS)
    write_table("t03_default_test_metrics", m)

    paired = pd.read_csv(SUMMARY / "default_paired_comparisons.csv")
    write_table("t04_paired_comparisons", paired[paired["metric"] == "roc_auc"], ".4f")

    auc = pd.read_csv(SUMMARY / "test_roc_auc_by_split.csv")
    auc["model"] = auc["model"].map(LABELS)
    write_table("t05_test_roc_auc_by_split", auc)

    rvt = pd.read_csv(SUMMARY / "random_vs_time.csv")
    rvt["model"] = rvt["model"].map(LABELS)
    write_table("t06_random_vs_time", rvt, ".4f")

    same = pd.read_csv(SUMMARY / "random_vs_time_same_period.csv")
    smean = same.groupby(["dataset", "model"], sort=False)[["time", "random_same_period", "diff"]].mean().reset_index()
    smean["windows_significant"] = same.groupby(["dataset", "model"], sort=False)["ci_excludes_zero"].sum().values
    smean["model"] = smean["model"].map(LABELS)
    write_table("t07_random_vs_time_same_period", smean, ".4f")

    write_table("t08_improvement_over_mlp", pd.read_csv(SUMMARY / "improvement_over_mlp.csv"), ".2f")

    cv = pd.read_csv(SUMMARY / "cv_summary.csv")
    show = cv[["dataset", "model"]].assign(model=cv["model"].map(LABELS))
    for k in ["roc_auc", "pr_auc", "f1", "accuracy"]:
        show[k] = [f"{mu:.3f} ± {sd:.3f}" for mu, sd in zip(cv[k + "_mean"], cv[k + "_std"])]
    write_table("t09_cross_validation", show)

    dom = pd.read_csv(SUMMARY / "drift_domain.csv")
    t = dom.groupby(["dataset", "kind"])["roc_auc"].agg(["min", "max"]).reset_index()
    t["range"] = [f"{a:.2f}-{b:.2f}" for a, b in zip(t["min"], t["max"])]
    write_table("t10_domain_classifier", t.pivot(index="dataset", columns="kind", values="range").reset_index())

    ev = pd.read_csv(SUMMARY / "drift_events.csv")
    write_table("t11_drift_events", ev[["dataset", "feature", "statistic", "cause", "change_at", "description"]])

    perf = pd.read_csv(SUMMARY / "drift_performance.csv")
    write_table("t12_drift_vs_performance", perf[["dataset", "window", "test", "domain_auc_time", "prevalence_train",
                                                  "prevalence_test", "random_minus_time_mean",
                                                  "models_gap_significant", "events_in_test_period"]])

    write_table("t13_research_questions", pd.DataFrame([
        ["Random splits overestimate future performance", "yes, in the label-shift window",
         "yes, small but consistent, in coding-revision years", "no (period-difficulty artefact)"],
        ["Random splits change the model ranking", "yes (MLP 2nd -> 4th)", "no", "partly (MLP, LR)"],
        ["GBDTs are the strongest models", "yes", "yes", "yes"],
        ["XGBoost's lead over the MLP shrinks under time splits", "no, it grows", "no (similar)", "no (same)"],
    ], columns=["TabReD claim", "NHANES (280 features)", "BRFSS (93)", "Natality (73)"]))

    pkgs = ["numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "scipy", "matplotlib"]
    write_table("t14_tech_stack", pd.DataFrame(
        [{"component": "Python", "version": platform.python_version(), "role": "language"}]
        + [{"component": p, "version": metadata.version(p),
            "role": {"numpy": "arrays", "pandas": "tables", "scikit-learn": "LR, RF, MLP, preprocessing, metrics",
                     "xgboost": "XGBoost", "lightgbm": "LightGBM", "scipy": "KS statistic",
                     "matplotlib": "figures"}[p]} for p in pkgs]
        + [{"component": "hardware", "version": "CPU only, 16 cores", "role": "all training (no GPU)"}]))


def charts():
    out = PACK / "charts_data"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    pd.concat([res(ds, "drift_periods.csv").assign(dataset=ds) for ds in DATASETS]).to_csv(
        out / "c01_prevalence.csv", index=False)
    pd.concat([res(ds, "default_test_ci.csv").assign(dataset=ds) for ds in DATASETS]).to_csv(
        out / "c02_default_auc_ci.csv", index=False)
    pd.read_csv(SUMMARY / "random_minus_time_by_window.csv").to_csv(out / "c03_random_minus_time.csv", index=False)
    roc, cms = [], []
    for ds in DATASETS:
        p = load_predictions(ds)
        p = p[p["protocol"] == "default"]
        best = (ROOT / DATASETS[ds]["out"] / "default_selected_model.txt").read_text().strip()
        thr = json.loads((ROOT / DATASETS[ds]["out"] / "selected_models.json").read_text())["default"][best]["threshold"]
        for m in MODELS:
            q = p[p["model"] == m]
            fpr, tpr, _ = roc_curve(q["label"], q["probability"])
            keep = np.unique(np.linspace(0, len(fpr) - 1, 200).astype(int))
            roc.append(pd.DataFrame({"dataset": ds, "model": m, "fpr": fpr[keep], "tpr": tpr[keep]}))
        q = p[p["model"] == best]
        pred = q["probability"] >= thr
        for t in (0, 1):
            for pr in (0, 1):
                cms.append({"dataset": ds, "model": best, "threshold": thr, "true": t, "predicted": pr,
                            "count": int(((q["label"] == t) & (pred == pr)).sum())})
    pd.concat(roc).to_csv(out / "c04_roc_points.csv", index=False)
    pd.DataFrame(cms).to_csv(out / "c05_confusion_matrices.csv", index=False)
    h = pd.concat([res(ds, "mlp_history.csv").assign(dataset=ds) for ds in DATASETS])
    h[h["protocol"] == "default"].to_csv(out / "c06_mlp_history.csv", index=False)
    pd.read_csv(SUMMARY / "drift_domain.csv").to_csv(out / "c07_domain_classifier.csv", index=False)
    pd.read_csv(SUMMARY / "drift_events.csv").to_csv(out / "c08_drift_events.csv", index=False)
    pd.concat([res(ds, "cv_folds.csv") for ds in DATASETS]).to_csv(out / "c09_cv_folds.csv", index=False)
    pd.read_csv(SUMMARY / "random_vs_time_same_period.csv").to_csv(out / "c10_random_minus_time_same_period.csv",
                                                                   index=False)


def key_numbers() -> dict:
    """Headline numbers, each with the file and filter it comes from."""
    out = {}

    def put(key, value, source, where, fmt=".3f"):
        out[key] = {"value": round(float(value), 4), "display": format(value, fmt), "source": source, "filter": where}

    d = pd.read_csv(SUMMARY / "datasets.csv")
    for ds, row in zip(DATASETS, d.itertuples()):
        put(f"{ds}_rows", row.rows, "results_summary/datasets.csv", f"dataset == {row.dataset}", ",.0f")
        put(f"{ds}_features", row.features, "results_summary/datasets.csv", f"dataset == {row.dataset}", ".0f")
    for ds in DATASETS:
        ci = res(ds, "default_test_ci.csv").set_index("model")
        best = ci["roc_auc"].idxmax()
        put(f"{ds}_best_default_roc_auc", ci.loc[best, "roc_auc"], f"{DATASETS[ds]['out']}/default_test_ci.csv",
            f"model == {best} (highest test ROC-AUC)")
        put(f"{ds}_default_roc_auc_spread", ci["roc_auc"].max() - ci["roc_auc"].min(),
            f"{DATASETS[ds]['out']}/default_test_ci.csv", "max - min over models")
    rvt = pd.read_csv(SUMMARY / "random_vs_time.csv")
    for ds in DATASETS:
        g = rvt[rvt["dataset"] == ds]
        put(f"{ds}_random_minus_time_min", g["random_minus_time"].min(), "results_summary/random_vs_time.csv",
            f"dataset == {ds}, min over models", "+.3f")
        put(f"{ds}_random_minus_time_max", g["random_minus_time"].max(), "results_summary/random_vs_time.csv",
            f"dataset == {ds}, max over models", "+.3f")
        put(f"{ds}_significant_model_windows", g["windows_significant"].sum(), "results_summary/random_vs_time.csv",
            f"dataset == {ds}, sum of windows_significant (of 15)", ".0f")
        mlp = g.set_index("model").loc["mlp"]
        put(f"{ds}_mlp_rank_time", mlp["rank_time"], "results_summary/random_vs_time.csv", "model == mlp", ".0f")
        put(f"{ds}_mlp_rank_random", mlp["rank_random"], "results_summary/random_vs_time.csv", "model == mlp", ".0f")
    imp = pd.read_csv(SUMMARY / "improvement_over_mlp.csv").set_index(["dataset", "splits"])
    for ds in DATASETS:
        for kind in ("time", "random"):
            put(f"{ds}_xgboost_over_mlp_pct_{kind}", imp.loc[(ds, kind), "XGBoost"],
                "results_summary/improvement_over_mlp.csv", f"dataset == {ds}, splits == {kind}", "+.2f")
    dom = pd.read_csv(SUMMARY / "drift_domain.csv")
    for ds in DATASETS:
        t = dom[(dom["dataset"] == ds) & (dom["kind"] == "time")]["roc_auc"]
        put(f"{ds}_domain_auc_time_min", t.min(), "results_summary/drift_domain.csv", f"dataset == {ds}, kind == time")
        put(f"{ds}_domain_auc_time_max", t.max(), "results_summary/drift_domain.csv", f"dataset == {ds}, kind == time")
    per = {ds: res(ds, "drift_periods.csv") for ds in DATASETS}
    for ds, p in per.items():
        put(f"{ds}_prevalence_first_pct", 100 * p["prevalence"].iloc[0], f"{DATASETS[ds]['out']}/drift_periods.csv",
            "first period", ".1f")
        put(f"{ds}_prevalence_last_pct", 100 * p["prevalence"].iloc[-1], f"{DATASETS[ds]['out']}/drift_periods.csv",
            "last period", ".1f")
    return out


def main():
    copy_figures()
    tables()
    charts()
    (PACK / "key_numbers.json").write_text(json.dumps(key_numbers(), indent=2) + "\n")
    print("wrote", PACK)


if __name__ == "__main__":
    main()

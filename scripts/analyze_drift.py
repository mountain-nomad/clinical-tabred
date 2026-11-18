"""Temporal drift analysis for the wide benchmark, with drift characterised by its clinical cause.

Usage: python scripts/analyze_drift.py --dataset {nhanes,brfss,natality}
Outputs (in the dataset's results directory):
  drift_periods.csv      rows and label prevalence per period
  drift_domain.csv       domain classifier ROC-AUC "train periods vs test period" for every split
                         (random splits = no-drift baseline, ~0.5 by construction) and for
                         adjacent periods
  drift_features.csv     per-feature drift, first vs last period: PSI and KS (numeric), total
                         variation distance (categorical, missing counted as a category), change
                         in missing share; and the largest year-to-year change
  drift_events.csv       documented changes with a clinical cause, measured in every period
  drift_performance.csv  per sliding window: time vs random test ROC-AUC (benchmark results),
                         domain shift and prevalence shift, and the events that hit the test period

The domain classifier is the same as the original project's (`clinical_tabred.drift`): logistic
regression with the shared preprocessing, 5-fold CV ROC-AUC. Each side is subsampled to at most
20,000 rows (fixed seed) to keep it fast.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from clinical_tabred.drift import psi  # noqa: E402
from clinical_tabred.models import make_preprocessor  # noqa: E402
from clinical_tabred.prepare import feature_matrix  # noqa: E402
from run_benchmark import DATASETS, load_dataset, make_splits  # noqa: E402

# (column, statistic, code, cause type, period of the change, description)
# statistic: "mean" of the column, "share" of rows equal to `code`, "missing" share.
EVENTS = {
    "nhanes": [
        ("label", "mean", None, "label / population shift", "steady",
         "Diagnosed-diabetes prevalence rises cycle after cycle"),
        ("HIQ011", "share", 2, "policy change", "2015-2016",
         "Not covered by health insurance: falls after the 2014 Affordable Care Act coverage expansion"),
        ("HIQ031D", "mean", None, "policy change", "2015-2016",
         "Covered by Medicaid: rises with the 2014 ACA Medicaid expansion"),
        ("BMXBMI", "mean", None, "population shift", "steady", "Mean measured BMI rises (obesity trend)"),
        ("LBXRDW", "mean", None, "protocol change (lab instrument)", "2013-2014",
         "Hematology analyzer replaced by Beckman Coulter DxH 800; CDC: mean RDW 12.84% -> 13.55% (CBC_H docs)"),
        ("LBXSLDSI", "mean", None, "protocol change (lab instrument)", "2017-2018",
         "Biochemistry analyzer and lab changed to Roche Cobas 6000; CDC bridging study: LDH +14.9% (BIOPRO_J docs)"),
        ("LBXSASSI", "mean", None, "protocol change (lab instrument)", "2017-2018",
         "Same 2017-2018 analyzer change; CDC bridging study: AST -18.0% (BIOPRO_J docs)"),
    ],
    "brfss": [
        ("label", "mean", None, "population shift", "2020",
         "Diagnosed-diabetes prevalence dips in the COVID-19 year 2020"),
        ("PERSDOC", "share", 2, "coding revision", "2021",
         "PERSDOC2 -> PERSDOC3 rewording: 'more than one personal doctor' jumps"),
        ("HLTHPLN", "share", 2, "coding revision", "2021",
         "Yes/no coverage question replaced by 'primary source of insurance' (CDC-derived series)"),
        ("_HCVU65", "share", 2, "coding revision", "2021",
         "CDC-derived 'no coverage, age 18-64' follows the same 2021 question change"),
        ("_MRACE", "share", 6, "coding revision", "2022",
         "2022 race question: 'other race only' category removed"),
        ("MSCODE", "missing", None, "survey-mode change", "steady",
         "Landline-only metro code: missing grows as the sample shifts to cell phones"),
    ],
    "natality": [
        ("label", "mean", None, "label shift", "2021",
         "Preterm-birth rate rises to a 2021 peak"),
        ("DMAR", "missing", None, "reporting change (state law)", "2017",
         "Marital status withheld for California births from 2017"),
        ("RF_GHYPE", "mean", None, "population / diagnostic practice", "steady",
         "Gestational hypertension nearly doubles"),
        ("RF_PHYPE", "mean", None, "population / diagnostic practice", "steady",
         "Pre-pregnancy hypertension doubles"),
        ("RF_GDIAB", "mean", None, "population / diagnostic practice", "steady",
         "Gestational diabetes rises"),
        ("CIG_0", "mean", None, "population shift", "steady",
         "Cigarettes per day before pregnancy fall by about two thirds"),
        ("FAGECOMB", "missing", None, "protocol change (COVID-19)", "2020",
         "Hospital COVID-19 policies: more births with unknown father information (NCHS User Guide)"),
    ],
}
MAX_ROWS = 20_000


def domain_auc(a: pd.DataFrame, b: pd.DataFrame, numeric, categorical, seed: int) -> float:
    """5-fold CV ROC-AUC of a logistic regression telling rows of `b` from rows of `a`."""
    a = a.sample(n=min(len(a), MAX_ROWS), random_state=seed)
    b = b.sample(n=min(len(b), MAX_ROWS), random_state=seed)
    both = pd.concat([a, b], ignore_index=True)
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    model = Pipeline([("prep", make_preprocessor(numeric, categorical)),
                      ("model", LogisticRegression(max_iter=2000))])
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return float(cross_val_score(model, feature_matrix(both, numeric + categorical), y, cv=folds,
                                 scoring="roc_auc").mean())


def tvd(a: pd.Series, b: pd.Series) -> float:
    """Total variation distance between two categorical distributions (missing = a category)."""
    pa = a.fillna(-1).value_counts(normalize=True)
    pb = b.fillna(-1).value_counts(normalize=True)
    return float(0.5 * pa.subtract(pb, fill_value=0).abs().sum())


def feature_drift(data, numeric, categorical, periods) -> pd.DataFrame:
    first, last = data[data["cycle"] == periods[0]], data[data["cycle"] == periods[-1]]
    rows = []
    for f in numeric + categorical:
        a, b = first[f], last[f]
        row = {"feature": f, "type": "numeric" if f in numeric else "categorical",
               "first": periods[0], "last": periods[-1],
               "missing_first": a.isna().mean(), "missing_last": b.isna().mean()}
        if f in numeric and a.notna().sum() > 50 and b.notna().sum() > 50:
            row.update(mean_first=a.mean(), mean_last=b.mean(),
                       ks=ks_2samp(a.dropna(), b.dropna()).statistic,
                       psi=psi(a.dropna().to_numpy(), b.dropna().to_numpy(), 10))
        row["tvd"] = tvd(a, b) if f in categorical else np.nan
        # largest change between consecutive periods: TVD (categorical) / PSI (numeric)
        steps = []
        for p, q in zip(periods[:-1], periods[1:]):
            x, z = data.loc[data["cycle"] == p, f], data.loc[data["cycle"] == q, f]
            if f in categorical:
                steps.append(tvd(x, z))
            elif x.notna().sum() > 50 and z.notna().sum() > 50:
                steps.append(psi(x.dropna().to_numpy(), z.dropna().to_numpy(), 10))
            else:
                steps.append(0.0)
        k = int(np.argmax(steps))
        row.update(max_step=steps[k], max_step_between=f"{periods[k]}->{periods[k + 1]}")
        rows.append(row)
    out = pd.DataFrame(rows)
    out["drift_score"] = out["psi"].fillna(out["tvd"])
    return out.sort_values("drift_score", ascending=False)


def event_table(data, events, periods) -> pd.DataFrame:
    rows = []
    for col, stat, code, cause, when, desc in events:
        if col not in data:
            continue
        values = {}
        for p in periods:
            x = data.loc[data["cycle"] == p, col]
            values[p] = (x.mean() if stat == "mean" else (x == code).mean() if stat == "share"
                         else x.isna().mean())
        rows.append({"feature": col, "statistic": stat if code is None else f"share == {code}",
                     "cause": cause, "change_at": when, "description": desc, **values})
    return pd.DataFrame(rows)


def period_table(data, periods) -> pd.DataFrame:
    """Rows and label prevalence per period."""
    return data.groupby("cycle").agg(n=("label", "size"), prevalence=("label", "mean")).reindex(periods)


def domain_table(data, splits, numeric, categorical, periods, seed: int, verbose: bool = True) -> pd.DataFrame:
    """Domain-classifier ROC-AUC: train vs test of every split, and each pair of adjacent periods."""
    dom = []
    for name, parts in splits.items():
        auc = domain_auc(parts["train"], parts["test"], numeric, categorical, seed)
        dom.append({"comparison": f"{name}: train vs test", "kind": "random" if name.startswith("random")
                    else "time", "split": name, "roc_auc": auc})
        if verbose:
            print(f"domain {name:18s} train vs test ROC-AUC {auc:.3f}", flush=True)
    for p, q in zip(periods[:-1], periods[1:]):
        auc = domain_auc(data[data["cycle"] == p], data[data["cycle"] == q], numeric, categorical, seed)
        dom.append({"comparison": f"{p} vs {q}", "kind": "adjacent", "split": "", "roc_auc": auc})
        if verbose:
            print(f"domain {p} vs {q} ROC-AUC {auc:.3f}", flush=True)
    return pd.DataFrame(dom)


def drift_performance(gaps: pd.DataFrame, splits, dom: pd.DataFrame, events: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """One row per sliding window: time vs random test ROC-AUC next to the drift measured for it."""
    periods = spec["periods"]
    gaps = gaps[gaps["metric"] == "roc_auc"]
    rows = []
    for k, start in enumerate(spec["windows"]):
        g = gaps[gaps["window"] == k]
        w = splits[f"sliding-window-{k}"]
        test_p = periods[start + 4]
        hit = events[events["change_at"].isin([test_p])]["description"].tolist() if not events.empty else []
        rows.append({"window": k, "train": f"{periods[start]}..{periods[start + 2]}",
                     "val": periods[start + 3], "test": test_p,
                     "time_roc_auc_mean": g["time"].mean(), "random_roc_auc_mean": g["random"].mean(),
                     "random_minus_time_mean": g["diff"].mean(),
                     "models_gap_significant": int(g["ci_excludes_zero"].sum()), "n_models": len(g),
                     "domain_auc_time": dom.loc[dom["split"] == f"sliding-window-{k}", "roc_auc"].item(),
                     "domain_auc_random": dom.loc[dom["split"] == f"random-{k}", "roc_auc"].item(),
                     "prevalence_train": w["train"]["label"].mean(),
                     "prevalence_test": w["test"]["label"].mean(),
                     "events_in_test_period": " | ".join(hit)})
    return pd.DataFrame(rows)


def run_drift(dataset: str, verbose: bool = True) -> dict[str, pd.DataFrame]:
    """All drift tables of one dataset; written to its results directory."""
    config = yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text())
    seed, spec = config["seed"], DATASETS[dataset]
    out = ROOT / spec["out"]
    periods = spec["periods"]
    data, numeric, categorical = load_dataset(dataset)
    tables = {"drift_periods": period_table(data, periods)}
    tables["drift_periods"].to_csv(out / "drift_periods.csv")

    splits = make_splits(data, seed, spec)
    tables["drift_domain"] = domain_table(data, splits, numeric, categorical, periods, seed, verbose)
    tables["drift_features"] = feature_drift(data, numeric, categorical, periods)
    tables["drift_events"] = event_table(data, EVENTS[dataset], periods)
    for name in ["drift_domain", "drift_features", "drift_events"]:
        tables[name].to_csv(out / f"{name}.csv", index=False)
    gaps_file = out / "random_minus_time_by_window.csv"
    if gaps_file.exists():
        tables["drift_performance"] = drift_performance(pd.read_csv(gaps_file), splits, tables["drift_domain"],
                                                        tables["drift_events"], spec)
        tables["drift_performance"].to_csv(out / "drift_performance.csv", index=False)
    return tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nhanes", choices=list(DATASETS))
    tables = run_drift(ap.parse_args().dataset)
    if "drift_performance" in tables:
        print(tables["drift_performance"].drop(columns="events_in_test_period").round(4).to_string(index=False))
    feat = tables["drift_features"]
    print("\nTop drifting features (first vs last period):")
    print(feat.head(12)[["feature", "type", "drift_score", "missing_first", "missing_last",
                         "max_step", "max_step_between"]].round(3).to_string(index=False))
    print("\nEvents:\n" + tables["drift_events"].drop(columns="description").round(4).to_string(index=False))


if __name__ == "__main__":
    main()

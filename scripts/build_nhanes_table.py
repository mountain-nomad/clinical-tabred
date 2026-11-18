"""Build one wide table (one row per eligible adult) from NHANES 2005-2018.

Input : data/raw/<COMPONENT>_<S>.xpt for cycles D..J (downloaded by scripts/download_nhanes.ps1)
Output: data/processed/nhanes_wide_2005_2018.csv  (features + SEQN, cycle, start_year, label)
        data/processed/nhanes_wide_columns.csv    (one row per feature: component, label, missing %)
        data/processed/nhanes_wide_flow.csv       (eligibility counts per cycle)

Columns more than 95% empty are dropped (TabReD HomeCredit rule); rows are never dropped for
missing values. Eligibility is the same as the original pipeline: age >= 20, has a BMX record, has a DIQ record,
DIQ010 in {1 = yes, 2 = no}.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

CYCLES = {"D": (2005, "2005-2006"), "E": (2007, "2007-2008"), "F": (2009, "2009-2010"),
          "G": (2011, "2011-2012"), "H": (2013, "2013-2014"), "I": (2015, "2015-2016"),
          "J": (2017, "2017-2018")}
COMPONENTS = ["DEMO", "DIQ", "BMX", "BPX", "BPQ", "SMQ", "ALQ", "PAQ", "MCQ", "KIQ_U", "HIQ", "HUQ",
              "DPQ", "SLQ", "WHQ", "CDQ", "OCQ", "DBQ", "HSQ", "PFQ", "DUQ", "RHQ", "TCHOL", "HDL",
              "TRIGLY", "BIOPRO", "CBC", "ALB_CR"]

# Never model inputs.
DROP_EXACT = {"SEQN", "SDDSRVYR", "RIDSTATR", "RIDEXPRG", "RIDAGEMN", "RIDEXMON",
              "LBXSGL", "LBDSGLSI",                     # serum glucose in the metabolic panel
              "LBXSOSSI",                               # serum osmolality: rises with blood glucose
              "BPQ090D", "BPQ100D",
              # "Health problem causing difficulty": answer code 16 = Diabetes (the label itself)
              "PFQ063A", "PFQ063B", "PFQ063C", "PFQ063D", "PFQ063E",
              "SMAQUEX2",                               # questionnaire mode flag (admin)
              "SMQ670"}                                 # empty in the 2005-2012 training cycles                     # cholesterol drugs: guideline treatment for
                                                        # diagnosed diabetics (consequence of the label)
DROP_PATTERNS = [r"^WT", r"^SDMV",                       # survey weights / design
                 r"^SIA", r"^FIA", r"^MIA", r"^HSAQUEX$", r"^PEASCCT", r"^DMDHR",  # interview admin
                 r"^BPAEN", r"^BPAARM", r"^BPACSZ", r"^BPXCHR$",  # BP procedure / child-only
                 r"^BMI", r"^BMD", r"^BMXRECUM$", r"^BMXHEAD$", r"^BMXSUB$", r"^BMXTRI$",  # BMX status/child
                 r"LC$",                                  # lab comment codes
                 r"^LBD.*SI$"]                            # SI-unit duplicates of lab values
LEAK_TEXT = re.compile(r"diabet|insulin|sugar|glucose|a1c|prediab", re.I)
# Not leakage: family history is known before any diagnosis of the respondent.
LEAK_TEXT_ALLOW = {"MCQ300C"}
# Advice/behaviour that is typically a CONSEQUENCE of being diagnosed (target-derived):
# "doctor told you to lose weight / exercise / reduce salt / reduce fat" and "are you now doing it".
CONSEQUENCE = re.compile(r"^MCQ36[56][A-D]$|^MCQ37[01][A-D]$")
# Exact unit duplicates of another kept column.
UNIT_DUPLICATES = {"URXUMS", "URXCRS"}
MISSING_CODES = {}
for codes, names in json.loads((ROOT / "configs" / "nhanes_missing_codes.json").read_text()).items():
    for n in names:
        MISSING_CODES[n] = [float(c) for c in codes.split(",")]


def read_xpt(path):
    reader = pd.read_sas(path, format="xport", iterator=True)
    labels = {f["name"].decode().strip(): f["label"].decode(errors="ignore").strip() for f in reader.fields}
    df = reader.read()
    reader.close()
    df.columns = [c.strip() for c in df.columns]
    return df, labels


def dropped(col):
    return col in DROP_EXACT or any(re.search(p, col) for p in DROP_PATTERNS)


def main():
    missing_files = [f"{c}_{s}.xpt" for s in CYCLES for c in COMPONENTS if not (RAW / f"{c}_{s}.xpt").exists()]
    if missing_files:
        sys.exit(f"Missing files in {RAW}: {missing_files[:10]} ... run scripts/download_nhanes.ps1 first")

    per_cycle, col_sets, labels, comp_of, flow = [], [], {}, {}, []
    for s, (year, name) in CYCLES.items():
        merged = None
        present = {}
        for comp in COMPONENTS:
            df, lab = read_xpt(RAW / f"{comp}_{s}.xpt")
            if df["SEQN"].duplicated().any():
                sys.exit(f"{comp}_{s}: SEQN is not unique (not one row per person)")
            present[comp] = set(df["SEQN"])
            for c in df.columns:
                labels.setdefault(c, lab.get(c, ""))
                comp_of.setdefault(c, comp)
            if merged is None:
                merged = df
            else:
                dup = [c for c in df.columns if c != "SEQN" and c in merged.columns]
                merged = merged.merge(df.drop(columns=dup), on="SEQN", how="left")
        n_all = len(merged)
        adult = merged["RIDAGEYR"] >= 20
        exam = merged["SEQN"].isin(present["BMX"])
        diq = merged["SEQN"].isin(present["DIQ"])
        valid = merged["DIQ010"].isin([1, 2])
        keep = adult & exam & diq & valid
        flow.append({"cycle": name, "all": n_all, "adults": int(adult.sum()), "with_exam": int((adult & exam).sum()),
                     "with_diq": int((adult & exam & diq).sum()), "eligible": int(keep.sum()),
                     "positives": int((keep & (merged["DIQ010"] == 1)).sum())})
        df = merged[keep].copy()
        df["label"] = (df["DIQ010"] == 1).astype(int)
        df["cycle"] = name
        df["start_year"] = year
        per_cycle.append(df)
        col_sets.append(set(df.columns))
        print(f"{name}: {n_all} respondents -> {keep.sum()} eligible ({flow[-1]['positives']} positives), "
              f"{df.shape[1]} columns", flush=True)

    common = set.intersection(*col_sets)
    features = sorted(c for c in common
                      if c not in {"label", "cycle", "start_year"}
                      and comp_of.get(c) != "DIQ"            # whole diabetes questionnaire excluded
                      and not dropped(c))
    leak_flag = [c for c in features if LEAK_TEXT.search(labels.get(c, "")) and c not in LEAK_TEXT_ALLOW]
    consequence = [c for c in features if CONSEQUENCE.match(c)]
    features = [c for c in features if c not in leak_flag and c not in consequence and c not in UNIT_DUPLICATES]

    wide = pd.concat([d[["SEQN", "cycle", "start_year", "label"] + features] for d in per_cycle], ignore_index=True)

    # Refused / don't know -> missing, using the codes listed in the CDC codebooks.
    n_recoded = 0
    for c in features:
        codes = MISSING_CODES.get(c.upper())
        if codes:
            mask = wide[c].isin(codes)
            n_recoded += int(mask.sum())
            wide.loc[mask, c] = float("nan")
    # Text columns (e.g. cigarette brand, UPC code) and constant columns carry no usable signal.
    text_cols = [c for c in features if wide[c].dtype == object]
    single = [c for c in features if c not in text_cols and wide[c].nunique(dropna=True) == 1]
    # "Mark all that apply" items store one code when the option was chosen and are blank otherwise:
    # turn them into 0/1 indicators (kept only if at least 1% of adults chose the option).
    indicators = [c for c in single if wide[c].notna().mean() >= 0.01]
    for c in indicators:
        wide[c] = wide[c].notna().astype(float)
    const_cols = [c for c in features if c not in text_cols and c not in indicators
                  and wide[c].nunique(dropna=True) <= 1]
    features = [c for c in features if c not in text_cols and c not in const_cols]
    # Same column rule as TabReD's HomeCredit preprocessing (preprocessing/homecredit.py):
    # drop columns that are more than 95% empty. Rows are never dropped for missing values.
    too_empty = [c for c in features if wide[c].isna().mean() > 0.95]
    features = [c for c in features if c not in too_empty]
    wide = wide[["SEQN", "cycle", "start_year", "label"] + features]
    OUT.mkdir(parents=True, exist_ok=True)
    wide.to_csv(OUT / "nhanes_wide_2005_2018.csv", index=False)

    miss = wide[features].isna().mean() * 100
    cols = pd.DataFrame({"feature": features, "component": [comp_of[c] for c in features],
                         "label": [labels.get(c, "") for c in features],
                         "missing_pct": miss.round(1).values,
                         "n_unique": [wide[c].nunique() for c in features]})
    cols.to_csv(OUT / "nhanes_wide_columns.csv", index=False)
    pd.DataFrame(flow).to_csv(OUT / "nhanes_wide_flow.csv", index=False)

    print("\nRows:", len(wide), " positives:", int(wide['label'].sum()))
    print("Columns present in all 7 cycles (after merge):", len(common))
    print("Features kept:", len(features))
    print("Dropped by diabetes-text filter:", [(c, labels.get(c)) for c in leak_flag])
    print("Dropped as consequence of diagnosis:", [(c, labels.get(c)) for c in consequence])
    print("Dropped text / constant columns:", text_cols, const_cols)
    print("Converted 'mark all that apply' items to 0/1 indicators:", len(indicators))
    print("Dropped columns more than 95% empty (TabReD HomeCredit rule):", len(too_empty))
    print("Refused/don't-know answers recoded to missing:", n_recoded)
    for t in (10, 25, 50, 80):
        print(f"Features with <= {t}% missing: {(miss <= t).sum()}")
    print("\nFeatures per component:")
    print(cols.groupby("component").size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()

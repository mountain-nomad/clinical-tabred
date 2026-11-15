"""Build one wide table (one row per sampled respondent) from CDC BRFSS (LLCP) 2017-2023.

Input : data/raw/brfss/LLCP<YEAR>XPT.zip  (read directly from the zip, nothing is extracted)
        configs/brfss_codebook.json       (value labels per year, from scripts/parse_brfss_codebooks.py)
Output: data/processed/brfss_wide_2017_2023.csv.gz  (row_id, period, year, month, label, features)
        data/processed/brfss_wide_columns.csv       (one row per feature: type, source names, label, missing %)
        data/processed/brfss_wide_flow.csv          (eligibility counts per year)

Target: "(Ever told) you have diabetes" (DIABETE3 in 2017-18, DIABETE4 from 2019, same codes):
1 = yes -> 1; 3 = no and 4 = pre-diabetes/borderline -> 0; 2 = only during pregnancy,
7 = don't know, 9 = refused -> not eligible. A uniform random sample of eligible respondents is
drawn per year (default 50,000, fixed seed): each year has ~400-450K respondents and TabReD also
subsampled its largest datasets.

Cleaning follows scripts/build_wide_table.py (NHANES):
1. Renamed questions are harmonised to one name (HARMONISE) when the codes are the same;
   income (INCOME2 -> INCOME3 in 2021, more top categories) is collapsed to the old 8 levels.
2. Don't know / not sure / refused / not asked codes -> missing, taken from the CDC codebook of
   that year and variable. "None" (88/888) -> 0 for count questions (days, drinks, children).
3. Keep only features recorded in every year (questions asked only in odd years, e.g. blood
   pressure and cholesterol, are therefore dropped), drop admin/weights/design/phone columns,
   constant and text columns, and columns more than 95% empty. Rows are never dropped.
4. Leakage removals: the diabetes and pre-diabetes modules and any feature whose label mentions
   diabetes/insulin/sugar/glucose/A1C (asked only of diabetics or only of non-diabetics, so even
   their missingness gives the label away); pneumococcal vaccination (recommended for adults with
   diabetes: a consequence of the diagnosis, like NHANES BPQ090D).
5. CDC calculated variables (age groups, BMI category, smoking status, drinks per week, ...) are
   kept and harmonised across renames, like TabReD keeps the features of production pipelines
   (correlated ones included). Unit duplicates (HTIN4, WEIGHT2, HEIGHT3) and date-coded items
   are dropped.

Usage: python scripts/build_brfss_table.py [--per-year 50000] [--seed 42]
"""
import argparse
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "brfss"
OUT = ROOT / "data" / "processed"
YEARS = list(range(2017, 2024))
TARGET = {y: ("DIABETE3" if y < 2019 else "DIABETE4") for y in YEARS}

# Canonical name -> source names (the one present in a given year is used).
HARMONISE = {
    "SEX": ["SEX", "SEX1", "SEXVAR"],
    "ADDEPEV": ["ADDEPEV2", "ADDEPEV3"],
    "ALCDAY": ["ALCDAY5", "ALCDAY4"],
    "AVEDRNK": ["AVEDRNK2", "AVEDRNK3"],
    "CHCCOPD": ["CHCCOPD1", "CHCCOPD2", "CHCCOPD3"],
    "CHCKDNY": ["CHCKIDNY", "CHCKDNY1", "CHCKDNY2"],
    "CHCOCNC": ["CHCOCNCR", "CHCOCNC1"],
    "CHCSCNC": ["CHCSCNCR", "CHCSCNC1"],
    "DRNKANY": ["DRNKANY5", "DRNKANY6"],
    "FLUSHOT": ["FLUSHOT6", "FLUSHOT7"],
    "HAVARTH": ["HAVARTH3", "HAVARTH4", "HAVARTH5"],
    "HIVTST": ["HIVTST6", "HIVTST7"],
    # 2021: the yes/no coverage question was replaced by "primary source of insurance";
    # CDC's calculated any-coverage variable continues the series (coding revision).
    "HLTHPLN": ["HLTHPLN1", "_HLTHPLN", "_HLTHPL1"],
    "INCOME": ["INCOME2", "INCOME3"],
    "MEDCOST": ["MEDCOST", "MEDCOST1"],
    "PERSDOC": ["PERSDOC2", "PERSDOC3"],
    "TETANUS": ["TETANUS", "TETANUS1"],
    # CDC calculated variables (renamed when their source question changed)
    "_AIDTST": ["_AIDTST3", "_AIDTST4"],
    "_DRDXAR": ["_DRDXAR1", "_DRDXAR2", "_DRDXAR3"],
    "_DRNKWK": ["_DRNKWEK", "_DRNKWK1", "_DRNKWK2"],
    "DROCDY": ["DROCDY3_", "DROCDY4_"],
    "_FLSHOT": ["_FLSHOT6", "_FLSHOT7"],
    "_HCVU65": ["_HCVU651", "_HCVU652", "_HCVU653"],
    "_INCOMG": ["_INCOMG", "_INCOMG1"],
    "_MRACE": ["_MRACE1", "_MRACE2"],
    "_PRACE": ["_PRACE1", "_PRACE2"],
    "_RACE": ["_RACE", "_RACE1"],
    "_RACEG2": ["_RACEG21", "_RACEG22"],
    "_RACEGR": ["_RACEGR3", "_RACEGR4"],
    "_RFBING": ["_RFBING5", "_RFBING6"],
    "_RFDRHV": ["_RFDRHV5", "_RFDRHV6", "_RFDRHV7", "_RFDRHV8"],
}
SOURCE_TO_CANON = {s: c for c, ss in HARMONISE.items() for s in ss}

ADMIN = {
    "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "DISPCODE", "SEQNO", "_PSU", "_STSTR", "_STRWT",
    "_RAWRAKE", "_WT2RAKE", "_LLCPWT", "_LLCPWT2", "_CLLCPWT", "_DUALUSE", "_DUALCOR", "QSTVER",
    "QSTLANG", "CTELENM1", "CTELNUM1", "PVTRESD1", "PVTRESD2", "PVTRESD3", "COLGHOUS", "CCLGHOUS",
    "STATERE1", "CSTATE1", "CELPHONE", "CELPHON1", "LANDLINE", "RESPSLCT", "SAFETIME", "NUMMEN",
    "NUMWOMEN", "NUMADULT", "HHADULT", "CELLFON4", "CELLFON5",
}
ADMIN_PATTERNS = [r"^LADULT", r"^CADULT", r"^LANDSEX", r"^CELLSEX", r"^COLGSEX", r"^NUMHHOL",
                  r"^NUMPHON", r"^CPDEMO",
                  # random-child module: describes a child in the household, not the respondent
                  r"^RCS", r"^CASTH", r"^_CASTHM", r"^_CHISPNC", r"^_CRACE", r"^CAGEG$"]
DIABETES_MODULE = {"DIABETE3", "DIABETE4", "DIABAGE2", "DIABAGE3", "DIABAGE4", "PREDIAB1",
                   "PREDIAB2", "PDIABTST", "INSULIN", "INSULIN1", "BLDSUGAR", "FEETCHK", "FEETCHK2",
                   "FEETCHK3", "DOCTDIAB", "CHKHEMO3", "FEETSORE", "EYEEXAM", "EYEEXAM1", "DIABEYE",
                   "DIABEYE1", "DIABEDU", "DIABEDU1"}
LEAK_TEXT = re.compile(r"diabet|insulin|sugar|glucose|a1c|hemoglobin|prediab|pre-diab", re.I)
CONSEQUENCE = {"PNEUVAC3", "PNEUVAC4", "_PNEUMO2", "_PNEUMO3"}   # vaccine recommended for diabetics
# CDC calculated variables are KEPT (they are the agency's own feature pipeline, like the production
# features TabReD keeps). Only exact duplicates in other units are dropped.
UNIT_DUPLICATES = {"HTIN4", "WEIGHT2", "HEIGHT3"}
DATE_CODED = re.compile(r"MONTH AND YEAR|MONTH/YEAR", re.I)
SCALE_100 = {"_BMI5": 100, "WTKG3": 100, "HTM4": 100,      # implied two decimals in the XPT files
             "_DRNKWK": 100, "DROCDY": 100}
MISSING_LABEL = re.compile(r"don.?t know|not sure|refused|missing|not asked", re.I)
NONE_LABEL = re.compile(r"^(none|no drinks)", re.I)
QUANTITIES = {"_DRNKWK", "DROCDY", "PHYSHLTH", "MENTHLTH", "POORHLTH", "CHILDREN", "ALCDAY", "AVEDRNK", "DRNK3GE5",
              "MAXDRNKS", "HHADULTS", "CNCRAGE", "LCSFIRST", "LCSLAST", "LCSNUMCG", "CRGVHRS1"}
MAX_CATEGORY_LEVELS = 12   # same rule as the NHANES benchmark
CATEGORICAL = {"_STATE"}   # state FIPS code: a label, not a quantity


def dropped(col: str, label: str) -> str | None:
    if col in ADMIN or any(re.search(p, col) for p in ADMIN_PATTERNS):
        return "admin"
    if col in DIABETES_MODULE or LEAK_TEXT.search(label or ""):
        return "diabetes"
    if col in CONSEQUENCE:
        return "consequence"
    if col in UNIT_DUPLICATES:
        return "unit duplicate"
    if DATE_CODED.search(label or ""):
        return "date"
    return None


def special_codes(codebook: dict, year: int, var: str):
    """(codes -> missing, codes -> 0) for one variable in one year, from the CDC codebook."""
    miss, none = set(), set()
    # If this year's codebook lacks the variable, use the nearest year that has it (same name).
    years = sorted((int(y) for y in codebook if var in codebook[y]), key=lambda y: abs(y - year))
    for value, label in (codebook[str(years[0])][var] if years else []):
        if not re.fullmatch(r"\d+", value):
            continue
        if MISSING_LABEL.search(label):
            miss.add(float(value))
        elif NONE_LABEL.search(label) and value in {"8", "88", "888", "8888"}:
            none.add(float(value))
    return miss, none


def read_year(year: int):
    with zipfile.ZipFile(RAW / f"LLCP{year}XPT.zip") as z, z.open(z.namelist()[0]) as f:
        reader = pd.read_sas(f, format="xport", iterator=True, encoding="ISO-8859-1")
        dec = lambda b: (b.decode("latin-1") if isinstance(b, bytes) else b).strip()  # noqa: E731
        labels = {dec(fd["name"]): dec(fd["label"]) for fd in reader.fields}
        df = reader.read()
        reader.close()
    df.columns = [c.strip() for c in df.columns]
    num = df.select_dtypes("number").columns
    df[num] = df[num].mask(df[num].abs() < 1e-30, 0.0)   # SAS XPORT stores 0 as ~5.4e-79
    return df, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-year", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    codebook = json.loads((ROOT / "configs" / "brfss_codebook.json").read_text())

    frames, flow, labels, sources, no_codebook, reasons = [], [], {}, {}, set(), {}
    for year in YEARS:
        df, lab = read_year(year)
        tgt = df[TARGET[year]]
        ok = tgt.isin([1, 3, 4])
        flow.append({"year": year, "respondents": len(df), "diabetes_yes": int((tgt == 1).sum()),
                     "pregnancy_only": int((tgt == 2).sum()), "dont_know_refused": int(tgt.isin([7, 9]).sum()),
                     "missing": int(tgt.isna().sum()), "eligible": int(ok.sum()),
                     "prevalence": round(float((tgt[ok] == 1).mean()), 4)})
        s = df[ok].sample(n=min(args.per_year, int(ok.sum())), random_state=args.seed).reset_index(drop=True)
        # SEQNO is unique only within a state: the respondent key is year + state + SEQNO.
        out = pd.DataFrame({"row_id": [f"B{year}-{int(st):02d}-{int(q):010d}" for st, q in zip(s["_STATE"], s["SEQNO"])],
                            "period": str(year), "year": year,
                            "month": s["IMONTH"].astype(int),   # interview month (metadata, not a feature)
                            "label": (s[TARGET[year]] == 1).astype(int)})
        # Adults in household: landline (NUMADULT) and cell-phone (HHADULT) samples ask it separately.
        hh = s["NUMADULT"].where(s["NUMADULT"].notna(), s["HHADULT"]) if "HHADULT" in s else s["NUMADULT"]
        cols = {"HHADULTS": hh.where(~hh.isin([77, 99]))}
        labels.setdefault("HHADULTS", "NUMBER OF ADULTS IN HOUSEHOLD (landline NUMADULT / cell HHADULT)")
        sources.setdefault("HHADULTS", {})[year] = "NUMADULT/HHADULT"
        for src in s.columns:
            canon = SOURCE_TO_CANON.get(src, src)
            why = dropped(src, lab.get(src, ""))
            if why:
                reasons.setdefault(why, set()).add(src)
                continue
            if canon in cols:          # two sources for one canonical name in the same year
                raise SystemExit(f"{year}: {canon} has more than one source")
            x = pd.to_numeric(s[src], errors="coerce") if s[src].dtype != object else s[src]
            if x.dtype != object:
                miss, none = special_codes(codebook, year, src)
                if not any(src in codebook[y] for y in codebook):
                    no_codebook.add(src)
                x = x.where(~x.isin(miss))
                if canon in QUANTITIES:
                    x = x.where(~x.isin(none), 0.0)
                if canon == "ALCDAY":   # 1xx = days per week, 2xx = days in past 30 -> days per 30
                    x = np.where((x >= 101) & (x <= 107), (x - 100) * 30 / 7,
                                 np.where((x >= 201) & (x <= 230), x - 200, np.where(x == 0, 0, np.nan)))
                if canon == "INCOME" and src == "INCOME3":   # 9-11 (>= $100K) -> old top code 8 (>= $75K)
                    x = x.where(x <= 8, 8.0)
                if src == "_INCOMG1":   # 2021+: 5-7 split $50K+ into three groups -> old top group 5
                    x = x.where(x <= 5, 5.0)
                if src == "_MRACE2":    # 2022: "other race only" removed, multiracial 7 -> 6, 88 = no choice
                    x = x.replace({6.0: 7.0, 88.0: np.nan})
                if src == "_PRACE2":    # 2022: 88 = no race choice given
                    x = x.replace({88.0: np.nan})
                if canon in SCALE_100:
                    x = x / SCALE_100[canon]
            cols[canon] = x
            labels.setdefault(canon, lab.get(src, ""))
            sources.setdefault(canon, {})[year] = src
        frames.append(pd.concat([out, pd.DataFrame(cols)], axis=1))
        print(f"{year}: {len(df):,} respondents, {int(ok.sum()):,} eligible "
              f"(prevalence {flow[-1]['prevalence']:.1%}), sampled {len(s):,}, {len(cols)} candidate columns",
              flush=True)

    wide = pd.concat(frames, ignore_index=True)
    meta = ["row_id", "period", "year", "month", "label"]
    features = [c for c in wide.columns if c not in meta]
    present = wide.groupby("year")[features].apply(lambda g: g.notna().mean())
    not_every_year = [f for f in features if (present[f] == 0).any() or present[f].isna().any()]
    features = [f for f in features if f not in not_every_year]
    text_cols = [f for f in features if wide[f].dtype == object]
    const = [f for f in features if f not in text_cols and wide[f].nunique(dropna=True) <= 1]
    too_empty = [f for f in features if f not in text_cols + const and wide[f].isna().mean() > 0.95]
    features = [f for f in features if f not in text_cols + const + too_empty]
    wide = wide[meta + features]
    assert not wide["row_id"].duplicated().any(), "row_id is not unique"
    for f in features:
        wide[f] = wide[f].astype(float)

    OUT.mkdir(parents=True, exist_ok=True)
    wide.to_csv(OUT / "brfss_wide_2017_2023.csv.gz", index=False)
    nuniq = {f: wide[f].nunique() for f in features}
    types = ["categorical" if f in CATEGORICAL else "numeric" if f in QUANTITIES or f in SCALE_100 or f == "_AGE80" or nuniq[f] > MAX_CATEGORY_LEVELS
             else "categorical" for f in features]
    miss = wide[features].isna().mean() * 100
    pd.DataFrame({"feature": features, "type": types,
                  "sources": [";".join(sorted(set(sources[f].values()))) for f in features],
                  "label": [labels.get(f, "") for f in features],
                  "missing_pct": miss.round(1).values, "n_unique": [nuniq[f] for f in features]}
                 ).to_csv(OUT / "brfss_wide_columns.csv", index=False)
    pd.DataFrame(flow).to_csv(OUT / "brfss_wide_flow.csv", index=False)

    print(f"\nRows: {len(wide):,}  positives: {int(wide['label'].sum()):,}  features: {len(features)} "
          f"({types.count('numeric')} numeric / {types.count('categorical')} categorical)")
    for why, cs in sorted(reasons.items()):
        print(f"Dropped as {why} ({len(cs)}):", sorted(cs))
    print(f"Dropped: not recorded in every year ({len(not_every_year)}); text {text_cols}; constant {const}; "
          f"more than 95% empty ({len(too_empty)}): {too_empty}")
    print("Kept features without a codebook entry (check):", sorted(set(features) & no_codebook))
    print(pd.DataFrame(flow).to_string(index=False))


if __name__ == "__main__":
    main()

"""Build one wide table (one row per sampled birth) from the NCHS natality files 2016-2024.

Input : data/raw/natality/Nat<YEAR>us.zip  (NCHS public-use natality files, fixed-width text)
Output: data/processed/natality_wide_2016_2024.csv.gz  (row_id, period, year, month, label, features)
        data/processed/natality_wide_columns.csv       (one row per feature: type, label, missing %)
        data/processed/natality_wide_flow.csv          (eligibility counts per year)

The zips are streamed line by line (each unzipped year is ~5 GB, so nothing is extracted to disk)
and a uniform random sample of eligible births is drawn per year with reservoir sampling
(fixed seed, so the same table is rebuilt every time).

Target: preterm birth = obstetric estimate of gestation (OEGest_Comb, the NCHS standard item since
2014) below 37 completed weeks. Eligible births: US residents (RESTATUS != 4, as in NCHS
statistics) with a stated gestational age. Rows are never dropped for missing feature values.

NCHS recodes and imputation flags of kept items are kept as well (they are part of the official
file, like TabReD keeps the features of the production pipelines, correlated ones included).
Features: ONLY items known before delivery (maternal/paternal demographics, pregnancy history,
prenatal care, pre-pregnancy body size, smoking before/early in pregnancy, risk factors and
infections in this pregnancy, payer, plurality, infant sex). Everything recorded at or after
delivery is excluded (see LEAKAGE_EXCLUDED below for the reason per item).

Field positions are from the NCHS User Guides 2016-2024 (identical for all fields used here;
the guides only differ in bridged race, which was discontinued, and MHISPX/FHISPX, which were
added; fields that are empty in any year are dropped by the "present in all years" rule).

Usage: python scripts/build_natality_table.py [--per-year 50000] [--seed 42] [--jobs 9]
"""
import argparse
import random
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "natality"
OUT = ROOT / "data" / "processed"
YEARS = list(range(2016, 2025))

# (start, end) 1-based inclusive positions, missing codes, kind, description (NCHS User Guide).
# kind: "num" = quantity, "cat" = code, "ynu" = Y/N/U checkbox (X = not applicable -> 0),
#       "flag" = NCHS imputation flag (blank = not imputed -> 0)
FIELDS = {
    "MAGE_IMPFLG": (73, 73, [], "flag", "Mother's age imputed"),
    "MAGE_REPFLG": (74, 74, [], "flag", "Reported age of mother used"),
    "MAGER":       (75, 76, [], "num", "Mother's single years of age"),
    "MAGER14":     (77, 78, [], "cat", "Mother's age recode 14"),
    "MAGER9":      (79, 79, [], "cat", "Mother's age recode 9"),
    "MBSTATE_REC": (84, 84, ["3"], "cat", "Mother's nativity (US / outside US)"),
    "RESTATUS":    (104, 104, [], "cat", "Residence status"),
    "MRACE31":     (105, 106, [], "cat", "Mother's race recode 31"),
    "MRACE6":      (107, 107, [], "cat", "Mother's race recode 6"),
    "MRACE15":     (108, 109, [], "cat", "Mother's race recode 15"),
    "MRACEIMP":    (111, 111, [], "flag", "Mother's race imputed (1 unknown, 2 other race)"),
    "MHISPX":      (112, 112, ["9"], "cat", "Mother's Hispanic origin (detail)"),
    "MHISP_R":     (115, 115, ["9"], "cat", "Mother's Hispanic origin recode"),
    "MRACEHISP":   (117, 117, ["8"], "cat", "Mother's race/Hispanic origin"),
    "DMAR":        (120, 120, ["9"], "cat", "Marital status"),
    "MAR_IMP":     (121, 121, [], "flag", "Marital status imputed"),
    "MEDUC":       (124, 124, ["9"], "cat", "Mother's education"),
    "FAGERPT_FLG": (142, 142, [], "flag", "Father's reported age used"),
    "FAGECOMB":    (147, 148, ["99"], "num", "Father's combined age"),
    "FAGEREC11":   (149, 150, ["11"], "cat", "Father's age recode 11"),
    "FRACE31":     (151, 152, ["99"], "cat", "Father's race recode 31"),
    "FRACE6":      (153, 153, ["9"], "cat", "Father's race recode 6"),
    "FRACE15":     (154, 155, ["99"], "cat", "Father's race recode 15"),
    "FHISPX":      (159, 159, ["9"], "cat", "Father's Hispanic origin (detail)"),
    "FHISP_R":     (160, 160, ["9"], "cat", "Father's Hispanic origin recode"),
    "FRACEHISP":   (162, 162, ["8", "9"], "cat", "Father's race/Hispanic origin"),
    "FEDUC":       (163, 163, ["9"], "cat", "Father's education"),
    "PRIORLIVE":   (171, 172, ["99"], "num", "Prior births now living"),
    "PRIORDEAD":   (173, 174, ["99"], "num", "Prior births now dead"),
    "PRIORTERM":   (175, 176, ["99"], "num", "Prior other terminations"),
    "LBO_REC":     (179, 179, ["9"], "cat", "Live birth order recode"),
    "TBO_REC":     (182, 182, ["9"], "cat", "Total birth order recode"),
    # 000-003 = plural delivery, 888 = not applicable (first birth/pregnancy), 999 = unknown
    "ILLB_R":      (198, 200, ["888", "999"], "num", "Interval since last live birth (months)"),
    "ILLB_R11":    (201, 202, ["99"], "cat", "Interval since last live birth recode 11 (88 = first)"),
    "ILOP_R":      (206, 208, ["888", "999"], "num", "Interval since last other pregnancy (months)"),
    "ILOP_R11":    (209, 210, ["99"], "cat", "Interval since last other pregnancy recode 11"),
    "ILP_R":       (214, 216, ["888", "999"], "num", "Interval since last pregnancy (months)"),
    "ILP_R11":     (217, 218, ["99"], "cat", "Interval since last pregnancy recode 11"),
    "PRECARE":     (224, 225, ["99"], "num", "Month prenatal care began (0 = none)"),
    "PRECARE5":    (227, 227, ["5"], "cat", "Month prenatal care began recode"),
    "WIC":         (251, 251, [], "ynu", "WIC food during pregnancy"),
    "CIG_0":       (253, 254, ["99"], "num", "Cigarettes per day, 3 months before pregnancy"),
    "CIG_1":       (255, 256, ["99"], "num", "Cigarettes per day, 1st trimester"),
    "CIG_2":       (257, 258, ["99"], "num", "Cigarettes per day, 2nd trimester"),
    "CIG0_R":      (261, 261, ["6"], "cat", "Cigarettes before pregnancy recode"),
    "CIG1_R":      (262, 262, ["6"], "cat", "Cigarettes 1st trimester recode"),
    "CIG2_R":      (263, 263, ["6"], "cat", "Cigarettes 2nd trimester recode"),
    "M_Ht_In":     (280, 281, ["99"], "num", "Mother's height (inches)"),
    "BMI":         (283, 286, ["99.9"], "num", "Mother's pre-pregnancy BMI"),
    "BMI_R":       (287, 287, ["9"], "cat", "Pre-pregnancy BMI recode"),
    "PWgt_R":      (292, 294, ["999"], "num", "Pre-pregnancy weight (pounds)"),
    "RF_PDIAB":    (313, 313, [], "ynu", "Pre-pregnancy diabetes"),
    "RF_GDIAB":    (314, 314, [], "ynu", "Gestational diabetes"),
    "RF_PHYPE":    (315, 315, [], "ynu", "Pre-pregnancy hypertension"),
    "RF_GHYPE":    (316, 316, [], "ynu", "Gestational hypertension"),
    "RF_EHYPE":    (317, 317, [], "ynu", "Hypertension eclampsia"),
    "RF_PPTERM":   (318, 318, [], "ynu", "Previous preterm birth"),
    "RF_INFTR":    (325, 325, [], "ynu", "Infertility treatment used"),
    "RF_FEDRG":    (326, 326, [], "ynu", "Fertility-enhancing drugs"),
    "RF_ARTEC":    (327, 327, [], "ynu", "Assisted reproductive technology"),
    "RF_CESAR":    (331, 331, [], "ynu", "Previous cesarean"),
    "RF_CESARN":   (332, 333, ["99"], "num", "Number of previous cesareans"),
    "NO_RISKS":    (337, 337, ["9"], "cat", "No risk factors reported"),
    "IP_GON":      (343, 343, [], "ynu", "Gonorrhea"),
    "IP_SYPH":     (344, 344, [], "ynu", "Syphilis"),
    "IP_CHLAM":    (345, 345, [], "ynu", "Chlamydia"),
    "IP_HEPB":     (346, 346, [], "ynu", "Hepatitis B"),
    "IP_HEPC":     (347, 347, [], "ynu", "Hepatitis C"),
    "NO_INFEC":    (353, 353, ["9"], "cat", "No infections reported"),
    "PAY":         (435, 435, ["9"], "cat", "Payment source for delivery (insurance)"),
    "PAY_REC":     (436, 436, ["9"], "cat", "Payment recode"),
    "DPLURAL":     (454, 454, [], "cat", "Plurality"),
    "IMP_PLUR":    (456, 456, [], "flag", "Plurality imputed"),
    "SEX":         (475, 475, [], "cat", "Sex of infant"),
    "IMP_SEX":     (476, 476, [], "flag", "Infant sex imputed"),
}
TIME = {"DOB_YY": (9, 12), "DOB_MM": (13, 14)}
RESTATUS_POS = (104, 104)
GEST_POS = (499, 500)          # OEGest_Comb: 17-47 weeks, 99 = not stated

# Not used as features (documented reason; never read into the table).
LEAKAGE_EXCLUDED = {
    "OEGest_* / COMBGEST / GESTREC* / LMPUSED / OBGEST_FLG": "gestational age = the target itself",
    "DLMP_MM / DLMP_YY": "last menstrual period date + birth date gives gestational age",
    "DBWT / BWTR12 / BWTR4": "birth weight: measured after delivery, strongly determined by gestation",
    "PREVIS / PREVIS_REC": "number of prenatal visits grows with pregnancy length (proxy of gestation)",
    "WTGAIN / WTGAIN_REC / DWgt_R": "weight gain / weight at delivery: measured at delivery, grows with gestation",
    "CIG_3 / CIG3_R / CIG_REC": "3rd-trimester smoking is edited to 'unknown' when gestation < 27 weeks (User Guide)",
    "OB_ECVS / OB_ECVF": "external cephalic version is done at >= 36-37 weeks (implies term)",
    "LD_* (induction, augmentation, steroids, antibiotics, chorioamnionitis, anesthesia)":
        "labor and delivery events; antenatal steroids are given because preterm birth is expected",
    "ME_* / RDMETH_REC / DMETH_REC": "method of delivery (at delivery)",
    "MM_*": "maternal morbidity (at/after delivery)",
    "ATTEND / MTRAN / BFACIL / BFACIL3": "attendant, transfer, place of birth (at delivery)",
    "DOB_TT / DOB_WK": "time and weekday of birth (scheduled deliveries; after the outcome)",
    "APGAR5 / APGAR10 / AB_* / CA_* / ITRAN / ILIVE / BFED / SETORDER_R": "newborn status after birth",
    "MAR_P": "paternity acknowledgement is completed after the birth",
    "MBRACE / FBRACE (bridged race)": "discontinued after 2018 (would fail the every-year rule)",
    "F_* reporting flags": "administrative: whether the state reports an item (all states use the 2003 "
                           "revised certificate since 2016)",
    "COMPGST_IMP": "gestation imputation flag (about the target)",
}


def sl(line: bytes, pos) -> bytes:
    return line[pos[0] - 1:pos[1]]


def scan_year(job):
    """Stream one year's zip; count eligibility; reservoir-sample `n` eligible records."""
    year, n, seed = job
    path = RAW / f"Nat{year}us.zip"
    rng = random.Random(seed * 10_000 + year)
    sample, k = [], 0
    c = {"year": year, "records": 0, "foreign_residents": 0, "gestation_unknown": 0,
         "eligible": 0, "preterm": 0, "wrong_year": 0}
    # Several NCHS zips use Deflate64, which Python's zipfile cannot read: stream through `unzip -p`.
    with subprocess.Popen(["unzip", "-p", str(path)], stdout=subprocess.PIPE, bufsize=1 << 20) as proc:
        for line in proc.stdout:
            c["records"] += 1
            if sl(line, TIME["DOB_YY"]) != str(year).encode():
                c["wrong_year"] += 1
            if sl(line, RESTATUS_POS) == b"4":
                c["foreign_residents"] += 1
                continue
            g = sl(line, GEST_POS)
            if not g.strip() or g == b"99":
                c["gestation_unknown"] += 1
                continue
            c["eligible"] += 1
            c["preterm"] += int(g) < 37
            k += 1
            if len(sample) < n:
                sample.append(line)
            else:
                j = rng.randrange(k)
                if j < n:
                    sample[j] = line
    if proc.returncode != 0:
        raise RuntimeError(f"unzip failed for {path} (exit {proc.returncode})")
    cols = {"year": [int(sl(r, TIME["DOB_YY"])) for r in sample],
            "month": [int(sl(r, TIME["DOB_MM"])) for r in sample],
            "gestation_weeks": [int(sl(r, GEST_POS)) for r in sample]}
    for name, (a, b, *_rest) in FIELDS.items():
        cols[name] = [r[a - 1:b].decode("latin-1").strip() for r in sample]
    df = pd.DataFrame(cols)
    c["sampled"] = len(df)
    c["sampled_preterm"] = int((df["gestation_weeks"] < 37).sum())
    print(f"{year}: {c['records']:,} records, {c['eligible']:,} eligible, "
          f"preterm {c['preterm'] / c['eligible']:.2%}; sampled {len(df):,}", flush=True)
    return df, c


def clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["year", "month"]].copy()
    out["label"] = (df["gestation_weeks"] < 37).astype(int)
    for name, (_a, _b, missing, kind, _desc) in FIELDS.items():
        s = df[name].replace("", np.nan)
        s = s.where(~s.isin(missing))
        if kind == "flag":
            s = pd.to_numeric(s, errors="coerce").fillna(0.0)
        elif kind == "ynu":
            s = s.map({"Y": 1.0, "N": 0.0, "X": 0.0})          # U (unknown) and blank -> missing
        elif name == "SEX":
            s = s.map({"M": 1.0, "F": 0.0})
        else:
            s = pd.to_numeric(s, errors="coerce")
        out[name] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-year", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=len(YEARS))
    args = ap.parse_args()
    missing = [y for y in YEARS if not (RAW / f"Nat{y}us.zip").exists()]
    if missing:
        raise SystemExit(f"Missing natality zips for {missing} in {RAW}")

    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(scan_year, [(y, args.per_year, args.seed) for y in YEARS]))
    flow = pd.DataFrame([c for _, c in results])
    flow["preterm_rate"] = (flow["preterm"] / flow["eligible"]).round(4)
    wide = pd.concat([clean(d) for d, _ in results], ignore_index=True)
    wide.insert(0, "period", wide["year"].astype(str))
    wide.insert(0, "row_id", [f"N{y}-{i:06d}" for y, i in zip(wide["year"], wide.groupby("year").cumcount())])

    features = list(FIELDS)
    # Keep only features recorded in every year (same rule as NHANES "present in all cycles").
    by_year = wide.groupby("year")[features].apply(lambda g: g.notna().mean())
    absent = [f for f in features if (by_year[f] == 0).any()]
    const = [f for f in features if f not in absent and wide[f].nunique(dropna=True) <= 1]
    too_empty = [f for f in features if f not in absent + const and wide[f].isna().mean() > 0.95]
    features = [f for f in features if f not in absent + const + too_empty]
    wide = wide[["row_id", "period", "year", "month", "label"] + features]

    OUT.mkdir(parents=True, exist_ok=True)
    wide.to_csv(OUT / "natality_wide_2016_2024.csv.gz", index=False)
    miss = wide[features].isna().mean() * 100
    kinds = {f: FIELDS[f][3] for f in features}
    pd.DataFrame({"feature": features,
                  "type": ["numeric" if kinds[f] == "num" else "categorical" for f in features],
                  "label": [FIELDS[f][4] for f in features],
                  "missing_pct": miss.round(1).values,
                  "n_unique": [wide[f].nunique() for f in features]}
                 ).to_csv(OUT / "natality_wide_columns.csv", index=False)
    flow.to_csv(OUT / "natality_wide_flow.csv", index=False)

    print(f"\nRows: {len(wide):,}  positives: {int(wide['label'].sum()):,}  features: {len(features)}")
    print("Dropped (empty in at least one year):", absent)
    print("Dropped (constant):", const, " (more than 95% empty):", too_empty)
    print("Excluded as recorded at/after delivery or target proxies:")
    for k, v in LEAKAGE_EXCLUDED.items():
        print(f"  - {k}: {v}")
    print(flow.to_string(index=False))


if __name__ == "__main__":
    main()

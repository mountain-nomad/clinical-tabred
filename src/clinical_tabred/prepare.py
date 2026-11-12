"""Turn the raw NHANES XPT files into one clean, labelled table of adults.

Steps: read XPT -> merge components on SEQN -> keep adults -> build the binary
label -> turn special codes into missing values -> record every exclusion.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("clinical_tabred")


def read_xpt(path: Path) -> pd.DataFrame:
    """Read one XPT file and clean the floating-point quirks of `read_sas`.

    NHANES stores some zeros as tiny numbers such as 5.4e-79, so values that are
    numerically zero are rounded to exactly 0.
    """
    df = pd.read_sas(path, format="xport")
    numeric = df.select_dtypes("number").columns
    df[numeric] = df[numeric].mask(df[numeric].abs() < 1e-10, 0.0)
    if not df["SEQN"].is_unique:
        raise ValueError(f"SEQN is not unique in {path}; the file may be corrupted.")
    df["SEQN"] = df["SEQN"].astype("int64")
    return df


def merge_cycle(raw_dir: Path, cycle: dict, data_cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Merge DEMO, BMX and DIQ of one cycle on SEQN (left join starting from DEMO).

    Returns the merged table and a dict with how many adults lack each component.
    """
    suffix = cycle["suffix"]
    demo = read_xpt(Path(raw_dir) / f"DEMO_{suffix}.xpt")
    bmx = read_xpt(Path(raw_dir) / f"BMX_{suffix}.xpt")
    diq = read_xpt(Path(raw_dir) / f"DIQ_{suffix}.xpt")
    demo_cols = ["SEQN", "WTINT2YR", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "DMDEDUC2", "INDFMPIR"]
    bmx_cols = ["SEQN"] + [c for c in data_cfg["numeric_features"] if c.startswith("BMX")]
    merged = demo[demo_cols].merge(bmx[bmx_cols], on="SEQN", how="left", validate="one_to_one")
    merged = merged.merge(diq[["SEQN", data_cfg["target_column"]]], on="SEQN", how="left",
                          validate="one_to_one")
    merged["has_bmx"] = merged["SEQN"].isin(bmx["SEQN"])
    merged["has_diq"] = merged["SEQN"].isin(diq["SEQN"])
    merged.insert(1, "cycle", cycle["name"])
    adults = merged[merged["RIDAGEYR"] >= data_cfg["min_age"]]
    missing = {"adults_without_bmx": int((~adults["has_bmx"]).sum()),
               "adults_without_diq": int((~adults["has_diq"]).sum())}
    return merged, missing


def map_target(codes: pd.Series, positive_code: int, negative_code: int) -> pd.Series:
    """Map DIQ010 answers to a binary label: 1 -> 1, 2 -> 0, anything else -> <NA>.

    Codes 3 (borderline), 7 (refused), 9 (don't know) and missing are excluded later,
    because they are neither a clear "yes" nor a clear "no".
    """
    codes = codes.round().astype("Int64")
    label = pd.Series(pd.NA, index=codes.index, dtype="Int64")
    label[codes == positive_code] = 1
    label[codes == negative_code] = 0
    return label


def clean_special_codes(df: pd.DataFrame, data_cfg: dict) -> pd.DataFrame:
    """Cast categorical codes to Int64 and replace refused/don't-know codes with <NA>."""
    df = df.copy()
    for column in data_cfg["categorical_features"]:
        df[column] = df[column].round().astype("Int64")
    for column, codes in data_cfg["special_missing_codes"].items():
        df[column] = df[column].mask(df[column].isin(codes))
    return df


def flow_row(step: str, cycle: str, rows: int, removed: int, reason: str) -> dict:
    """One row of the attrition (data flow) table."""
    return {"step": step, "cycle": cycle, "rows": rows, "removed": removed, "reason": reason}


def filter_cycle(merged: pd.DataFrame, cycle: str, data_cfg: dict) -> tuple[pd.DataFrame, list]:
    """Apply the eligibility filters to one cycle and log each exclusion step."""
    flow = [flow_row("1_demographics", cycle, len(merged), 0, "all DEMO respondents")]
    steps = [
        ("2_adults", merged["RIDAGEYR"] >= data_cfg["min_age"], "age below 20 years"),
        ("3_has_exam", merged["has_bmx"], "no body-measurement (BMX) record: interview only"),
        ("4_has_diq", merged["has_diq"], "no diabetes questionnaire (DIQ) record"),
        ("5_valid_label", merged["label"].notna(), "DIQ010 is 3/7/9 or missing"),
    ]
    keep = pd.Series(True, index=merged.index)
    for step, condition, reason in steps:
        before = int(keep.sum())
        keep &= condition
        flow.append(flow_row(step, cycle, int(keep.sum()), before - int(keep.sum()), reason))
    return merged[keep].copy(), flow


def build_dataset(config: dict, raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the analysis table for all cycles.

    Returns (dataset, data_flow, missing_components) where data_flow lists every
    exclusion step and missing_components counts adults lacking BMX or DIQ records.
    """
    data_cfg = config["data"]
    tables, flow, missing_rows = [], [], []
    for cycle in config["cycles"]:
        merged, missing = merge_cycle(raw_dir, cycle, data_cfg)
        merged["label"] = map_target(merged[data_cfg["target_column"]],
                                     data_cfg["positive_code"], data_cfg["negative_code"])
        merged = clean_special_codes(merged, data_cfg)
        kept, cycle_flow = filter_cycle(merged, cycle["name"], data_cfg)
        tables.append(kept)
        flow.extend(cycle_flow)
        missing_rows.append({"cycle": cycle["name"], **missing})
    features = data_cfg["numeric_features"] + data_cfg["categorical_features"]
    dataset = pd.concat(tables, ignore_index=True)[["SEQN", "cycle", "WTINT2YR"] + features
                                                    + ["label"]]
    check_prevalence(dataset, data_cfg["prevalence_bounds"])
    return dataset, pd.DataFrame(flow), pd.DataFrame(missing_rows)


def check_prevalence(dataset: pd.DataFrame, bounds: list) -> None:
    """Stop if the positive rate of any cycle is implausible (likely a target-mapping bug)."""
    prevalence = dataset.groupby("cycle")["label"].mean()
    logger.info("Positive prevalence per cycle:\n%s", prevalence.round(4).to_string())
    low, high = bounds
    bad = prevalence[(prevalence < low) | (prevalence > high)]
    if not bad.empty:
        raise ValueError(
            f"Prevalence outside [{low}, {high}] for {bad.to_dict()}. "
            "Check the DIQ010 target mapping before training any model."
        )


def missingness_table(dataset: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Percentage of missing values per feature and cycle (rows: features)."""
    return (dataset.groupby("cycle")[features].agg(lambda s: s.isna().mean() * 100).T
            .rename_axis("feature").reset_index())


def save_processed(dataset: pd.DataFrame, config: dict, processed_dir: Path) -> None:
    """Save the processed table as CSV plus a JSON schema describing every column."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(processed_dir / "nhanes_adults.csv", index=False)
    data_cfg = config["data"]
    roles = {c: "numeric_feature" for c in data_cfg["numeric_features"]}
    roles.update({c: "categorical_feature" for c in data_cfg["categorical_features"]})
    roles.update({"SEQN": "id", "cycle": "metadata", "WTINT2YR": "survey_weight_metadata",
                  "label": "target"})
    schema = {c: {"role": roles[c], "dtype": str(dataset[c].dtype)} for c in dataset.columns}
    for c in data_cfg["categorical_features"]:
        schema[c]["categories"] = sorted(int(v) for v in dataset[c].dropna().unique())
    (processed_dir / "schema.json").write_text(json.dumps(schema, indent=2))


def load_processed(processed_dir: Path, config: dict) -> pd.DataFrame:
    """Load the processed CSV with the correct dtypes (categorical codes as Int64)."""
    dataset = pd.read_csv(Path(processed_dir) / "nhanes_adults.csv")
    for column in config["data"]["categorical_features"] + ["label"]:
        dataset[column] = dataset[column].astype("Int64")
    return dataset


def subsample_per_cycle(dataset: pd.DataFrame, n_rows: int, seed: int) -> pd.DataFrame:
    """Keep at most `n_rows` random rows per cycle (used by the --quick smoke test)."""
    parts = [g.sample(n=min(n_rows, len(g)), random_state=seed)
             for _, g in dataset.groupby("cycle")]
    return pd.concat(parts).sort_index().reset_index(drop=True)


def feature_matrix(dataset: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return the model inputs as floats (pd.NA -> np.nan) so sklearn can impute them."""
    return dataset[features].astype("float64").replace({pd.NA: np.nan})

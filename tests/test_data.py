"""Tests for reading XPT files, building labels and cleaning special codes."""

import numpy as np
import pandas as pd
import pytest
from xpt_writer import write_xpt

from clinical_tabred.download import validate_xpt
from clinical_tabred.prepare import build_dataset, clean_special_codes, map_target, read_xpt


def test_read_xpt_roundtrip_and_zero_rounding(tmp_path):
    df = pd.DataFrame({"SEQN": [1.0, 2.0, 3.0], "X": [0.0, 1.5, np.nan]})
    path = tmp_path / "t.xpt"
    write_xpt(df, path)
    assert validate_xpt(path) == "ok"
    result = read_xpt(path)
    assert result["X"].iloc[0] == 0.0  # the tiny 5.4e-79 value is rounded to exactly 0
    assert result["X"].iloc[1] == 1.5
    assert np.isnan(result["X"].iloc[2])
    assert result["SEQN"].dtype == "int64"


def test_read_xpt_rejects_duplicate_seqn(tmp_path):
    path = tmp_path / "dup.xpt"
    write_xpt(pd.DataFrame({"SEQN": [1.0, 1.0], "X": [1.0, 2.0]}), path)
    with pytest.raises(ValueError, match="SEQN"):
        read_xpt(path)


def test_validate_xpt_rejects_html(tmp_path):
    path = tmp_path / "fake.xpt"
    path.write_text("<html><body>Page not found</body></html>")
    assert "HTML" in validate_xpt(path)


def test_target_mapping():
    codes = pd.Series([1.0, 2.0, 3.0, 7.0, 9.0, np.nan])
    label = map_target(codes, positive_code=1, negative_code=2)
    assert label.iloc[0] == 1
    assert label.iloc[1] == 0
    assert label.iloc[2:].isna().all()  # 3, 7, 9 and missing are excluded


def test_special_codes_become_missing(real_config):
    df = pd.DataFrame({"RIAGENDR": [1.0, 2.0, 1.0, 2.0],
                       "RIDRETH3": [1.0, 3.0, 4.0, 6.0],
                       "DMDEDUC2": [1.0, 5.0, 7.0, 9.0]})
    cleaned = clean_special_codes(df, real_config["data"])
    assert cleaned["DMDEDUC2"].tolist()[:2] == [1, 5]
    assert cleaned["DMDEDUC2"].iloc[2:].isna().all()
    assert str(cleaned["DMDEDUC2"].dtype) == "Int64"


def test_dataset_schema_is_consistent_across_cycles(test_config):
    dataset, flow, _ = build_dataset(test_config, test_config["paths"]["raw_dir"])
    data_cfg = test_config["data"]
    features = data_cfg["numeric_features"] + data_cfg["categorical_features"]
    assert set(dataset["cycle"]) == {c["name"] for c in test_config["cycles"]}
    for _, part in dataset.groupby("cycle"):
        assert list(part.columns) == ["SEQN", "cycle", "WTINT2YR"] + features + ["label"]
        assert part[data_cfg["categorical_features"]].dtypes.astype(str).eq("Int64").all()
    assert (dataset["RIDAGEYR"] >= 20).all()
    assert set(dataset["label"].unique()) == {0, 1}
    # every exclusion is recorded: the final count equals the last flow step
    last = flow.groupby("cycle").tail(1).set_index("cycle")["rows"]
    assert (dataset.groupby("cycle").size() == last).all()


def test_no_forbidden_columns_among_features(real_config):
    data_cfg = real_config["data"]
    features = data_cfg["numeric_features"] + data_cfg["categorical_features"]
    forbidden_prefixes = ("SEQN", "DIQ", "LBX", "WT", "SDMV", "cycle", "label")
    assert not [f for f in features if f.startswith(forbidden_prefixes)]
    assert data_cfg["target_column"] not in features

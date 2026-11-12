"""Shared test fixtures: a tiny synthetic NHANES-like raw folder and a matching config.

All data here is synthetic and is used only to test the code, never for results.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from xpt_writer import write_xpt

REPO = Path(__file__).resolve().parents[1]


def pytest_addoption(parser):
    parser.addoption("--run-network", action="store_true", help="run tests that need internet")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-network"):
        return
    skip = pytest.mark.skip(reason="needs internet; use --run-network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


def synthetic_cycle(suffix: str, seqn_start: int, n: int, rng) -> dict[str, pd.DataFrame]:
    """DEMO, BMX and DIQ tables for one fake cycle with the real NHANES column names."""
    seqn = np.arange(seqn_start, seqn_start + n, dtype=float)
    age = rng.integers(5, 81, n).astype(float)
    adult = age >= 20
    demo = pd.DataFrame({
        "SEQN": seqn, "RIDAGEYR": age, "RIAGENDR": rng.integers(1, 3, n).astype(float),
        "RIDRETH3": rng.choice([1, 2, 3, 4, 6, 7], n).astype(float),
        "DMDEDUC2": np.where(adult, rng.choice([1, 2, 3, 4, 5, 7, 9], n), np.nan),
        "INDFMPIR": np.where(rng.random(n) < 0.1, np.nan, rng.uniform(0, 5, n).round(2)),
        "WTINT2YR": rng.uniform(1000, 50000, n)})
    height = rng.normal(167, 10, n).round(1)
    bmi = rng.normal(28, 6, n).clip(15, 60).round(1)
    bmx = pd.DataFrame({"SEQN": seqn, "BMXHT": height, "BMXBMI": bmi,
                        "BMXWT": (bmi * (height / 100) ** 2).round(1),
                        "BMXWAIST": np.where(rng.random(n) < 0.05, np.nan, bmi * 3.4)})
    bmx = bmx[rng.random(n) > 0.05]  # some respondents were interviewed but not examined
    risk = 1 / (1 + np.exp(-(-2.3 + 0.05 * (age - 50) + 0.1 * (bmi - 28))))
    diq010 = np.where(rng.random(n) < risk, 1.0, 2.0)
    diq010[rng.random(n) < 0.03] = 3.0  # borderline
    diq010[rng.random(n) < 0.01] = 9.0  # don't know
    diq = pd.DataFrame({"SEQN": seqn, "DIQ010": diq010})
    return {"DEMO": demo, "BMX": bmx, "DIQ": diq}


@pytest.fixture(scope="session")
def synthetic_raw_dir(tmp_path_factory) -> Path:
    """A raw-data folder with nine synthetic XPT files (3 components x 3 cycles)."""
    raw_dir = tmp_path_factory.mktemp("raw")
    rng = np.random.default_rng(0)
    for i, suffix in enumerate("GHI"):
        for component, table in synthetic_cycle(suffix, 10_000 * (i + 1), 500, rng).items():
            write_xpt(table, raw_dir / f"{component}_{suffix}.xpt", f"{component}_{suffix}")
    return raw_dir


@pytest.fixture(scope="session")
def real_config() -> dict:
    """The project configuration file, loaded as a dict."""
    return yaml.safe_load((REPO / "configs" / "experiment.yaml").read_text())


@pytest.fixture()
def test_config(real_config, synthetic_raw_dir, tmp_path) -> dict:
    """The project config with all paths pointing to temporary folders."""
    config = yaml.safe_load(yaml.safe_dump(real_config))
    config["paths"] = {"raw_dir": str(synthetic_raw_dir), "processed_dir": str(tmp_path / "proc"),
                       "results_dir": str(tmp_path / "results"),
                       "figures_dir": str(tmp_path / "figures")}
    return config

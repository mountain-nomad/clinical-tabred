"""Shared fixtures. All data here is synthetic and only tests the code, never produces results."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

PERIODS = [str(y) for y in range(2016, 2025)]
SPEC = {"periods": PERIODS, "default": (6, 1), "windows": [0, 2, 4]}


@pytest.fixture
def synthetic():
    """A small wide table in the processed format (SEQN, cycle, label, 2 numeric, 1 categorical)."""
    rng = np.random.default_rng(0)
    rows = []
    for k, p in enumerate(PERIODS):
        n = 120
        x1 = rng.normal(k * 0.1, 1, n)
        x2 = rng.normal(0, 1, n)
        cat = rng.integers(1, 4, n)
        logit = 0.9 * x1 + 0.5 * (cat == 2) - 1.5
        label = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
        x2[rng.random(n) < 0.1] = np.nan
        rows.append(pd.DataFrame({"SEQN": [f"{p}-{i}" for i in range(n)], "cycle": p, "label": label,
                                  "x1": x1, "x2": x2, "cat": cat}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def settings():
    return {"numeric": ["x1", "x2"], "categorical": ["cat"], "seed": 42, "n_jobs": 1, "balanced": True,
            "max_epochs": 60, "patience": 5}

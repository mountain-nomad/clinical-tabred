import numpy as np
from analyze_drift import event_table, tvd
from conftest import PERIODS
from leakage_audit import leakage_audit

from clinical_tabred.drift import psi


def test_leakage_audit_flags_a_copy_of_the_label(synthetic):
    data = synthetic.assign(leak=synthetic["label"] * 2.0 + 1)
    audit = leakage_audit(data, ["x1", "x2", "leak"], ["cat"]).set_index("feature")
    assert audit.loc["leak", "flagged"] and audit.loc["leak", "auc"] == 1.0
    assert not audit.loc[["x1", "x2", "cat"], "flagged"].any()


def test_psi_and_tvd():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 5000), 10) < 0.01
    assert psi(a, rng.normal(1, 1, 5000), 10) > 0.25
    import pandas as pd
    assert tvd(pd.Series([1, 1, 2, 2]), pd.Series([1, 1, 2, 2])) == 0
    assert tvd(pd.Series([1, 1]), pd.Series([2, 2])) == 1


def test_event_table_measures_every_period(synthetic):
    t = event_table(synthetic, [("label", "mean", None, "label shift", "steady", "x"),
                                ("x2", "missing", None, "protocol change", "2020", "y")], PERIODS)
    assert list(t["feature"]) == ["label", "x2"] and all(p in t for p in PERIODS)

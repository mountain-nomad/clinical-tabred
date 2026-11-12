"""Tests that splits never share respondents and respect chronological order."""

import pytest

from clinical_tabred.benchmark import make_all_splits
from clinical_tabred.prepare import build_dataset


@pytest.fixture()
def dataset(test_config):
    return build_dataset(test_config, test_config["paths"]["raw_dir"])[0]


def test_no_seqn_overlap_in_any_protocol(dataset, test_config):
    runs = make_all_splits(dataset, test_config)
    assert {p for p, _, _ in runs} == {"temporal", "in_period", "random"}
    for _, _, splits in runs:
        names = list(splits)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                assert not set(splits[a]["SEQN"]) & set(splits[b]["SEQN"]), (a, b)


def test_temporal_order(dataset, test_config):
    order = [c["name"] for c in test_config["cycles"]]
    runs = {p: s for p, _, s in make_all_splits(dataset, test_config)}
    for protocol in ["temporal", "in_period"]:
        splits = runs[protocol]
        train = {order.index(c) for c in splits["train"]["cycle"]}
        val = {order.index(c) for c in splits["val"]["cycle"]}
        test = {order.index(c) for c in splits["test"]["cycle"]}
        assert max(train) < min(val) and max(val) < min(test)


def test_in_period_holdout_comes_from_training_cycle(dataset, test_config):
    runs = {p: s for p, _, s in make_all_splits(dataset, test_config)}
    splits = runs["in_period"]
    assert set(splits["test_in_period"]["cycle"]) == set(splits["train"]["cycle"])

from conftest import PERIODS, SPEC
from run_benchmark import DATASETS, make_splits


def test_default_split_is_chronological(synthetic):
    s = make_splits(synthetic, 42, SPEC)["default"]
    assert set(s["train"]["cycle"]) == set(PERIODS[:6])
    assert set(s["val"]["cycle"]) == {PERIODS[6]}
    assert set(s["test"]["cycle"]) == set(PERIODS[7:])


def test_sliding_windows_and_random_splits(synthetic):
    splits = make_splits(synthetic, 42, SPEC)
    assert len(splits) == 7
    for k, start in enumerate(SPEC["windows"]):
        w, r = splits[f"sliding-window-{k}"], splits[f"random-{k}"]
        assert set(w["train"]["cycle"]) == set(PERIODS[start:start + 3])
        assert set(w["test"]["cycle"]) == {PERIODS[start + 4]}
        # random split: same rows as the window, same part sizes, but periods mixed
        assert {p: len(r[p]) for p in r} == {p: len(w[p]) for p in w}
        pooled = lambda parts: set().union(*(set(parts[p]["SEQN"]) for p in parts))
        assert pooled(r) == pooled(w)
        assert r["test"]["cycle"].nunique() > 1


def test_real_dataset_specs_are_consistent():
    for spec in DATASETS.values():
        n_tr, n_va = spec["default"]
        assert n_tr + n_va < len(spec["periods"])
        assert max(spec["windows"]) + 5 <= len(spec["periods"])

"""Train / validation / test splits for the three evaluation protocols.

- temporal  (A):  train 2011-2012, validate 2013-2014, test 2015-2016
- in_period (A2): train on 80% of 2011-2012, test on the other 20% AND on 2015-2016
- random    (B):  pool all cycles, stratified 70/15/15 split
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def temporal_split(dataset: pd.DataFrame, temporal_cfg: dict) -> dict[str, pd.DataFrame]:
    """Split by survey cycle so that the model is always tested on a later period."""
    return {
        "train": dataset[dataset["cycle"] == temporal_cfg["train"]],
        "val": dataset[dataset["cycle"] == temporal_cfg["val"]],
        "test": dataset[dataset["cycle"] == temporal_cfg["test"]],
    }


def in_period_split(
    dataset: pd.DataFrame, temporal_cfg: dict, holdout_fraction: float, seed: int
) -> dict[str, pd.DataFrame]:
    """Protocol A2: one model, evaluated on an in-period holdout and on the future cycle.

    The training cycle is split 80/20 (stratified by label). Validation is still the
    2013-2014 cycle, exactly as in protocol A, so the only thing that differs between
    the two test sets is the time period the respondents come from.
    """
    first = dataset[dataset["cycle"] == temporal_cfg["train"]]
    train, holdout = train_test_split(
        first, test_size=holdout_fraction, stratify=first["label"], random_state=seed
    )
    return {
        "train": train,
        "val": dataset[dataset["cycle"] == temporal_cfg["val"]],
        "test_in_period": holdout,
        "test": dataset[dataset["cycle"] == temporal_cfg["test"]],
    }


def random_split(dataset: pd.DataFrame, random_cfg: dict, seed: int) -> dict[str, pd.DataFrame]:
    """Protocol B: pool all cycles and make a stratified 70/15/15 split.

    Here the test set contains people from every period, so it measures
    performance without any time gap between training and testing.
    """
    test_size = random_cfg["test"]
    train_val, test = train_test_split(
        dataset, test_size=test_size, stratify=dataset["label"], random_state=seed
    )
    val_share = random_cfg["val"] / (1.0 - test_size)  # fraction of the remaining rows
    train, val = train_test_split(
        train_val, test_size=val_share, stratify=train_val["label"], random_state=seed
    )
    return {"train": train, "val": val, "test": test}


def check_no_overlap(splits: dict[str, pd.DataFrame]) -> None:
    """Raise an error if any respondent (SEQN) appears in more than one split."""
    names = list(splits)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            shared = set(splits[first]["SEQN"]) & set(splits[second]["SEQN"])
            if shared:
                raise ValueError(f"{len(shared)} respondents appear in both {first} and {second}")


def describe_splits(splits: dict[str, pd.DataFrame], protocol: str) -> pd.DataFrame:
    """Size, number of positives and prevalence of every split (for reporting)."""
    rows = []
    for name, part in splits.items():
        rows.append({"protocol": protocol, "split": name, "n": len(part),
                     "n_pos": int(part["label"].sum()), "prevalence": part["label"].mean(),
                     "cycles": ",".join(sorted(part["cycle"].unique()))})
    return pd.DataFrame(rows)

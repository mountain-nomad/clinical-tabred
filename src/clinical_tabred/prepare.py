"""Model-input helper shared by the benchmark, CV and drift scripts."""

import numpy as np
import pandas as pd


def feature_matrix(dataset: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return the model inputs as floats (pd.NA -> np.nan) so sklearn can impute them."""
    return dataset[features].astype("float64").replace({pd.NA: np.nan})

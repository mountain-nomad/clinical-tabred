"""Drift statistic shared by scripts/analyze_drift.py."""

import numpy as np


def psi(reference: np.ndarray, current: np.ndarray, n_bins: int) -> float:
    """Population Stability Index with quantile bins fixed on the reference period.

    PSI = sum over bins of (p_cur - p_ref) * ln(p_cur / p_ref). A common rule of
    thumb: < 0.1 small change, 0.1-0.25 moderate, > 0.25 large.
    """
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    p_ref = np.histogram(reference, edges)[0] / len(reference)
    p_cur = np.histogram(current, edges)[0] / len(current)
    p_ref, p_cur = np.clip(p_ref, 1e-6, None), np.clip(p_cur, 1e-6, None)  # avoid log(0)
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))

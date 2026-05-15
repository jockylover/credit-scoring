from typing import Dict, Iterable

import numpy as np
import pandas as pd


def compute_psi(base: np.ndarray, target: np.ndarray, buckets: int = 10) -> float:
    """
    Population Stability Index between two numeric distributions.
    """
    base = np.asarray(base)
    target = np.asarray(target)
    quantiles = np.linspace(0, 1, buckets + 1)
    bins = np.unique(np.quantile(base, quantiles))
    if len(bins) < 2:
        return 0.0

    base_counts, _ = np.histogram(base, bins=bins)
    target_counts, _ = np.histogram(target, bins=bins)
    base_perc = (base_counts + 1e-6) / (len(base) + 1e-6)
    target_perc = (target_counts + 1e-6) / (len(target) + 1e-6)
    psi_values = (base_perc - target_perc) * np.log(base_perc / target_perc)
    return float(np.sum(psi_values))


def frame_psi(
    base_df: pd.DataFrame,
    target_df: pd.DataFrame,
    cols: Iterable[str],
    buckets: int = 10,
) -> Dict[str, float]:
    """PSI per selected feature."""
    scores = {}
    for col in cols:
        scores[col] = compute_psi(base_df[col].dropna().values, target_df[col].dropna().values, buckets=buckets)
    return scores


def time_slice(df: pd.DataFrame, time_col: str = "Month_idx", split: int = 6) -> Dict[str, pd.DataFrame]:
    """
    Split dataframe into early vs late slices to approximate temporal drift checks.
    """
    early = df[df[time_col] <= split]
    late = df[df[time_col] > split]
    return {"early": early, "late": late}

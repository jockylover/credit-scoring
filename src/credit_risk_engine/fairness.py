from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .constants import AGE_BUCKETS, AGE_BUCKET_LABELS, DEFAULT_PD_THRESHOLDS


def age_to_bucket(age: float) -> str:
    if pd.isna(age):
        return "Unknown"
    for upper, label in zip(AGE_BUCKETS[1:], AGE_BUCKET_LABELS):
        if age < upper:
            return label
    return AGE_BUCKET_LABELS[-1]


def group_fairness_metrics(
    df: pd.DataFrame,
    pd_scores: np.ndarray,
    group_col: str = "Age",
    thresholds: Iterable[float] = DEFAULT_PD_THRESHOLDS,
) -> pd.DataFrame:
    """
    Compute group-level PD means and approval/rejection rates for a set of thresholds.
    """
    temp = df.copy()
    temp["pd_score"] = pd_scores
    if group_col == "Age":
        temp["group"] = temp[group_col].apply(age_to_bucket)
    else:
        temp["group"] = temp[group_col]

    rows = []
    for th in thresholds:
        temp["decision"] = (temp["pd_score"] >= th).astype(int)  # 1 = reject
        grp = temp.groupby("group")
        for group, data in grp:
            rows.append(
                {
                    "group": group,
                    "threshold": th,
                    "n": len(data),
                    "mean_pd": data["pd_score"].mean(),
                    "reject_rate": data["decision"].mean(),
                    "approve_rate": 1 - data["decision"].mean(),
                }
            )
    return pd.DataFrame(rows)

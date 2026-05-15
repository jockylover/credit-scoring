from typing import Dict, Iterable

import numpy as np
import pandas as pd


def age_to_bucket(age: float) -> str:
    if pd.isna(age):
        return "Unknown"
    if age < 25:
        return "<25"
    if age < 35:
        return "25-34"
    if age < 45:
        return "35-44"
    if age < 55:
        return "45-54"
    return "55+"


def group_fairness_metrics(
    df: pd.DataFrame,
    pd_scores: np.ndarray,
    group_col: str = "Age",
    thresholds: Iterable[float] = (0.3, 0.4, 0.5, 0.6),
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

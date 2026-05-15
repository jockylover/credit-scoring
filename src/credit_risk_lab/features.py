from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time ratios that are useful in credit scoring."""
    df = df.copy()
    # Defensive numeric coercion to avoid string artifacts from raw data.
    for col in ["Outstanding_Debt", "Annual_Income", "Monthly_Inhand_Salary", "Total_EMI_per_month"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        df["emi_to_income"] = df["Total_EMI_per_month"] / df["Monthly_Inhand_Salary"]
        df["debt_to_income"] = df["Outstanding_Debt"] / (df["Annual_Income"] / 12.0)
        df["balance_to_income"] = df["Monthly_Balance"] / df["Monthly_Inhand_Salary"]
        df["balance_to_limit_change"] = df["Monthly_Balance"] / df["Changed_Credit_Limit"]
    return df


def _rolling_feature(group: pd.Series, window: int, fn: str) -> pd.Series:
    return getattr(group.rolling(window=window, min_periods=1), fn)()


def add_behavioral_rollups(
    df: pd.DataFrame,
    windows: Sequence[int] = (3, 6),
    volatility_cols: Iterable[str] = ("Credit_Utilization_Ratio", "Monthly_Balance"),
) -> pd.DataFrame:
    """
    Add rolling delinquency/utilization dynamics per customer to mimic behavior features.
    """
    df = df.sort_values(["Customer_ID", "Month_idx"]).copy()
    # Ensure numeric for rolling calculations.
    for col in ["Num_of_Delayed_Payment", "Delay_from_due_date"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    grp = df.groupby("Customer_ID", group_keys=False)

    for w in windows:
        df[f"delinq_avg_w{w}"] = grp["Num_of_Delayed_Payment"].transform(
            lambda x: _rolling_feature(x, w, "mean")
        )
        df[f"delay_days_avg_w{w}"] = grp["Delay_from_due_date"].transform(
            lambda x: _rolling_feature(x, w, "mean")
        )
        df[f"delay_gt15_rate_w{w}"] = grp["Delay_from_due_date"].transform(
            lambda x: (x > 15).rolling(window=w, min_periods=1).mean()
        )

    for col in volatility_cols:
        for w in (3, 6):
            df[f"{col}_vol_w{w}"] = grp[col].transform(lambda x: _rolling_feature(x, w, "std"))

    # Short-term momentum signals.
    df["util_momentum_1m"] = grp["Credit_Utilization_Ratio"].transform(lambda x: x.diff())
    df["balance_momentum_1m"] = grp["Monthly_Balance"].transform(lambda x: x.diff())
    return df


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Domain-inspired categorical buckets."""
    df = df.copy()
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["age_bucket"] = pd.cut(
        df["Age"],
        bins=[0, 25, 35, 45, 55, 120],
        labels=["<25", "25-35", "35-45", "45-55", "55+"],
        include_lowest=True,
    )
    df["income_bucket"] = pd.qcut(df["Annual_Income"], q=5, duplicates="drop", labels=False)
    df["debt_heavy_flag"] = (df["debt_to_income"] > 8).astype(int)
    df["high_util_flag"] = (df["Credit_Utilization_Ratio"] > 0.9).astype(int)
    return df


def prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature pass to create a point-in-time modeling frame.
    Assumes Month_idx exists for temporal ordering.
    """
    enriched = add_ratio_features(df)
    enriched = add_behavioral_rollups(enriched)
    enriched = add_flags(enriched)
    return enriched


def snapshot_panel(df: pd.DataFrame, label_col: str = "Credit_Score") -> pd.DataFrame:
    """
    Collapse to latest record per customer while carrying engineered features.
    """
    if "Customer_ID" not in df.columns or "Month_idx" not in df.columns:
        raise ValueError("Customer_ID and Month_idx are required.")
    temp = df.sort_values(["Customer_ID", "Month_idx"])
    last = temp.groupby("Customer_ID").tail(1).copy()
    if label_col not in last.columns:
        return last
    return last

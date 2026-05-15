import re
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# Mapping month name to numeric index to enable temporal ordering.
MONTH_MAP = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


def _parse_credit_history_age(value: str) -> Optional[int]:
    """
    Parse strings like '22 Years and 9 Months' into total months.
    Returns None when parsing fails.
    """
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    match = re.search(r"(?P<years>\\d+)\\s*Years?\\s*(?:and\\s*(?P<months>\\d+)\\s*Months?)?", str(value))
    if not match:
        return None
    years = int(match.group("years"))
    months = int(match.group("months") or 0)
    return years * 12 + months


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """Append Month_idx to allow sorting/rolling by calendar order."""
    df = df.copy()
    df["Month_idx"] = df["Month"].map(MONTH_MAP)
    return df


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Light cleaning:
    - strip whitespace
    - coerce numeric-like object columns
    - parse credit history age into months
    - normalize loan type string and Payment_of_Min_Amount marker
    """
    df = df.copy()
    # Normalize string columns.
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    # Coerce mixed-type numeric columns.
    numeric_like = [
        "Num_of_Loan",
        "Num_Credit_Inquiries",
        "Num_Bank_Accounts",
        "Num_Credit_Card",
        "Interest_Rate",
        "Delay_from_due_date",
        "Num_of_Delayed_Payment",
        "Changed_Credit_Limit",
        "Outstanding_Debt",
        "Credit_Utilization_Ratio",
        "Amount_invested_monthly",
        "Monthly_Balance",
        "Annual_Income",
        "Monthly_Inhand_Salary",
        "Total_EMI_per_month",
        "Age",
    ]
    df = _coerce_numeric(df, numeric_like)

    # Parse credit history age.
    if "Credit_History_Age" in df.columns:
        df["Credit_History_Age_Months"] = pd.to_numeric(
            df["Credit_History_Age"].apply(_parse_credit_history_age), errors="coerce"
        )

    # Handle Type_of_Loan: count number of distinct tokens as a signal of loan mix.
    if "Type_of_Loan" in df.columns:
        df["Type_of_Loan"] = df["Type_of_Loan"].replace({"nan": np.nan})
        df["loan_token_count"] = (
            df["Type_of_Loan"]
            .fillna("")
            .apply(lambda x: len({t.strip() for t in str(x).split(" and ") if t.strip()}))
        )

    # Normalize Payment_of_Min_Amount to stable categories.
    if "Payment_of_Min_Amount" in df.columns:
        df["Payment_of_Min_Amount"] = (
            df["Payment_of_Min_Amount"]
            .replace({"NM": "No", "nan": np.nan})
            .fillna("Unknown")
        )

    # Basic handling for Payment_Behaviour anomalies.
    if "Payment_Behaviour" in df.columns:
        df["Payment_Behaviour"] = df["Payment_Behaviour"].replace({"!@9#%8": "Unknown"})

    # Month index for temporal features.
    if "Month" in df.columns and "Month_idx" not in df.columns:
        df = add_time_index(df)

    return df


def latest_snapshot(df: pd.DataFrame, label_col: str = "Credit_Score") -> pd.DataFrame:
    """
    Collapse panel data to the latest month per customer to approximate an application snapshot.
    Keeps the label if present.
    """
    if "Customer_ID" not in df.columns or "Month_idx" not in df.columns:
        raise ValueError("DataFrame must include Customer_ID and Month_idx columns.")
    sort_cols = ["Customer_ID", "Month_idx"]
    temp = df.sort_values(sort_cols)
    last = temp.groupby("Customer_ID").tail(1).copy()
    cols = [c for c in last.columns if c != "Month"]
    return last[cols]


def load_dataset(path: str) -> pd.DataFrame:
    """Read CSV with low_memory=False to avoid dtype warnings."""
    return pd.read_csv(path, low_memory=False)

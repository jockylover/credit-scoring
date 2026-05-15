import pathlib
import sys
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Add src to path for local runs.
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from credit_risk_lab import data_prep, explain, features, model as credit_model, stress


st.set_page_config(page_title="Credit Risk Analytics Workbench", layout="wide")


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


@st.cache_data
def load_engineered(path: str, nrows: Optional[int] = None, snapshot: bool = True) -> pd.DataFrame:
    raw = pd.read_csv(path, nrows=nrows, low_memory=False)
    clean = data_prep.clean_raw(raw)
    feat = features.prepare_feature_frame(clean)
    if snapshot:
        feat = features.snapshot_panel(feat, label_col="Credit_Score")
    return feat


def pick_record(df: pd.DataFrame) -> pd.Series:
    customer_id = st.selectbox("Select customer (latest snapshot)", df["Customer_ID"].unique())
    record = df[df["Customer_ID"] == customer_id].iloc[0]
    return record


def decision_label(pd_score: float, review_band: Tuple[float, float] = (0.3, 0.6)) -> str:
    if pd_score < review_band[0]:
        return "Approve"
    if pd_score < review_band[1]:
        return "Review"
    return "Reject"


def show_shap(model, record_df: pd.DataFrame):
    poor_idx = credit_model.get_poor_class_index(model)
    shap_values, feature_names = explain.compute_shap(
        model,
        record_df,
        sample_size=len(record_df),
        class_index=poor_idx,
    )
    shap_row = shap_values[0]
    contributions = pd.Series(shap_row, index=feature_names).sort_values()
    st.subheader("Top positive/negative contributions (Poor class)")
    st.write("Top Negative (lowers PD)")
    st.table(contributions.head(3).reset_index().rename(columns={"index": "feature", 0: "impact"}))
    st.write("Top Positive (raises PD)")
    st.table(contributions.tail(3).reset_index().rename(columns={"index": "feature", 0: "impact"}))


st.title("Credit Risk Analytics Workbench")
st.markdown("Single-case scoring + multiclass risk view + scenario stress")

with st.sidebar:
    model_path = st.text_input("Model path (joblib)", "artifacts/model.joblib")
    data_path = st.text_input("Data path (train.csv)", "train.csv")
    load_btn = st.button("Load model & data")

if load_btn:
    model_file = pathlib.Path(model_path)
    data_file = pathlib.Path(data_path)
    if not model_file.exists():
        st.error(f"Model file not found: {model_file}")
        st.stop()
    if not data_file.exists():
        st.error(f"Data file not found: {data_file}")
        st.stop()

    loaded_model = load_model(str(model_file))
    df = load_engineered(str(data_file), snapshot=True)

    st.success("Loaded")
    record = pick_record(df)
    record_df = record.to_frame().T.drop(columns=["Credit_Score"], errors="ignore")

    pred_label = str(credit_model.predict_credit_label(loaded_model, record_df)[0])
    class_labels = getattr(loaded_model, "class_labels_", np.array(["class_0", "class_1"]))
    class_proba = loaded_model.predict_proba(record_df)[0]
    proba_table = pd.DataFrame({"credit_class": class_labels, "probability": class_proba}).sort_values(
        "probability", ascending=False
    )

    pd_score = float(credit_model.predict_pd(loaded_model, record_df)[0])
    st.metric("Predicted Credit Rating", pred_label)
    st.metric("PD (probability of default)", f"{pd_score:.3f}", delta=None)
    st.write("Suggested decision:", decision_label(pd_score))
    st.subheader("Class probability breakdown")
    st.table(proba_table.reset_index(drop=True))

    # Scenario analysis on the single account treated as mini-portfolio.
    scenario = st.selectbox("Select scenario", list(stress.SCENARIOS.keys()))
    params = stress.SCENARIOS[scenario]
    stressed_pd = stress.adjust_pd(np.array([pd_score]), multiplier=params["multiplier"], logit_shift=params["logit_shift"])[0]
    st.metric("Stressed PD", f"{stressed_pd:.3f}")

    try:
        show_shap(loaded_model, record_df)
    except Exception as exc:  # pragma: no cover - visualization convenience
        st.warning(f"SHAP computation failed: {exc}")

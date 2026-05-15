from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import shap


def compute_shap(
    model,
    X: pd.DataFrame,
    sample_size: int = 2000,
    random_state: int = 42,
    class_index: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute SHAP values using the model's preprocessor + tree estimator.
    For multiclass models, pass class_index to select one class contribution matrix.
    Returns (shap_values_2d, feature_names).
    """
    booster = model.named_steps["model"]
    preprocessor = model.named_steps["preprocess"]
    if not hasattr(booster, "get_booster"):
        raise ValueError("SHAP TreeExplainer is only supported for tree-based XGBoost models in this project.")
    if len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=random_state)
    else:
        X_sample = X.copy()

    X_proc = preprocessor.transform(X_sample)
    feature_names = preprocessor.get_feature_names_out()
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_proc)
    values = np.asarray(shap_values)
    if isinstance(shap_values, list):
        idx = class_index or 0
        values = np.asarray(shap_values[idx])
    elif values.ndim == 3:
        idx = class_index or 0
        values = values[:, :, idx]
    return values, feature_names


def archetype_profiles(
    shap_values: np.ndarray,
    feature_names: np.ndarray,
    y_scores: np.ndarray,
    high_pct: float = 0.2,
    low_pct: float = 0.2,
) -> Dict[str, pd.Series]:
    """Summarize average absolute SHAP contributions for risk archetypes.

    Splits the population by ``y_scores`` (typically the predicted PD) and
    averages ``|shap_value|`` per feature within each cohort:

    - ``high_pct`` (default 0.2): top fraction by ``y_scores`` -> "high_risk"
    - ``low_pct``  (default 0.2): bottom fraction by ``y_scores`` -> "low_risk"

    Useful for narrative explanations such as "which features dominate the
    PD signal for the riskiest 20% of customers".
    """
    order = np.argsort(y_scores)
    n = len(y_scores)
    high_idx = order[int((1 - high_pct) * n) :]
    low_idx = order[: int(low_pct * n)]

    high_contrib = np.abs(shap_values[high_idx]).mean(axis=0)
    low_contrib = np.abs(shap_values[low_idx]).mean(axis=0)

    return {
        "high_risk": pd.Series(high_contrib, index=feature_names).sort_values(ascending=False),
        "low_risk": pd.Series(low_contrib, index=feature_names).sort_values(ascending=False),
    }

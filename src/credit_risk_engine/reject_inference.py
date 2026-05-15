from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from .model import predict_pd


def pseudo_label_rejects(
    model,
    reject_df: pd.DataFrame,
    low: float = 0.3,
    high: float = 0.7,
    assumed_bad_rate: float = 0.5,
    uncertain_weight: float = 0.3,
) -> pd.DataFrame:
    """
    Simple reject inference via pseudo labeling + sensitivity.
    - PD <= low   -> good (0)
    - PD >= high  -> bad (1)
    - low < PD < high -> sampled using assumed_bad_rate with down-weight
    """
    df = reject_df.copy()
    pd_score = predict_pd(model, df)
    df["pd_estimate"] = pd_score

    pseudo = np.full(len(df), np.nan)
    pseudo[pd_score <= low] = 0
    pseudo[pd_score >= high] = 1
    uncertain_mask = np.isnan(pseudo)

    rng = np.random.default_rng(42)
    pseudo[uncertain_mask] = rng.binomial(1, assumed_bad_rate, size=uncertain_mask.sum())

    df["pseudo_label"] = pseudo
    weights = np.ones(len(df))
    weights[uncertain_mask] = uncertain_weight
    df["sample_weight"] = weights
    return df


def build_sensitivity_grid(
    model,
    reject_df: pd.DataFrame,
    bad_rate_grid: Iterable[float] = (0.2, 0.35, 0.5, 0.65, 0.8),
    low: float = 0.3,
    high: float = 0.7,
    uncertain_weight: float = 0.3,
) -> List[Tuple[float, pd.DataFrame]]:
    """
    Produce multiple pseudo-labeled datasets under different assumed bad rates
    to test stability of the model to reject inference assumptions.
    """
    outputs = []
    for br in bad_rate_grid:
        inferred = pseudo_label_rejects(
            model=model,
            reject_df=reject_df,
            low=low,
            high=high,
            assumed_bad_rate=br,
            uncertain_weight=uncertain_weight,
        )
        outputs.append((br, inferred))
    return outputs

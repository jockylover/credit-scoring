from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import numpy as np
import pandas as pd

from .constants import DEFAULT_LGD, DEFAULT_N_SIMS


SCENARIOS: Mapping[str, Dict[str, float]] = {
    "baseline": {"multiplier": 1.0, "logit_shift": 0.0},
    "mild_recession": {"multiplier": 1.2, "logit_shift": 0.35},
    "severe_stress": {"multiplier": 1.5, "logit_shift": 0.9},
}


def _logistic_shift(p: np.ndarray, shift: float) -> np.ndarray:
    logit = np.log(p / (1 - p + 1e-9) + 1e-9)
    shifted = logit + shift
    return 1 / (1 + np.exp(-shifted))


def adjust_pd(pd_series: np.ndarray, multiplier: float = 1.0, logit_shift: float = 0.0) -> np.ndarray:
    """Apply a stress adjustment to a vector of probabilities of default.

    Two adjustment styles, both clipped to ``[1e-5, 0.999]``:

    - ``multiplier`` (default 1.0): naive proportional scaling. Useful for
      simulating "the whole portfolio gets X% riskier" assumptions.
    - ``logit_shift`` (default 0.0): adds the shift on the log-odds scale,
      preserving the bounded shape near 0 and 1. Closer to how regulators
      describe macro stress (e.g. a +0.5 logit shift roughly doubles PD
      near the population mean while leaving extreme PDs intact).

    When ``logit_shift != 0`` it takes precedence over ``multiplier``.
    """
    adjusted = pd_series * multiplier
    if logit_shift != 0:
        adjusted = _logistic_shift(pd_series, logit_shift)
    return np.clip(adjusted, 1e-5, 0.999)


@dataclass
class PortfolioResult:
    scenario: str
    expected_loss: float
    unexpected_loss: float
    var_95: float
    var_99: float
    es_95: float
    default_rate_mean: float


def simulate_losses(
    pd_series: np.ndarray,
    ead: np.ndarray,
    lgd: float = DEFAULT_LGD,
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = 42,
) -> np.ndarray:
    """
    Monte Carlo defaults: Bernoulli draws per exposure with provided PD.
    Returns portfolio loss array (shape n_sims,).
    """
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(n_sims):
        defaults = rng.binomial(1, pd_series)
        loss = (defaults * ead * lgd).sum()
        losses.append(loss)
    return np.array(losses)


def summarize_losses(losses: np.ndarray) -> Dict[str, float]:
    var_95 = float(np.quantile(losses, 0.95))
    var_99 = float(np.quantile(losses, 0.99))
    es_95 = float(losses[losses >= var_95].mean()) if np.any(losses >= var_95) else 0.0
    return {
        "expected_loss": float(np.mean(losses)),
        "unexpected_loss": float(np.std(losses)),
        "var_95": var_95,
        "var_99": var_99,
        "es_95": es_95,
    }


def run_scenarios(
    pd_series: np.ndarray,
    ead: np.ndarray,
    scenarios: Mapping[str, Dict[str, float]] = SCENARIOS,
    lgd: float = DEFAULT_LGD,
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = 42,
) -> pd.DataFrame:
    """Apply stress scenarios to PDs and simulate portfolio losses."""
    results = []
    for name, params in scenarios.items():
        adj_pd = adjust_pd(pd_series, multiplier=params.get("multiplier", 1.0), logit_shift=params.get("logit_shift", 0.0))
        losses = simulate_losses(adj_pd, ead=ead, lgd=lgd, n_sims=n_sims, seed=seed)
        summary = summarize_losses(losses)
        results.append(
            PortfolioResult(
                scenario=name,
                expected_loss=summary["expected_loss"],
                unexpected_loss=summary["unexpected_loss"],
                var_95=summary["var_95"],
                var_99=summary["var_99"],
                es_95=summary["es_95"],
                default_rate_mean=float(adj_pd.mean()),
            )
        )
    return pd.DataFrame([r.__dict__ for r in results])

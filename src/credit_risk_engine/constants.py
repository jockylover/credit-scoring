"""Shared constants used across the credit risk engine."""
from typing import Final, List, Tuple

AGE_BUCKETS: Final[List[int]] = [0, 25, 35, 45, 55, 120]
AGE_BUCKET_LABELS: Final[List[str]] = ["<25", "25-35", "35-45", "45-55", "55+"]

DEFAULT_PSI_BUCKETS: Final[int] = 10

DEFAULT_LGD: Final[float] = 0.45
DEFAULT_N_SIMS: Final[int] = 3000

DEFAULT_PD_THRESHOLDS: Final[Tuple[float, ...]] = (0.3, 0.4, 0.5, 0.6)

RISK_LABEL_POOR: Final[str] = "Poor"

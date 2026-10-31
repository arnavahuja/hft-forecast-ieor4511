"""Models: arrays in, arrays out.

Importing the package registers the shipped models; user models register themselves through
the usual plugin discovery walk.
"""

from hft_forecast.ml.models import (  # noqa: F401  - registration
    baselines,
    gbm,
    h30s_lgbm,
    torch_seq,
)
from hft_forecast.ml.models.registry import MODELS, register_model

__all__ = ["MODELS", "register_model"]

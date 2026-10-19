"""Models: arrays in, arrays out.

Importing the package registers the shipped models; user models register themselves through
the usual plugin discovery walk.
"""

from hft_forecast.ml.models import baselines, gbm, torch_seq  # noqa: F401  - registration
from hft_forecast.ml.models.registry import MODELS, register_model

__all__ = ["MODELS", "register_model"]

"""Feature sets: bar frame in, ``f_*`` columns out.

The shipped sets are imported here so the registry is populated by importing the package;
user plugins register themselves through the usual discovery walk.
"""

from hft_forecast.ml.features import basic, book  # noqa: F401  - registration side effect
from hft_forecast.ml.features.registry import (
    FEATURE_SETS,
    check_feature_lookahead,
    feature_columns,
    register_feature_set,
)

__all__ = [
    "FEATURE_SETS",
    "check_feature_lookahead",
    "feature_columns",
    "register_feature_set",
]

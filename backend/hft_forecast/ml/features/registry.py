"""The feature-set registry and the causality check every set must pass."""

from __future__ import annotations

from typing import Final

import numpy as np
import polars as pl

from hft_forecast.core import columns as C
from hft_forecast.core.errors import LookaheadError
from hft_forecast.ml.base import FEATURE_PREFIX, SESSION_ID, FeatureSet, MlRegistry

FEATURE_SETS: Final[MlRegistry[FeatureSet]] = MlRegistry("feature_set", FeatureSet)
register_feature_set = FEATURE_SETS.decorator
"""Decorator: ``@register_feature_set(name="basic", version="1.0.0")``."""

DEFAULT_CUTS: Final = 5
TOLERANCE: Final = 1e-12


def feature_columns(frame: pl.DataFrame) -> list[str]:
    """The ``f_*`` columns of a frame, in order."""
    return [c for c in frame.columns if c.startswith(FEATURE_PREFIX)]


def check_feature_lookahead(
    feature_set: FeatureSet, frame: pl.DataFrame, n_cuts: int = DEFAULT_CUTS
) -> None:
    """Assert that truncating the frame never changes the features before the cut.

    The same idea as :mod:`engine.lookahead`: a feature that peeks at a later bar changes
    when the later bars are removed.

    Raises:
        LookaheadError: If any feature differs between the full and the truncated frame.
    """
    full = feature_set.compute(frame)
    columns = feature_columns(full)
    stamps = frame.get_column(C.TIMESTAMP).unique().sort()
    step = max(len(stamps) // (n_cuts + 1), 1)
    for cut in stamps[step::step][:n_cuts]:
        truncated = frame.filter(pl.col(C.TIMESTAMP) <= cut)
        if truncated.height == frame.height:
            continue
        got = feature_set.compute(truncated)
        want = full.filter(pl.col(C.TIMESTAMP) <= cut)
        for column in columns:
            a = got.get_column(column).to_numpy().astype(float)
            b = want.get_column(column).to_numpy().astype(float)
            differs = (a != b) & ~(np.isnan(a) & np.isnan(b))
            if differs.any():
                raise LookaheadError(
                    f"Feature '{column}' of '{feature_set.__class__.__name__}' changes when "
                    f"bars after {cut} are removed",
                    details={"cut": str(cut), "rows": int(differs.sum())},
                )


def session_group() -> list[str]:
    """Grouping every rolling feature must use."""
    return [C.INSTRUMENT, SESSION_ID]

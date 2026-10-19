"""Every registered feature set must be causal, and the basic set must be useful."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from hft_forecast.core import columns as C
from hft_forecast.core.errors import ConfigError, LookaheadError
from hft_forecast.ml.base import SESSION_ID, FeatureSet
from hft_forecast.ml.features import (
    FEATURE_SETS,
    check_feature_lookahead,
    feature_columns,
    register_feature_set,
)

SETS = [s.name for s in FEATURE_SETS.specs()]


@pytest.mark.parametrize("name", SETS)
def test_every_registered_feature_set_is_causal(name, quote_bars):
    spec = FEATURE_SETS.get(name)
    feature_set = spec.create()
    try:
        check_feature_lookahead(feature_set, quote_bars.head(2000))
    except ConfigError:
        pytest.skip(f"'{name}' needs columns this dataset does not have")


def test_the_check_catches_a_peeking_feature(quote_bars):
    class Peeker(FeatureSet):
        """Looks one bar ahead."""

        def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
            """Tomorrow's mid, which is exactly what must be impossible."""
            return frame.with_columns(pl.col(C.MID).shift(-1).alias("f_peek"))

    with pytest.raises(LookaheadError, match="f_peek"):
        check_feature_lookahead(Peeker(), quote_bars.head(2000))


def test_basic_produces_finite_features_after_the_warmup(quote_bars):
    frame = FEATURE_SETS.get("basic").create().compute(quote_bars)
    columns = feature_columns(frame)
    assert len(columns) == 12
    settled = frame.group_by([C.INSTRUMENT, SESSION_ID]).tail(100)
    values = settled.select(columns).to_numpy()
    assert np.isfinite(values).all()


def test_rolling_features_restart_each_session(quote_bars):
    frame = FEATURE_SETS.get("basic").create().compute(quote_bars)
    first = frame.sort(C.TIMESTAMP).group_by([C.INSTRUMENT, SESSION_ID]).first()
    assert first["f_vol_20"].is_null().all()
    assert first["f_logret_1"].is_null().all()


def test_a_feature_set_must_subclass_the_base():
    with pytest.raises(Exception, match="must subclass"):

        @register_feature_set(name="t_not_a_set")
        class NotASet:  # pragma: no cover - the decorator raises first
            pass

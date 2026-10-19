"""Lookahead truncation test on every shipped strategy, plus a negative control."""

from __future__ import annotations

import numpy as np
import pytest

from hft_forecast.core import columns as C
from hft_forecast.core.errors import ConfigError
from hft_forecast.core.interfaces import Strategy
from hft_forecast.core.registry import REGISTRIES, PluginKind
from hft_forecast.core.types import MarketData, TargetFrame
from hft_forecast.engine.lookahead import check_lookahead
from tests.helpers import synthetic, with_required_fields

DATA = synthetic(instruments=("AAA", "BBB", "CCC", "DDD", "EEE", "FFF"))
SHIPPED = [s for s in REGISTRIES[PluginKind.STRATEGY].specs() if s.origin == "builtin"]


@pytest.mark.parametrize("spec", SHIPPED, ids=lambda s: s.name)
def test_shipped_strategy_has_no_lookahead(spec):
    strategy = spec.create()
    try:
        report = check_lookahead(strategy, with_required_fields(strategy, DATA))
    except ConfigError as exc:
        # prediction_signal needs a predictions file; it is checked in tests/unit/ml
        pytest.skip(f"needs configuration this test cannot guess: {exc.message}")
    assert report.passed, report.mismatches


class Peeker(Strategy):
    """Uses tomorrow's close: must be caught."""

    def generate_targets(self, data: MarketData) -> TargetFrame:
        close = data.panel(C.CLOSE)
        tomorrow = np.vstack([close[1:], np.full((1, close.shape[1]), np.nan)])
        return TargetFrame(data.timestamps, data.instruments, np.where(tomorrow > close, 1.0, 0.0))


def test_peeking_strategy_is_detected():
    report = check_lookahead(Peeker(), DATA)
    assert not report.passed
    assert report.mismatches

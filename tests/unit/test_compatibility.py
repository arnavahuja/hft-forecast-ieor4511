"""Config compatibility checks, in particular the sub-minute frequency gate."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from hft_forecast.core.interfaces import (
    ENGINE_VECTORIZED,
    FREQ_DAILY,
    FREQ_INTRADAY,
    FREQ_SUBMINUTE,
    Strategy,
)
from hft_forecast.core.registry import register, registry
from hft_forecast.core.types import Frequency, MarketData, TargetFrame
from hft_forecast.services.compatibility import CompatibilityChecker
from tests.helpers import config

CHECKER = CompatibilityChecker()


@pytest.fixture(scope="module", autouse=True)
def _daily_only_strategy():
    @register("strategy", name="t_daily_only", version="1.0.0")
    class DailyOnly(Strategy):
        """Declares daily and intraday bars but not sub-minute ones."""

        capabilities: ClassVar[frozenset[str]] = frozenset(
            {ENGINE_VECTORIZED, FREQ_DAILY, FREQ_INTRADAY}
        )

        def generate_targets(self, data: MarketData) -> TargetFrame:
            """Flat."""
            return TargetFrame(
                data.timestamps, data.instruments, np.zeros((len(data.timestamps), 1))
            )

    yield
    registry("strategy").remove_module(DailyOnly.__module__)


def _issues(strategy: str, frequency: Frequency) -> list[str]:
    cfg = config(strategy)
    cfg = cfg.model_copy(update={"data": cfg.data.model_copy(update={"frequency": frequency})})
    return [i.message for i in CHECKER.check(cfg).issues if i.level == "error"]


@pytest.mark.parametrize("frequency", [Frequency.D1, Frequency.MIN1, Frequency.S5])
def test_buy_and_hold_runs_on_every_frequency(frequency):
    assert _issues("buy_and_hold", frequency) == []


def test_a_strategy_without_the_subminute_capability_is_rejected_on_second_bars():
    assert _issues("t_daily_only", Frequency.MIN1) == []
    errors = _issues("t_daily_only", Frequency.S5)
    assert any(FREQ_SUBMINUTE in e for e in errors), errors

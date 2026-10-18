"""Buy and hold."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from hft_forecast.core import columns as C
from hft_forecast.core.interfaces import (
    ASSET_EQUITY,
    ENGINE_EVENT,
    ENGINE_VECTORIZED,
    FREQ_DAILY,
    FREQ_INTRADAY,
    FREQ_SUBMINUTE,
    Context,
    Strategy,
)
from hft_forecast.core.params import StrategyParams
from hft_forecast.core.registry import register
from hft_forecast.core.types import MarketData, TargetFrame


class BuyAndHoldParams(StrategyParams):
    """Buy and hold has no parameters."""


@register("strategy", name="buy_and_hold", version="1.0.0", tags=["benchmark", "basic"])
class BuyAndHold(Strategy):
    """Equal-weight every instrument that has a price, then let positions drift.

    With the default ``on_change`` rebalance rule the portfolio is only traded when an
    instrument starts or stops having prices.
    """

    Params = BuyAndHoldParams
    capabilities: ClassVar[frozenset[str]] = frozenset(
        {
            ENGINE_VECTORIZED,
            ENGINE_EVENT,
            ASSET_EQUITY,
            FREQ_DAILY,
            FREQ_INTRADAY,
            FREQ_SUBMINUTE,
        }
    )

    def generate_targets(self, data: MarketData) -> TargetFrame:
        """Equal weights across instruments with a price at each bar."""
        valid = np.isfinite(data.panel(C.CLOSE))
        count = valid.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(valid & (count > 0), 1.0 / count, 0.0)
        return TargetFrame(data.timestamps, data.instruments, weights)

    def on_bar(self, ctx: Context) -> None:
        """Set equal weights whenever the set of priced instruments changes."""
        valid = np.isfinite(ctx.current(C.CLOSE))
        key = tuple(bool(v) for v in valid)
        if ctx.state.get("key") == key:
            return
        ctx.state["key"] = key
        n = int(valid.sum())
        ctx.set_target_weights(
            {i: 1.0 / n for i, ok in zip(ctx.instruments, valid, strict=True) if ok} if n else {}
        )

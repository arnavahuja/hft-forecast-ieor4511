"""Example strategy: a user strategy."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from pydantic import Field

from hft_forecast.core import (
    MarketData,
    Strategy,
    StrategyParams,
    TargetFrame,
    register,
)
from hft_forecast.core import (
    columns as C,
)
from hft_forecast.core.numeric import rolling_mean


class ExampleStrategyParams(StrategyParams):
    """Parameters (the UI form is generated from these fields)."""

    window: int = Field(50, ge=2, le=500, description="Moving average window (bars)")


@register("strategy", name="example_strategy", version="0.1.0", tags=["user"])
class ExampleStrategy(Strategy):
    """Long instruments trading above their moving average, equal-weighted."""

    Params = ExampleStrategyParams
    params: ExampleStrategyParams
    capabilities: ClassVar[frozenset[str]] = frozenset(
        {"engine:vectorized", "asset:equity", "freq:daily"}
    )

    def warmup(self) -> int:
        """Bars needed before the first decision."""
        return self.params.window

    def generate_targets(self, data: MarketData) -> TargetFrame:
        """Use only rows up to each timestamp (trailing windows); the engine applies the lag."""
        close = data.panel(C.CLOSE)
        above = close > rolling_mean(close, self.params.window)
        n = max(len(data.instruments), 1)
        weights = np.where(np.isfinite(close), above / n, np.nan)
        return TargetFrame(data.timestamps, data.instruments, weights)

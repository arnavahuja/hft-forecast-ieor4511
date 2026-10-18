"""Equal-weight constructor."""

from __future__ import annotations

import numpy as np
from pydantic import Field

from hft_forecast.core.interfaces import ConstructionContext, PortfolioConstructor
from hft_forecast.core.params import ConstructorParams
from hft_forecast.core.registry import register
from hft_forecast.core.types import TargetFrame, TargetKind


class EqualWeightParams(ConstructorParams):
    """Equal weight parameters."""

    allow_short: bool = Field(True, description="Short instruments with negative signals")
    gross: float = Field(1.0, gt=0, le=5, description="Target gross exposure")


@register("portfolio_constructor", name="equal_weight", version="1.0.0", tags=["basic"])
class EqualWeight(PortfolioConstructor):
    """Equal weight across instruments with an active signal (sign gives the side)."""

    Params = EqualWeightParams
    params: EqualWeightParams

    def construct(self, signals: TargetFrame, ctx: ConstructionContext) -> TargetFrame:
        """``sign(signal) / count`` scaled to the gross target."""
        s = np.nan_to_num(signals.values, nan=0.0)
        side = np.sign(s) if self.params.allow_short else (s > 0).astype(np.float64)
        count = np.abs(side).sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(count > 0, side / count * self.params.gross, 0.0)
        return signals.replace(values=w, kind=TargetKind.WEIGHTS)

"""Signal-proportional constructor."""

from __future__ import annotations

import numpy as np
from pydantic import Field

from hft_forecast.core.interfaces import ConstructionContext, PortfolioConstructor
from hft_forecast.core.numeric import normalize_gross
from hft_forecast.core.params import ConstructorParams
from hft_forecast.core.registry import register
from hft_forecast.core.types import TargetFrame, TargetKind


class SignalProportionalParams(ConstructorParams):
    """Signal-proportional parameters."""

    allow_short: bool = Field(True, description="Keep negative signals as shorts")
    gross: float = Field(1.0, gt=0, le=5, description="Target gross exposure")
    demean: bool = Field(False, description="Subtract the cross-sectional mean first")


@register("portfolio_constructor", name="signal_proportional", version="1.0.0", tags=["basic"])
class SignalProportional(PortfolioConstructor):
    """Weights proportional to the signal value, scaled to a gross exposure."""

    Params = SignalProportionalParams
    params: SignalProportionalParams

    def construct(self, signals: TargetFrame, ctx: ConstructionContext) -> TargetFrame:
        """Normalize signals by their absolute sum each row."""
        s = signals.values.copy()
        if self.params.demean:
            s = s - np.nanmean(np.where(np.isfinite(s), s, np.nan), axis=1, keepdims=True)
        s = np.nan_to_num(s, nan=0.0)
        if not self.params.allow_short:
            s = np.clip(s, 0.0, None)
        return signals.replace(
            values=normalize_gross(s, self.params.gross), kind=TargetKind.WEIGHTS
        )

"""Slippage and market impact models."""

from __future__ import annotations

from typing import ClassVar, Final

import numpy as np
from pydantic import Field

from hft_forecast.core.interfaces import (
    ENGINE_EVENT,
    ENGINE_VECTORIZED,
    BarSnapshot,
    CostContext,
    SlippageModel,
)
from hft_forecast.core.numeric import rolling_mean, rolling_std
from hft_forecast.core.params import CostParams
from hft_forecast.core.registry import register
from hft_forecast.core.types import FloatArray

BPS: Final = 1e-4
BOTH_ENGINES = frozenset({ENGINE_VECTORIZED, ENGINE_EVENT})


class HalfSpreadParams(CostParams):
    """Half-spread slippage."""

    spread_bps: float = Field(5.0, ge=0, le=1000, description="Full bid-ask spread in bps")


@register("slippage_model", name="half_spread", version="1.0.0", tags=["basic"])
class HalfSpread(SlippageModel):
    """Pay half the quoted spread on every trade."""

    Params = HalfSpreadParams
    params: HalfSpreadParams
    capabilities: ClassVar[frozenset[str]] = BOTH_ENGINES

    def vectorized(self, ctx: CostContext) -> FloatArray:
        """Half spread x |traded weight|."""
        return np.abs(ctx.trades) * 0.5 * self.params.spread_bps * BPS

    def price_adjustment(self, quantity: float, price: float, bar: BarSnapshot) -> float:
        """Half the spread per unit."""
        return price * 0.5 * self.params.spread_bps * BPS


class VolScaledParams(CostParams):
    """Volatility-scaled slippage."""

    k: float = Field(0.1, ge=0, le=10, description="Fraction of one-bar volatility paid")
    window: int = Field(20, ge=2, le=252, description="Volatility window (bars)")


@register("slippage_model", name="volatility_scaled", version="1.0.0", tags=["impact"])
class VolatilityScaled(SlippageModel):
    """Slippage proportional to recent one-bar volatility (trailing, lookahead-safe)."""

    Params = VolScaledParams
    params: VolScaledParams
    capabilities: ClassVar[frozenset[str]] = BOTH_ENGINES

    def vectorized(self, ctx: CostContext) -> FloatArray:
        """K x trailing vol x |traded weight|."""
        vol = np.nan_to_num(rolling_std(ctx.returns, self.params.window, 2), nan=0.0)
        return np.abs(ctx.trades) * self.params.k * vol

    def price_adjustment(self, quantity: float, price: float, bar: BarSnapshot) -> float:
        """K x vol per unit."""
        vol = bar.volatility if np.isfinite(bar.volatility) else 0.0
        return price * self.params.k * vol


class SqrtImpactParams(CostParams):
    """Square-root market impact."""

    k: float = Field(1.0, ge=0, le=10, description="Impact coefficient")
    vol_window: int = Field(20, ge=2, le=252, description="Volatility window (bars)")
    adv_window: int = Field(20, ge=1, le=252, description="Average volume window (bars)")


@register("slippage_model", name="sqrt_impact", version="1.0.0", tags=["impact", "capacity"])
class SqrtImpact(SlippageModel):
    """Square-root law: cost per unit = k x sigma x sqrt(order size / ADV)."""

    Params = SqrtImpactParams
    params: SqrtImpactParams
    capabilities: ClassVar[frozenset[str]] = BOTH_ENGINES

    def vectorized(self, ctx: CostContext) -> FloatArray:
        """Impact fraction x |traded weight|."""
        p = self.params
        vol = np.nan_to_num(rolling_std(ctx.returns, p.vol_window, 2), nan=0.0)
        if ctx.volumes is None:
            return np.zeros_like(ctx.trades)
        adv = rolling_mean(ctx.volumes, p.adv_window, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            participation = np.nan_to_num(ctx.traded_units() / adv, nan=0.0, posinf=1.0)
        return np.abs(ctx.trades) * p.k * vol * np.sqrt(participation)

    def price_adjustment(self, quantity: float, price: float, bar: BarSnapshot) -> float:
        """Impact per unit using this bar's volume as the ADV proxy."""
        vol = bar.volatility if np.isfinite(bar.volatility) else 0.0
        if bar.volume <= 0:
            return 0.0
        return price * self.params.k * vol * float(np.sqrt(abs(quantity) / bar.volume))

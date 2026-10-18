"""Fill models for the event engine."""

from __future__ import annotations

from typing import ClassVar

from hft_forecast.core.interfaces import ENGINE_EVENT, BarSnapshot, FillModel
from hft_forecast.core.params import CostParams
from hft_forecast.core.registry import register
from hft_forecast.core.types import Order, Side

EVENT_ONLY = frozenset({ENGINE_EVENT})


class NoParams(CostParams):
    """No parameters."""


@register("fill_model", name="close", version="1.0.0", tags=["basic"])
class CloseFill(FillModel):
    """Market orders fill at the bar's close."""

    Params = NoParams
    capabilities: ClassVar[frozenset[str]] = EVENT_ONLY

    def fill(self, order: Order, bar: BarSnapshot, max_quantity: float) -> tuple[float, float]:
        """Close price."""
        return bar.close, min(order.quantity, max_quantity)


@register("fill_model", name="open", version="1.0.0", tags=["basic"])
class OpenFill(FillModel):
    """Market orders fill at the bar's open."""

    Params = NoParams
    capabilities: ClassVar[frozenset[str]] = EVENT_ONLY

    def fill(self, order: Order, bar: BarSnapshot, max_quantity: float) -> tuple[float, float]:
        """Open price."""
        return bar.open, min(order.quantity, max_quantity)


@register("fill_model", name="typical_price", version="1.0.0", tags=["vwap"])
class TypicalPriceFill(FillModel):
    """Fill at the typical price (high + low + close) / 3, a VWAP proxy for bar data."""

    Params = NoParams
    capabilities: ClassVar[frozenset[str]] = EVENT_ONLY

    def fill(self, order: Order, bar: BarSnapshot, max_quantity: float) -> tuple[float, float]:
        """Typical price."""
        return (bar.high + bar.low + bar.close) / 3.0, min(order.quantity, max_quantity)


@register("fill_model", name="worst_price", version="1.0.0", tags=["conservative"])
class WorstPriceFill(FillModel):
    """Pessimistic fill: buys at the bar high, sells at the bar low."""

    Params = NoParams
    capabilities: ClassVar[frozenset[str]] = EVENT_ONLY

    def fill(self, order: Order, bar: BarSnapshot, max_quantity: float) -> tuple[float, float]:
        """High for buys, low for sells."""
        price = bar.high if order.side is Side.BUY else bar.low
        return price, min(order.quantity, max_quantity)

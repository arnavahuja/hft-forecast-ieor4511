"""Holding costs: short borrow fees and margin interest."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from pydantic import Field

from hft_forecast.core.interfaces import ENGINE_EVENT, ENGINE_VECTORIZED, CostContext, CostModel
from hft_forecast.core.params import CostParams
from hft_forecast.core.registry import register
from hft_forecast.core.results import COST_BORROW, COST_FINANCING
from hft_forecast.core.types import FloatArray, Instrument

BOTH_ENGINES = frozenset({ENGINE_VECTORIZED, ENGINE_EVENT})


class BorrowParams(CostParams):
    """Short borrow fee."""

    annual_rate: float = Field(0.005, ge=0, le=1, description="Annual fee on short notional")


@register("cost_model", name="short_borrow_fee", version="1.0.0", tags=["financing", "short"])
class ShortBorrowFee(CostModel):
    """Annual borrow fee charged every bar on the value of short positions."""

    Params = BorrowParams
    params: BorrowParams
    capabilities: ClassVar[frozenset[str]] = BOTH_ENGINES | {"supports:short"}
    category: ClassVar[str] = COST_BORROW

    def vectorized(self, ctx: CostContext) -> FloatArray:
        """Rate / periods_per_year x |short weight|."""
        short = -np.clip(ctx.holdings, None, 0.0)
        return short * self.params.annual_rate / ctx.periods_per_year

    def holding_cost(
        self, market_value: float, instrument: Instrument, year_fraction: float
    ) -> float:
        """Fee on a short position's value."""
        return max(-market_value, 0.0) * self.params.annual_rate * year_fraction


class MarginParams(CostParams):
    """Margin interest."""

    annual_spread: float = Field(
        0.015, ge=0, le=1, description="Annual spread over the cash rate on borrowed cash"
    )


@register("cost_model", name="margin_interest", version="1.0.0", tags=["financing", "leverage"])
class MarginInterest(CostModel):
    """Spread over the cash rate on borrowed cash (long exposure above 100%).

    The base cash rate itself is applied by the engine (``execution.cash_rate``) to positive
    and negative cash alike; this model adds the broker's spread on the borrowed part.
    """

    Params = MarginParams
    params: MarginParams
    capabilities: ClassVar[frozenset[str]] = BOTH_ENGINES
    category: ClassVar[str] = COST_FINANCING

    def vectorized(self, ctx: CostContext) -> FloatArray:
        """Borrowed fraction x spread / periods, allocated to long positions pro rata."""
        longs = np.clip(ctx.holdings, 0.0, None)
        long_total = longs.sum(axis=1, keepdims=True)
        borrowed = np.clip(ctx.holdings.sum(axis=1, keepdims=True) - 1.0, 0.0, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(long_total > 0, longs / long_total, 0.0)
        return share * borrowed * self.params.annual_spread / ctx.periods_per_year

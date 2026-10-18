"""Portfolio ledger: cash, positions, margin, corporate actions and holding costs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from hft_forecast.core.interfaces import CostModel
from hft_forecast.core.results import COST_BORROW, COST_FINANCING
from hft_forecast.core.types import (
    AssetClass,
    Fill,
    FloatArray,
    Instrument,
    PortfolioState,
    Position,
)

DEFAULT_MARGIN: Final = {
    AssetClass.FUTURE: 0.10,
    AssetClass.OPTION: 1.0,
}
SHORT_MARGIN: Final = 0.5


@dataclass
class Ledger:
    """Tracks the portfolio through time.

    Futures and shorts are carried at notional value (``units x price x multiplier``); the
    margin used is reported separately and does not change equity.

    Args:
        instruments: Instrument ids (column order).
        meta: Instrument metadata.
        cash: Starting cash.
        cash_rate: Annual interest on cash (paid on negative cash).
        periods_per_year: Bars per year.
    """

    instruments: tuple[str, ...]
    meta: dict[str, Instrument]
    cash: float
    cash_rate: float
    periods_per_year: float
    units: FloatArray = field(init=False)
    avg_price: FloatArray = field(init=False)
    multipliers: FloatArray = field(init=False)
    classes: tuple[AssetClass, ...] = field(init=False)

    def __post_init__(self) -> None:
        n = len(self.instruments)
        self.units = np.zeros(n)
        self.avg_price = np.zeros(n)
        self.multipliers = np.array(
            [self.meta[i].multiplier if i in self.meta else 1.0 for i in self.instruments]
        )
        self.classes = tuple(
            self.meta[i].asset_class if i in self.meta else AssetClass.EQUITY
            for i in self.instruments
        )

    # ---- valuation

    def market_values(self, prices: FloatArray) -> FloatArray:
        """Signed market value per instrument (0 where price missing)."""
        px = np.where(np.isfinite(prices), prices, 0.0)
        return self.units * px * self.multipliers

    def equity(self, prices: FloatArray) -> float:
        """Cash plus market value."""
        return float(self.cash + self.market_values(prices).sum())

    def margin_used(self, prices: FloatArray) -> float:
        """Margin requirement: futures initial margin plus short-sale margin."""
        mv = self.market_values(prices)
        rates = np.array([DEFAULT_MARGIN.get(c, 0.0) for c in self.classes])
        short_equity = (mv < 0) & (rates == 0)
        return float((np.abs(mv) * rates).sum() + (np.abs(mv[short_equity]) * SHORT_MARGIN).sum())

    def state(self, timestamp: np.datetime64, prices: FloatArray) -> PortfolioState:
        """Immutable snapshot."""
        positions = tuple(
            Position(
                inst,
                float(self.units[j]),
                float(self.avg_price[j]),
                float(prices[j]) if np.isfinite(prices[j]) else 0.0,
                float(self.multipliers[j]),
            )
            for j, inst in enumerate(self.instruments)
            if abs(self.units[j]) > 0
        )
        return PortfolioState(timestamp, self.cash, positions, self.margin_used(prices))

    # ---- events

    def apply_fills(self, fills: Sequence[Fill]) -> None:
        """Update cash and positions for fills (quantities are signed)."""
        for f in fills:
            j = self.instruments.index(f.instrument)
            mult = self.multipliers[j]
            old = self.units[j]
            new = old + f.quantity
            if new == 0:
                self.avg_price[j] = 0.0
            elif old == 0 or np.sign(new) != np.sign(old):
                self.avg_price[j] = f.price  # opened or flipped
            elif abs(new) > abs(old):
                self.avg_price[j] = (
                    self.avg_price[j] * abs(old) + f.price * abs(f.quantity)
                ) / abs(new)
            self.units[j] = new
            self.cash -= f.quantity * f.price * mult + f.commission

    def apply_dividends(self, dividends: FloatArray) -> float:
        """Credit (or debit, for shorts) cash dividends per unit. Returns the amount."""
        amount = float(np.nansum(self.units * np.nan_to_num(dividends) * self.multipliers))
        self.cash += amount
        return amount

    def apply_splits(self, ratios: FloatArray) -> None:
        """Adjust units and average prices for splits (raw price data only)."""
        r = np.where(np.isfinite(ratios) & (ratios > 0), ratios, 1.0)
        self.units = self.units * r
        self.avg_price = np.where(r != 1.0, self.avg_price / r, self.avg_price)

    def settle_expiries(self, today: np.datetime64, prices: FloatArray) -> float:
        """Cash-settle expired option contracts at intrinsic value. Returns the cash moved."""
        moved = 0.0
        day = today.astype("datetime64[D]")
        for j, inst in enumerate(self.instruments):
            meta = self.meta.get(inst)
            if (
                meta is None
                or meta.asset_class is not AssetClass.OPTION
                or meta.expiry is None
                or self.units[j] == 0
                or np.datetime64(meta.expiry) > day
            ):
                continue
            und = meta.underlying
            if und not in self.instruments or meta.strike is None or meta.right is None:
                continue
            spot = float(prices[self.instruments.index(und)])
            intrinsic = (
                max(spot - meta.strike, 0.0)
                if meta.right.value == "call"
                else max(meta.strike - spot, 0.0)
            )
            amount = float(self.units[j] * intrinsic * self.multipliers[j])
            self.cash += amount
            self.units[j] = 0.0
            self.avg_price[j] = 0.0
            moved += amount
        return moved

    def accrue(self, prices: FloatArray, cost_models: Sequence[CostModel]) -> dict[str, float]:
        """Charge one bar of cash interest and holding costs. Returns costs by category."""
        year_fraction = 1.0 / self.periods_per_year
        self.cash += self.cash * self.cash_rate * year_fraction
        costs: dict[str, float] = {}
        mv = self.market_values(prices)
        for model in cost_models:
            total = 0.0
            for j, inst in enumerate(self.instruments):
                if mv[j] != 0:
                    total += model.holding_cost(
                        float(mv[j]),
                        self.meta.get(inst, Instrument(id=inst, symbol=inst)),
                        year_fraction,
                    )
            if total:
                category = (
                    model.category
                    if model.category in (COST_BORROW, COST_FINANCING)
                    else COST_FINANCING
                )
                costs[category] = costs.get(category, 0.0) + total
                self.cash -= total
        return costs

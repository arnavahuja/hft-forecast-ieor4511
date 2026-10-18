"""Per-bar context handed to ``on_bar``: data up to the current bar only."""

from __future__ import annotations

from typing import Any

import numpy as np

from hft_forecast.core.errors import DataError, LookaheadError
from hft_forecast.core.types import (
    FloatArray,
    Frequency,
    Order,
    OrderType,
    PortfolioState,
    Side,
    TimeArray,
    TimeInForce,
)
from hft_forecast.engine.event.broker import SimulatedBroker
from hft_forecast.engine.event.ledger import Ledger
from hft_forecast.engine.resample import aggregate_blocks, period_codes


class EventContext:
    """Implements :class:`hft_forecast.core.interfaces.Context` for the event engine.

    History arrays are read-only views ending at the current bar; there is no way to obtain
    a later row, and :meth:`value_at` raises :class:`LookaheadError` for future indices.
    """

    def __init__(
        self,
        panels: dict[str, FloatArray],
        timestamps: TimeArray,
        instruments: tuple[str, ...],
        ledger: Ledger,
        broker: SimulatedBroker,
        equity_history: list[float],
        state: dict[str, Any],
        periods_per_year: float,
    ) -> None:
        self._panels = panels
        self._timestamps = timestamps
        self._instruments = instruments
        self._ledger = ledger
        self._broker = broker
        self._equity = equity_history
        self._state = state
        self._t = -1
        self._targets: dict[str, float] | None = None
        self._ppy = periods_per_year

    # ---- engine-facing

    def advance(self, t: int) -> None:
        """Move to bar ``t`` and clear per-bar targets."""
        self._t = t
        self._targets = None

    def use_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """Swap the scratch state (each plugin gets its own); returns the previous one."""
        previous, self._state = self._state, state
        return previous

    @property
    def staged_targets(self) -> dict[str, float] | None:
        """Target weights set during this bar (``None`` if untouched)."""
        return self._targets

    # ---- Context protocol

    @property
    def now(self) -> np.datetime64:
        """Current bar timestamp."""
        return self._timestamps[self._t]

    @property
    def bar_index(self) -> int:
        """Current bar index."""
        return self._t

    @property
    def instruments(self) -> tuple[str, ...]:
        """Instrument ids."""
        return self._instruments

    @property
    def portfolio(self) -> PortfolioState:
        """Portfolio marked at the current close."""
        return self._ledger.state(self.now, self._panels["close"][self._t])

    @property
    def periods_per_year(self) -> float:
        """Annualization factor."""
        return self._ppy

    @property
    def state(self) -> dict[str, Any]:
        """Persistent scratch storage for the strategy or overlay."""
        return self._state

    def _panel(self, field_name: str) -> FloatArray:
        try:
            return self._panels[field_name]
        except KeyError:
            raise DataError(
                f"Field '{field_name}' not available", details={"available": sorted(self._panels)}
            ) from None

    def history(self, field_name: str, lookback: int | None = None) -> FloatArray:
        """Rows up to and including the current bar."""
        view = self._panel(field_name)[: self._t + 1]
        if lookback is not None:
            view = view[max(0, len(view) - lookback) :]
        out = view.view()
        out.setflags(write=False)
        return out

    def current(self, field_name: str) -> FloatArray:
        """Current bar values."""
        return np.array(self._panel(field_name)[self._t])

    def value_at(self, field_name: str, bar_index: int) -> FloatArray:
        """Values at a past bar.

        Raises:
            LookaheadError: If ``bar_index`` is after the current bar.
        """
        if bar_index > self._t:
            raise LookaheadError(
                f"Bar {bar_index} is in the future (current bar {self._t})",
                details={"requested": bar_index, "current": self._t},
            )
        return np.array(self._panel(field_name)[bar_index])

    def timestamps(self, lookback: int | None = None) -> TimeArray:
        """Timestamps up to and including the current bar."""
        ts = self._timestamps[: self._t + 1]
        return ts[max(0, len(ts) - lookback) :] if lookback else ts

    def portfolio_returns(self, lookback: int | None = None) -> FloatArray:
        """Portfolio returns realized so far (last element = current bar)."""
        eq = np.asarray(self._equity, dtype=np.float64)
        rets = eq[1:] / eq[:-1] - 1.0 if len(eq) > 1 else np.zeros(0)
        return rets[-lookback:] if lookback else rets

    def resampled(self, field_name: str, frequency: str, lookback: int | None = None) -> FloatArray:
        """Completed coarser bars (e.g. ``1d``) aggregated from bars up to now."""
        ts = self._timestamps[: self._t + 1]
        codes = period_codes(ts, Frequency(frequency))
        completed = codes < codes[-1]
        if not completed.any():
            return np.zeros((0, len(self._instruments)))
        block = self._panel(field_name)[: self._t + 1][completed]
        _, starts = np.unique(codes[completed], return_index=True)
        out = aggregate_blocks(block, starts, field_name)
        return out[-lookback:] if lookback else out

    def order(
        self,
        instrument: str,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
        tif: TimeInForce = TimeInForce.GTC,
        tag: str = "",
    ) -> str:
        """Submit an explicit order (positive quantity buys)."""
        if instrument not in self._broker.col:
            raise DataError(f"Unknown instrument '{instrument}'")
        if quantity == 0:
            return ""
        side = Side.BUY if quantity > 0 else Side.SELL
        order = Order(
            id="",
            instrument=instrument,
            side=side,
            quantity=abs(quantity),
            type=order_type,
            tif=tif,
            limit_price=limit_price,
            stop_price=stop_price,
            created_at=self.now,
            tag=tag,
        )
        return self._broker.submit(order, self._t)

    def cancel(self, order_id: str) -> None:
        """Cancel an open order."""
        self._broker.cancel(order_id)

    def open_orders(self) -> list[Order]:
        """Open orders."""
        return self._broker.open_orders()

    def set_target_weights(self, weights: dict[str, float]) -> None:
        """Rebalance towards target weights; unlisted instruments go to zero."""
        unknown = set(weights) - set(self._instruments)
        if unknown:
            raise DataError("Unknown instruments in targets", details={"unknown": sorted(unknown)})
        self._targets = {k: float(v) for k, v in weights.items() if np.isfinite(v)}

    def target_weights(self) -> dict[str, float]:
        """Targets staged during this bar (for overlays)."""
        return dict(self._targets or {})

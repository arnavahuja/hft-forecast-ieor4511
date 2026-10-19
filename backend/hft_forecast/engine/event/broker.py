"""Broker interface and the simulated broker.

The engine talks to a :class:`Broker` only, so a live broker adapter can be added later
without engine changes. :class:`SimulatedBroker` supports market, limit, stop and stop-limit
orders, time in force, partial fills through a volume participation cap, latency in bars and
pluggable fill, slippage and cost models.

Target-weight orders (from ``set_target_weights``) are sized at execution time from the
equity and price at that moment, which is what the vectorized engine assumes and makes the
two engines agree exactly at zero cost.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Final, Protocol

import numpy as np

from hft_forecast.core.interfaces import BarSnapshot, CostModel, FillModel, SlippageModel
from hft_forecast.core.run_config import ExecutionPrice
from hft_forecast.core.types import (
    Fill,
    FloatArray,
    Instrument,
    Order,
    OrderType,
    Side,
    TimeInForce,
)

QTY_EPS: Final = 1e-9


@dataclass
class OrderState:
    """Mutable broker-side state of an order."""

    order: Order
    eligible_bar: int
    remaining: float
    status: str = "open"
    triggered: bool = False
    target_weight: float | None = None
    created_bar: int = 0


@dataclass(frozen=True)
class BarBook:
    """Bars for all instruments at one step (arrays of shape ``(N,)``)."""

    index: int
    timestamp: np.datetime64
    open: FloatArray
    high: FloatArray
    low: FloatArray
    close: FloatArray
    volume: FloatArray
    volatility: FloatArray
    spread: FloatArray

    def snapshot(self, j: int) -> BarSnapshot:
        """Bar of instrument ``j``."""
        return BarSnapshot(
            self.timestamp,
            float(self.open[j]),
            float(self.high[j]),
            float(self.low[j]),
            float(self.close[j]),
            float(self.volume[j]),
            float(self.volatility[j]),
            float(self.spread[j]),
        )


@dataclass
class FillReport:
    """Fills produced at a bar, plus cost totals."""

    fills: list[Fill] = field(default_factory=list)
    commission: float = 0.0
    slippage: float = 0.0


class Broker(Protocol):
    """What the engine needs from a broker (simulated or live)."""

    def submit(self, order: Order, bar: int, target_weight: float | None = None) -> str:
        """Submit an order at bar ``bar``; returns its id."""
        ...

    def cancel(self, order_id: str) -> None:
        """Cancel an open order."""
        ...

    def open_orders(self) -> list[Order]:
        """Open orders."""
        ...

    def process(
        self,
        book: BarBook,
        positions: FloatArray,
        equity_at: Callable[[FloatArray], float],
    ) -> FillReport:
        """Execute eligible orders against a bar."""
        ...


class ExecutionPriceFill(FillModel):
    """Default fill: market orders fill at the configured execution price (open or close)."""

    def __init__(self, price: ExecutionPrice) -> None:
        super().__init__()
        self.price = price

    def fill(self, order: Order, bar: BarSnapshot, max_quantity: float) -> tuple[float, float]:
        """Fill up to ``max_quantity`` at the open or close."""
        px = bar.open if self.price is ExecutionPrice.OPEN else bar.close
        return px, min(order.quantity, max_quantity)


class SimulatedBroker:
    """Order book simulator.

    Args:
        instruments: Instrument ids (column order).
        meta: Instrument metadata (multipliers, asset classes).
        execution_price: Default execution price for market and target orders.
        lag_bars: Bars between decision and eligibility.
        latency_bars: Additional latency.
        max_participation: Cap on the fraction of bar volume one bar's fills may take.
        fill_model: Fill model for market orders (default: execution price).
        slippage: Slippage model.
        cost_models: Commission and fee models.
    """

    def __init__(
        self,
        instruments: tuple[str, ...],
        meta: dict[str, Instrument],
        execution_price: ExecutionPrice,
        lag_bars: int,
        latency_bars: int,
        max_participation: float,
        fill_model: FillModel | None = None,
        slippage: SlippageModel | None = None,
        cost_models: Sequence[CostModel] = (),
    ) -> None:
        self.instruments = instruments
        self.col = {inst: j for j, inst in enumerate(instruments)}
        self.meta = {i: meta.get(i, Instrument(id=i, symbol=i)) for i in instruments}
        self.multipliers = np.array([self.meta[i].multiplier for i in instruments])
        self.execution_price = execution_price
        self.delay = lag_bars + latency_bars
        self.max_participation = max_participation
        self.fill_model = fill_model or ExecutionPriceFill(execution_price)
        self.slippage = slippage
        self.cost_models = tuple(cost_models)
        self._orders: dict[str, OrderState] = {}
        self._ids = itertools.count(1)
        self.history: list[tuple[OrderState, str]] = []

    # ---- order management

    def submit(self, order: Order, bar: int, target_weight: float | None = None) -> str:
        """Queue an order; it becomes eligible ``lag + latency`` bars later."""
        oid = order.id or f"o{next(self._ids)}"
        order = replace(order, id=oid)
        self._orders[oid] = OrderState(
            order, bar + self.delay, order.quantity, target_weight=target_weight, created_bar=bar
        )
        return oid

    def cancel(self, order_id: str) -> None:
        """Cancel an open order."""
        state = self._orders.pop(order_id, None)
        if state is not None:
            state.status = "cancelled"
            self.history.append((state, "cancelled"))

    def cancel_targets(self) -> None:
        """Cancel pending target-weight orders (a new target supersedes them)."""
        for oid in [o for o, s in self._orders.items() if s.target_weight is not None]:
            self.cancel(oid)

    def open_orders(self) -> list[Order]:
        """Open orders."""
        return [s.order for s in self._orders.values()]

    def pending_target_weights(self) -> dict[str, float]:
        """Target weights of queued target orders."""
        return {
            s.order.instrument: float(s.target_weight)
            for s in self._orders.values()
            if s.target_weight is not None
        }

    # ---- execution

    def _trigger(self, state: OrderState, bar: BarSnapshot) -> tuple[bool, float | None]:
        """Whether the order can execute on this bar and at what reference price."""
        o = state.order
        buy = o.side is Side.BUY
        if o.type in (OrderType.STOP, OrderType.STOP_LIMIT) and not state.triggered:
            assert o.stop_price is not None
            hit = bar.high >= o.stop_price if buy else bar.low <= o.stop_price
            if not hit:
                return False, None
            state.triggered = True
            if o.type is OrderType.STOP:
                gap = max(bar.open, o.stop_price) if buy else min(bar.open, o.stop_price)
                return True, gap
        if o.type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
            assert o.limit_price is not None
            ok = bar.low <= o.limit_price if buy else bar.high >= o.limit_price
            if not ok:
                return False, None
            better = min(bar.open, o.limit_price) if buy else max(bar.open, o.limit_price)
            return True, better
        return True, None

    def process(
        self,
        book: BarBook,
        positions: FloatArray,
        equity_at: Callable[[FloatArray], float],
    ) -> FillReport:
        """Execute eligible orders on ``book``.

        Args:
            book: Current bars.
            positions: Units held before this bar's fills (mutated as fills happen).
            equity_at: Equity given a price vector (for sizing target orders).
        """
        report = FillReport()
        used = np.zeros(len(self.instruments))
        exec_px = book.open if self.execution_price is ExecutionPrice.OPEN else book.close
        equity = equity_at(np.where(np.isfinite(exec_px), exec_px, book.close))
        for oid in list(self._orders):
            state = self._orders[oid]
            if state.eligible_bar > book.index:
                continue
            j = self.col[state.order.instrument]
            if not np.isfinite(book.close[j]):
                continue
            if state.target_weight is not None:
                self._resize_target(state, j, float(exec_px[j]), equity, positions)
                if state.remaining <= QTY_EPS:
                    self._orders.pop(oid)
                    state.status = "filled"
                    self.history.append((state, "filled"))
                    continue
            snap = book.snapshot(j)
            can, ref_price = self._trigger(state, snap)
            if can:
                self._execute(state, j, snap, ref_price, used, positions, report)
            self._expire(state, book.index)
        return report

    def _resize_target(
        self, state: OrderState, j: int, price: float, equity: float, positions: FloatArray
    ) -> None:
        mult = self.multipliers[j]
        desired = float(state.target_weight or 0.0) * equity / (price * mult)
        delta = desired - positions[j]
        side = Side.BUY if delta > 0 else Side.SELL
        state.order = replace(state.order, side=side, quantity=abs(delta))
        state.remaining = abs(delta)

    def _execute(
        self,
        state: OrderState,
        j: int,
        snap: BarSnapshot,
        ref_price: float | None,
        used: FloatArray,
        positions: FloatArray,
        report: FillReport,
    ) -> None:
        o = state.order
        cap = self._participation_cap_from(snap, j, used)
        pending = replace(o, quantity=state.remaining)
        if ref_price is None:
            price, qty = self.fill_model.fill(pending, snap, cap)
        else:
            price, qty = ref_price, min(state.remaining, cap)
        if qty <= QTY_EPS or not np.isfinite(price):
            return
        signed = qty if o.side is Side.BUY else -qty
        inst = self.meta[o.instrument]
        slip = self.slippage.price_adjustment(signed, price, snap) if self.slippage else 0.0
        fill_price = price + slip if o.side is Side.BUY else price - slip
        commission = sum(m.commission(signed, fill_price, inst) for m in self.cost_models)
        fill = Fill(
            o.id,
            o.instrument,
            snap.timestamp,
            fill_price,
            signed,
            commission,
            abs(slip * qty * inst.multiplier),
        )
        report.fills.append(fill)
        report.commission += commission
        report.slippage += fill.slippage
        positions[j] += signed
        used[j] += qty
        state.remaining -= qty
        if state.remaining <= QTY_EPS:
            self._orders.pop(o.id, None)
            state.status = "filled"
            self.history.append((state, "filled"))

    def _participation_cap_from(self, snap: BarSnapshot, j: int, used: FloatArray) -> float:
        if self.max_participation >= 1.0:
            return float("inf")
        vol = snap.volume if np.isfinite(snap.volume) else 0.0
        return max(self.max_participation * vol - used[j], 0.0)

    def _expire(self, state: OrderState, bar: int) -> None:
        if state.order.id not in self._orders:
            return
        tif = state.order.tif
        if state.target_weight is not None or tif in (TimeInForce.DAY, TimeInForce.IOC):
            self._orders.pop(state.order.id, None)
            state.status = (
                "partially_filled"
                if state.remaining < state.order.quantity - QTY_EPS
                else "expired"
            )
            self.history.append((state, state.status))

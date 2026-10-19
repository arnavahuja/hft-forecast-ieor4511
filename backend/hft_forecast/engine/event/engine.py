"""Event-driven engine.

Event loop over time-ordered bars::

    MarketEvent (bar t) -> corporate actions -> broker fills eligible orders -> ledger
    -> mark to market at the close -> strategy / overlays decide -> OrderEvents queued

Strategies with a vectorized form run through an adapter that replays their (pipeline-
processed) targets bar by bar as target-weight orders, using the same rebalance schedule as
the vectorized engine; with zero costs both engines agree to floating-point precision.
Strategies that only implement ``on_bar`` (path-dependent logic, explicit order types) run
here natively; overlays with an ``on_bar`` form adjust their staged targets.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import polars as pl

from hft_forecast.core import columns as C
from hft_forecast.core.errors import EngineError
from hft_forecast.core.interfaces import (
    CostModel,
    FillModel,
    Overlay,
    SlippageModel,
    Strategy,
)
from hft_forecast.core.numeric import rolling_std
from hft_forecast.core.results import (
    COST_COMMISSION,
    COST_SLIPPAGE,
    FILL_COLUMNS,
    ORDER_COLUMNS,
    BacktestResult,
    OverlayReport,
    aux_panels,
    instrument_meta_json,
)
from hft_forecast.core.run_config import BacktestConfig, RebalanceRule
from hft_forecast.core.types import (
    Adjustment,
    AssetClass,
    BoolArray,
    Fill,
    FloatArray,
    MarketData,
    Order,
    Side,
    TimeInForce,
    pl_times,
)
from hft_forecast.engine.base import (
    CASH_RETURNS_KEY,
    RunOptions,
    align_targets,
    apply_membership,
    risk_free_series,
)
from hft_forecast.engine.event.broker import BarBook, SimulatedBroker
from hft_forecast.engine.event.context import EventContext
from hft_forecast.engine.event.ledger import Ledger
from hft_forecast.engine.pipeline import Pipeline
from hft_forecast.engine.returns import ReturnPanels, compute_returns
from hft_forecast.engine.schedule import rebalance_mask
from hft_forecast.engine.trades import extract_trades
from hft_forecast.engine.vectorized import ROLL_FIELD, VectorizedEngine, lag_targets

VOL_WINDOW: Final = 20
PROGRESS_EVERY: Final = 250
WEIGHT_EPS: Final = 1e-12


class EventEngine:
    """Bar-by-bar engine with a simulated broker.

    Args:
        cost_models: Commission, fee and holding-cost models.
        slippage: Slippage model.
        fill_model: Fill model for market orders.
    """

    name = "event"

    def __init__(
        self,
        cost_models: Sequence[CostModel] = (),
        slippage: SlippageModel | None = None,
        fill_model: FillModel | None = None,
    ) -> None:
        self.cost_models = tuple(cost_models)
        self.slippage = slippage
        self.fill_model = fill_model

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _panels(data: MarketData) -> dict[str, FloatArray]:
        panels: dict[str, FloatArray] = {}
        for f in data.fields:
            try:
                panels[f] = data.panel(f)
            except Exception:
                continue
        close = panels[C.CLOSE]
        for f in (C.OPEN, C.HIGH, C.LOW):
            panels.setdefault(f, close)
        panels.setdefault(C.VOLUME, np.full_like(close, np.inf))
        panels.setdefault(C.SPREAD, np.full_like(close, np.nan))
        return panels

    def _adapter_schedule(
        self,
        config: BacktestConfig,
        data: MarketData,
        strategy: Strategy,
        pipeline: Pipeline,
        options: RunOptions,
    ) -> tuple[FloatArray, BoolArray, tuple[OverlayReport, ...]]:
        """Decision-time targets and the decision bars at which to submit them."""
        strat_data = (
            data.select_instruments(options.strategy_instruments)
            if options.strategy_instruments is not None
            else data
        )
        raw = align_targets(strategy.generate_targets(strat_data), data)
        raw = apply_membership(raw, options.membership)
        vec = VectorizedEngine()
        targets, reports = vec.targets_after_pipeline(config, data, raw, pipeline, options)
        ex = config.execution
        tradable = np.isfinite(data.panel(C.CLOSE))
        exec_targets = np.where(tradable, lag_targets(targets.values, ex.lag_bars), 0.0)
        mask = rebalance_mask(
            data.timestamps, ex.rebalance, ex.rebalance_every_n, ex.lag_bars, exec_targets
        )
        decisions = np.zeros(len(mask), dtype=bool)
        decisions[: len(mask) - ex.lag_bars] = mask[ex.lag_bars :]
        return exec_targets, decisions, reports

    @staticmethod
    def _submit_targets(
        broker: SimulatedBroker,
        ledger: Ledger,
        targets: dict[str, float],
        t: int,
        now: np.datetime64,
    ) -> None:
        broker.cancel_targets()
        held = {inst for j, inst in enumerate(ledger.instruments) if abs(ledger.units[j]) > 0}
        for inst in sorted(set(targets) | held):
            w = float(targets.get(inst, 0.0))
            order = Order(
                id="",
                instrument=inst,
                side=Side.BUY if w >= 0 else Side.SELL,
                quantity=0.0,
                tif=TimeInForce.DAY,
                created_at=now,
                tag="target",
            )
            broker.submit(order, t, target_weight=w)

    # ------------------------------------------------------------------ run

    def run(
        self,
        config: BacktestConfig,
        data: MarketData,
        strategy: Strategy,
        pipeline: Pipeline,
        options: RunOptions,
    ) -> BacktestResult:
        """Run the event loop."""
        t0 = time.perf_counter()
        if len(data.timestamps) == 0:
            raise EngineError("No bars to simulate")
        loop = self._setup(config, data, strategy, pipeline, options)
        n_t = loop.n_t
        for t in range(n_t):
            if options.cancelled is not None and options.cancelled():
                raise EngineError("Run cancelled")
            if t % PROGRESS_EVERY == 0:
                options.report(0.2 + 0.7 * t / n_t, f"bar {t}/{n_t}")
            self._pre_trade(loop, t)
            self._trade(loop, t)
            self._mark(loop, t)
            self._decide(loop, t)
        options.report(0.95, "assembling results")
        return self._result(loop, t0)

    def _setup(
        self,
        config: BacktestConfig,
        data: MarketData,
        strategy: Strategy,
        pipeline: Pipeline,
        options: RunOptions,
    ) -> _Loop:
        ex = config.execution
        panels = self._panels(data)
        returns = compute_returns(data)
        adjustment = str(data.metadata.get("adjustment", Adjustment.SPLIT.value))
        loop = _Loop(
            config=config,
            data=data,
            options=options,
            strategy=strategy,
            panels=panels,
            returns=returns,
            vol=rolling_std(returns.close_to_close, VOL_WINDOW, 2),
            dividends=panels.get(C.DIVIDEND)
            if adjustment != Adjustment.TOTAL_RETURN.value
            else None,
            splits=panels.get(C.SPLIT) if adjustment == Adjustment.RAW.value else None,
            rolls=panels.get(ROLL_FIELD),
            has_expiries=any(
                m.expiry is not None and m.asset_class is AssetClass.OPTION
                for m in data.instrument_meta.values()
            ),
            vectorized_mode=strategy.implements_vectorized(),
            ledger=Ledger(
                data.instruments,
                data.instrument_meta,
                config.initial_capital,
                ex.cash_rate,
                options.periods_per_year,
            ),
            broker=SimulatedBroker(
                data.instruments,
                data.instrument_meta,
                ex.price,
                ex.lag_bars,
                ex.latency_bars,
                ex.max_participation,
                self.fill_model,
                self.slippage,
                self.cost_models,
            ),
        )
        if loop.vectorized_mode:
            options.report(0.1, "generating targets (adapter)")
            loop.exec_targets, loop.decisions, loop.reports = self._adapter_schedule(
                config, data, strategy, pipeline, options
            )
        elif not strategy.implements_event():
            raise EngineError("Strategy implements neither generate_targets nor on_bar")
        else:
            loop.event_overlays = [
                o.overlay for o in pipeline.overlays if o.overlay.implements_event()
            ]
            loop.overlay_states = [{} for _ in loop.event_overlays]
        loop.ctx = EventContext(
            panels,
            data.timestamps,
            data.instruments,
            loop.ledger,
            loop.broker,
            loop.equity_hist,
            {},
            options.periods_per_year,
        )
        return loop

    def _pre_trade(self, loop: _Loop, t: int) -> None:
        """Holding costs and cash interest over bar t, then corporate actions."""
        cash_returns = loop.options.extras.get(CASH_RETURNS_KEY)
        if isinstance(cash_returns, np.ndarray):
            loop.ledger.cash_rate = (
                float(np.nan_to_num(cash_returns[t])) * loop.ledger.periods_per_year
            )
        if t > 0:
            for cat, amount in loop.ledger.accrue(loop.prev_close, self.cost_models).items():
                loop.costs.setdefault(cat, np.zeros(loop.n_t))[t] += amount
        if loop.rolls is not None and t > 0:
            self._charge_roll(loop, t)
        if loop.has_expiries and t > 0:
            loop.ledger.settle_expiries(loop.data.timestamps[t], loop.prev_close)
        if loop.dividends is not None:
            loop.ledger.apply_dividends(loop.dividends[t])
        if loop.splits is not None:
            loop.ledger.apply_splits(loop.splits[t])

    def _charge_roll(self, loop: _Loop, t: int) -> None:
        """Commission for closing and reopening held futures on a roll bar."""
        assert loop.rolls is not None
        ledger = loop.ledger
        for j in np.flatnonzero((np.nan_to_num(loop.rolls[t]) > 0) & (ledger.units != 0)):
            inst = loop.broker.meta[ledger.instruments[j]]
            qty = float(ledger.units[j])
            price = float(loop.prev_close[j])
            fee = sum(
                m.commission(-qty, price, inst) + m.commission(qty, price, inst)
                for m in self.cost_models
            )
            ledger.cash -= fee
            loop.costs[COST_COMMISSION][t] += fee
            loop.traded[t] += 2.0 * abs(qty * price) * ledger.multipliers[j]

    def _trade(self, loop: _Loop, t: int) -> None:
        """Execute eligible orders against bar t."""
        p = loop.panels
        ledger = loop.ledger
        book = BarBook(
            t,
            loop.data.timestamps[t],
            p[C.OPEN][t],
            p[C.HIGH][t],
            p[C.LOW][t],
            p[C.CLOSE][t],
            p[C.VOLUME][t],
            loop.vol[t],
            p[C.SPREAD][t],
        )

        def equity_at(px: FloatArray) -> float:
            value = ledger.units * np.where(np.isfinite(px), px, 0.0) * ledger.multipliers
            return float(ledger.cash + value.sum())

        report = loop.broker.process(book, ledger.units.copy(), equity_at)
        ledger.apply_fills(report.fills)
        loop.fills.extend(report.fills)
        loop.costs[COST_COMMISSION][t] = report.commission
        loop.costs[COST_SLIPPAGE][t] = report.slippage
        cols = loop.broker.col
        loop.traded[t] = sum(
            abs(f.quantity * f.price) * ledger.multipliers[cols[f.instrument]] for f in report.fills
        )

    def _mark(self, loop: _Loop, t: int) -> None:
        """Mark to market at the close and record the state."""
        close = loop.panels[C.CLOSE][t]
        mark = np.where(np.isfinite(close), close, loop.prev_close)
        eq = loop.ledger.equity(mark)
        loop.equity[t] = eq
        loop.units[t] = loop.ledger.units
        loop.weights[t] = loop.ledger.market_values(mark) / eq if eq else 0.0
        loop.equity_hist.append(eq)
        loop.prev_close = mark

    def _decide(self, loop: _Loop, t: int) -> None:
        """Strategy (or adapter) decisions at the close of bar t."""
        ts = loop.data.timestamps
        instruments = loop.data.instruments
        rule = loop.config.execution.rebalance
        if loop.vectorized_mode:
            if loop.decisions[t]:
                row = loop.exec_targets[min(t + loop.config.execution.lag_bars, loop.n_t - 1)]
                targets = {
                    instruments[j]: float(row[j])
                    for j in range(len(instruments))
                    if abs(row[j]) > WEIGHT_EPS
                }
                self._submit_targets(loop.broker, loop.ledger, targets, t, ts[t])
            return
        ctx = loop.ctx
        ctx.advance(t)
        loop.strategy.on_bar(ctx)
        for overlay, st in zip(loop.event_overlays, loop.overlay_states, strict=True):
            saved = ctx.use_state(st)  # each overlay keeps its own state
            overlay.on_bar(ctx)
            ctx.use_state(saved)
        staged = ctx.staged_targets
        if staged is not None and (staged != loop.last_targets or rule is RebalanceRule.EVERY_BAR):
            self._submit_targets(loop.broker, loop.ledger, staged, t, ts[t])
            loop.last_targets = staged

    # ------------------------------------------------------------------ results

    def _result(self, loop: _Loop, t0: float) -> BacktestResult:
        config, data, options = loop.config, loop.data, loop.options
        ts, instruments = data.timestamps, data.instruments
        equity = loop.equity
        prev = np.concatenate(([config.initial_capital], equity[:-1]))
        total_cost = np.sum(np.vstack(list(loop.costs.values())), axis=0)
        close = data.panel(C.CLOSE)
        c2c = loop.returns.close_to_close
        bench = None
        if options.benchmark_id is not None and options.benchmark_id in instruments:
            bench = np.nan_to_num(c2c[:, instruments.index(options.benchmark_id)], nan=0.0)
            bench[0] = 0.0
        return BacktestResult(
            timestamps=ts,
            instruments=instruments,
            equity=equity,
            returns=equity / prev - 1.0,
            gross_returns=(equity + total_cost) / prev - 1.0,
            weights=loop.weights,
            target_weights=loop.exec_targets,
            positions=loop.units,
            prices=close,
            turnover=0.5 * loop.traded / np.where(prev > 0, prev, 1.0),
            costs=loop.costs,
            periods_per_year=options.periods_per_year,
            frequency=data.frequency,
            benchmark_returns=bench,
            benchmark_id=options.benchmark_id,
            trades=extract_trades(ts, instruments, loop.weights, c2c, equity, close, loop.units),
            orders=_orders_frame(loop.broker),
            fills=_fills_frame(loop.fills),
            overlay_reports=loop.reports,
            aux=aux_panels(data.fields, data.panel),
            risk_free=risk_free_series(options),
            config=config.model_dump(mode="json"),
            metadata={
                "engine": self.name,
                "return_basis": loop.returns.basis,
                "mode": "adapter" if loop.vectorized_mode else "on_bar",
                "instrument_meta": instrument_meta_json(data.instrument_meta),
                "execution": config.execution.model_dump(mode="json"),
                "timings": {"engine_seconds": round(time.perf_counter() - t0, 4)},
                "data": {
                    k: v
                    for k, v in data.metadata.items()
                    if isinstance(v, str | int | float | bool)
                },
            },
        )


@dataclass
class _Loop:
    """Mutable state of one event-loop run."""

    config: BacktestConfig
    data: MarketData
    options: RunOptions
    strategy: Strategy
    panels: dict[str, FloatArray]
    returns: ReturnPanels
    vol: FloatArray
    dividends: FloatArray | None
    splits: FloatArray | None
    rolls: FloatArray | None
    vectorized_mode: bool
    has_expiries: bool
    ledger: Ledger
    broker: SimulatedBroker
    ctx: EventContext = field(init=False)
    event_overlays: list[Overlay] = field(default_factory=list)
    overlay_states: list[dict[str, Any]] = field(default_factory=list)
    reports: tuple[OverlayReport, ...] = ()
    fills: list[Fill] = field(default_factory=list)
    last_targets: dict[str, float] | None = None
    exec_targets: FloatArray = field(init=False)
    decisions: BoolArray = field(init=False)
    equity: FloatArray = field(init=False)
    units: FloatArray = field(init=False)
    weights: FloatArray = field(init=False)
    traded: FloatArray = field(init=False)
    costs: dict[str, FloatArray] = field(init=False)
    equity_hist: list[float] = field(init=False)
    prev_close: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        n_t, n_i = self.n_t, len(self.data.instruments)
        self.exec_targets = np.zeros((n_t, n_i))
        self.decisions = np.zeros(n_t, dtype=bool)
        self.equity = np.zeros(n_t)
        self.units = np.zeros((n_t, n_i))
        self.weights = np.zeros((n_t, n_i))
        self.traded = np.zeros(n_t)
        self.costs = {COST_COMMISSION: np.zeros(n_t), COST_SLIPPAGE: np.zeros(n_t)}
        self.equity_hist = [self.config.initial_capital]
        first = self.panels[C.CLOSE][0]
        self.prev_close = np.where(np.isfinite(first), first, 0.0)

    @property
    def n_t(self) -> int:
        """Number of bars."""
        return len(self.data.timestamps)


def _orders_frame(broker: SimulatedBroker) -> pl.DataFrame:
    rows = []
    for state, status in broker.history:
        o = state.order
        rows.append(
            {
                "order_id": o.id,
                "timestamp": o.created_at,
                "instrument_id": o.instrument,
                "side": o.side.value,
                "quantity": float(o.quantity),
                "type": o.type.value,
                "tif": o.tif.value,
                "limit_price": o.limit_price,
                "stop_price": o.stop_price,
                "status": status,
                "tag": o.tag,
            }
        )
    for o in broker.open_orders():
        rows.append(
            {
                "order_id": o.id,
                "timestamp": o.created_at,
                "instrument_id": o.instrument,
                "side": o.side.value,
                "quantity": float(o.quantity),
                "type": o.type.value,
                "tif": o.tif.value,
                "limit_price": o.limit_price,
                "stop_price": o.stop_price,
                "status": "open",
                "tag": o.tag,
            }
        )
    if not rows:
        return pl.DataFrame({c: [] for c in ORDER_COLUMNS})
    stamps = np.array([r["timestamp"] for r in rows], dtype="datetime64[us]")
    frame = pl.DataFrame(
        [{k: v for k, v in r.items() if k != "timestamp"} for r in rows],
        schema_overrides={"limit_price": pl.Float64, "stop_price": pl.Float64},
    )
    return frame.with_columns(pl_times(stamps).alias("timestamp")).select(list(ORDER_COLUMNS))


def _fills_frame(fills: list[Fill]) -> pl.DataFrame:
    if not fills:
        return pl.DataFrame({c: [] for c in FILL_COLUMNS})
    stamps = np.array([f.timestamp for f in fills], dtype="datetime64[us]")
    frame = pl.DataFrame(
        {
            "order_id": [f.order_id for f in fills],
            "instrument_id": [f.instrument for f in fills],
            "price": [f.price for f in fills],
            "quantity": [f.quantity for f in fills],
            "commission": [f.commission for f in fills],
            "slippage": [f.slippage for f in fills],
        }
    )
    return frame.with_columns(pl_times(stamps).alias("timestamp")).select(list(FILL_COLUMNS))

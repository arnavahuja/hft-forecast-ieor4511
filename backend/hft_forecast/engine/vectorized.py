"""Vectorized engine: target weights in, portfolio path out, via array operations.

Timing (default ``lag_bars = 1``): a target decided with data up to the close of bar ``t``
trades at the close (or the open) of bar ``t + 1``. Costs are charged at the trade bar.

Cost equity approximation: per-unit costs (per-share commissions, fixed fees) depend on
equity, which depends on costs. We compute costs against the gross equity path, then once
more against the resulting net path (two passes). The residual error is second order in the
cost level and documented in ADR 0005.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl

from hft_forecast.core import columns as C
from hft_forecast.core.errors import EngineError
from hft_forecast.core.interfaces import (
    ConstructionContext,
    CostContext,
    CostModel,
    SlippageModel,
    Strategy,
)
from hft_forecast.core.results import (
    COST_SLIPPAGE,
    FILL_COLUMNS,
    ORDER_COLUMNS,
    BacktestResult,
    OverlayReport,
    aux_panels,
    instrument_meta_json,
)
from hft_forecast.core.run_config import BacktestConfig, ExecutionPrice
from hft_forecast.core.types import (
    BoolArray,
    FloatArray,
    MarketData,
    TargetFrame,
    TimeArray,
    pl_times,
)
from hft_forecast.engine.base import (
    RunOptions,
    align_targets,
    apply_membership,
    cash_input,
    risk_free_series,
)
from hft_forecast.engine.pipeline import Pipeline
from hft_forecast.engine.returns import ReturnPanels, compute_returns
from hft_forecast.engine.schedule import rebalance_mask
from hft_forecast.engine.simulation import SegmentPath, compound_pairs, simulate_segments
from hft_forecast.engine.trades import extract_trades

COST_PASSES: Final = 2
ROLL_FIELD: Final = "roll"
"""Data field set to 1 on bars where a continuous futures series rolls."""
TRADE_EPS: Final = 1e-12


@dataclass(frozen=True)
class BarPath:
    """Per-bar portfolio path before costs."""

    gross: FloatArray
    start_weights: FloatArray
    end_weights: FloatArray
    trades: FloatArray
    exec_targets: FloatArray
    rebalance: BoolArray


def lag_targets(values: FloatArray, lag: int) -> FloatArray:
    """Shift decision-time targets forward by ``lag`` bars (leading rows become 0)."""
    out = np.zeros_like(values)
    if lag < len(values):
        out[lag:] = np.nan_to_num(values[: len(values) - lag], nan=0.0)
    return out


class VectorizedEngine:
    """Fast engine for research, sweeps and cross-sectional work.

    Args:
        cost_models: Commission/fee/borrow models.
        slippage: Slippage model.
        cost_scale: Multiplier on all costs (cost-sensitivity research).
    """

    name = "vectorized"

    def __init__(
        self,
        cost_models: Sequence[CostModel] = (),
        slippage: SlippageModel | None = None,
        cost_scale: float = 1.0,
    ) -> None:
        self.cost_models = tuple(cost_models)
        self.slippage = slippage
        self.cost_scale = cost_scale

    # ------------------------------------------------------------------ simulation

    def bar_path(
        self,
        config: BacktestConfig,
        timestamps: TimeArray,
        returns: ReturnPanels,
        targets: FloatArray,
        cash_rate: float | FloatArray,
        periods_per_year: float,
        tradable: BoolArray | None = None,
    ) -> BarPath:
        """Simulate the pre-cost path for decision-time ``targets`` ``(T, N)``.

        ``cash_rate`` is an annual rate, or per-bar cash returns ``(T,)`` (risk-free series).
        """
        ex = config.execution
        exec_targets = lag_targets(targets, ex.lag_bars)
        if tradable is not None:
            exec_targets = np.where(tradable, exec_targets, 0.0)
        mask = rebalance_mask(
            timestamps, ex.rebalance, ex.rebalance_every_n, ex.lag_bars, exec_targets
        )
        if isinstance(cash_rate, np.ndarray):
            cash_per_bar = np.nan_to_num(np.asarray(cash_rate, dtype=np.float64), nan=0.0).copy()
        else:
            cash_per_bar = np.full(len(timestamps), cash_rate / periods_per_year)
        if len(cash_per_bar):
            cash_per_bar[0] = 0.0  # no holding period before the first bar
        if ex.price is ExecutionPrice.CLOSE:
            path = simulate_segments(returns.close_to_close, exec_targets, mask, cash_per_bar)
            return BarPath(
                path.returns, path.start_of_step, path.post_trade, path.trades, exec_targets, mask
            )
        return self._open_path(timestamps, returns, exec_targets, mask, cash_per_bar)

    @staticmethod
    def _open_path(
        timestamps: TimeArray,
        returns: ReturnPanels,
        exec_targets: FloatArray,
        mask: BoolArray,
        cash_per_bar: FloatArray,
    ) -> BarPath:
        n_t, n_i = exec_targets.shape
        half = np.empty((2 * n_t, n_i))
        half[0::2] = returns.overnight
        half[1::2] = returns.intraday
        tgt = np.zeros((2 * n_t, n_i))
        tgt[0::2] = exec_targets
        rebal = np.zeros(2 * n_t, dtype=bool)
        rebal[0::2] = mask
        cash = np.zeros(2 * n_t)
        cash[1::2] = cash_per_bar
        path: SegmentPath = simulate_segments(half, tgt, rebal, cash)
        gross = compound_pairs(path.returns[0::2], path.returns[1::2])
        start = path.start_of_step[0::2]
        del timestamps  # grid is implied by the arrays
        return BarPath(gross, start, path.post_trade[1::2], path.trades[0::2], exec_targets, mask)

    def simulate_zero_cost(
        self,
        config: BacktestConfig,
        data: MarketData,
        returns: ReturnPanels,
        periods_per_year: float,
        tradable: BoolArray | None,
        cash: float | FloatArray | None = None,
    ) -> Callable[[TargetFrame], FloatArray]:
        """Return a function giving zero-cost returns of a target frame (for overlays)."""

        def simulate(targets: TargetFrame) -> FloatArray:
            aligned = targets.reindex(data.timestamps, data.instruments)
            path = self.bar_path(
                config,
                data.timestamps,
                returns,
                aligned.values,
                config.execution.cash_rate if cash is None else cash,
                periods_per_year,
                tradable,
            )
            return path.gross

        return simulate

    # ------------------------------------------------------------------ costs

    def _costs(self, ctx: CostContext) -> tuple[FloatArray, dict[str, FloatArray]]:
        total = np.zeros_like(ctx.trades)
        by_cat: dict[str, FloatArray] = {}
        models: list[tuple[str, CostModel | SlippageModel]] = [
            (m.category, m) for m in self.cost_models
        ]
        if self.slippage is not None:
            models.append((COST_SLIPPAGE, self.slippage))
        for category, model in models:
            frac = np.nan_to_num(model.vectorized(ctx), nan=0.0) * self.cost_scale
            if frac.shape != total.shape:
                raise EngineError(f"Cost model '{category}' returned wrong shape")
            total += frac
            by_cat[category] = by_cat.get(category, np.zeros_like(total)) + frac
        return total, by_cat

    # ------------------------------------------------------------------ run

    def run(
        self,
        config: BacktestConfig,
        data: MarketData,
        strategy: Strategy,
        pipeline: Pipeline,
        options: RunOptions,
    ) -> BacktestResult:
        """Run a vectorized backtest."""
        if not strategy.implements_vectorized():
            raise EngineError(
                f"Strategy '{strategy.plugin_name}' has no vectorized form; use the event engine"
            )
        t0 = time.perf_counter()
        strat_data = (
            data.select_instruments(options.strategy_instruments)
            if options.strategy_instruments is not None
            else data
        )
        options.report(0.1, "generating targets")
        raw = align_targets(strategy.generate_targets(strat_data), data)
        raw = apply_membership(raw, options.membership)
        targets, reports = self.targets_after_pipeline(config, data, raw, pipeline, options)
        options.report(0.6, "simulating")
        result = self.simulate(config, data, targets, options, reports)
        result.metadata["timings"] = {"engine_seconds": round(time.perf_counter() - t0, 4)}
        return result

    def targets_after_pipeline(
        self,
        config: BacktestConfig,
        data: MarketData,
        raw: TargetFrame,
        pipeline: Pipeline,
        options: RunOptions,
    ) -> tuple[TargetFrame, tuple[OverlayReport, ...]]:
        """Run constructor and overlays on aligned raw strategy output."""
        ppy = options.periods_per_year
        weights = pipeline.construct(raw, ConstructionContext(data, ppy))
        weights = align_targets(weights, data)
        returns = compute_returns(data)
        tradable = np.isfinite(data.panel(C.CLOSE))
        simulate = self.simulate_zero_cost(
            config, data, returns, ppy, tradable, cash_input(config, options)
        )
        options.report(0.4, "applying overlays")
        final, reports = pipeline.apply_overlays(
            weights,
            data,
            ppy,
            options.benchmark_id,
            config.execution.lag_bars,
            simulate,
            config.initial_capital,
        )
        return align_targets(final, data), reports

    def simulate(
        self,
        config: BacktestConfig,
        data: MarketData,
        targets: TargetFrame,
        options: RunOptions,
        overlay_reports: tuple[OverlayReport, ...] = (),
    ) -> BacktestResult:
        """Simulate final decision-time weights into a :class:`BacktestResult`."""
        ppy = options.periods_per_year
        ts = data.timestamps
        close = data.panel(C.CLOSE)
        returns = compute_returns(data)
        tradable = np.isfinite(close)
        path = self.bar_path(
            config, ts, returns, targets.values, cash_input(config, options), ppy, tradable
        )
        exec_price = (
            data.panel(C.OPEN)
            if config.execution.price is ExecutionPrice.OPEN and data.has_field(C.OPEN)
            else close
        )
        volumes = data.panel(C.VOLUME) if data.has_field(C.VOLUME) else None
        meta = data.instrument_meta
        multipliers = np.array([meta[i].multiplier if i in meta else 1.0 for i in data.instruments])
        classes = tuple(
            str(meta[i].asset_class) if i in meta else "equity" for i in data.instruments
        )

        # futures rolls: holdings are closed and reopened, so they trade twice (costs only)
        cost_trades = path.trades
        if data.has_field(ROLL_FIELD):
            roll = np.nan_to_num(data.panel(ROLL_FIELD), nan=0.0) > 0
            roll_trades = np.where(roll, 2.0 * np.abs(path.start_weights), 0.0)
            cost_trades = np.sign(path.trades) * (np.abs(path.trades) + roll_trades)
        capital = config.initial_capital
        equity_pre = capital * np.cumprod(1.0 + path.gross)
        net = path.gross
        cost_frac = np.zeros_like(path.trades)
        by_cat: dict[str, FloatArray] = {}
        for _ in range(COST_PASSES):
            prev_equity = np.concatenate(([capital], (capital * np.cumprod(1.0 + net))[:-1]))
            equity_pre = prev_equity * (1.0 + path.gross)
            ctx = CostContext(
                timestamps=ts,
                instruments=data.instruments,
                trades=cost_trades,
                holdings=path.start_weights,
                prices=exec_price,
                volumes=volumes,
                equity=equity_pre,
                returns=returns.close_to_close,
                multipliers=multipliers,
                asset_classes=classes,
                periods_per_year=ppy,
            )
            cost_frac, by_cat = self._costs(ctx)
            net = (1.0 + path.gross) * (1.0 - cost_frac.sum(axis=1)) - 1.0
        equity = capital * np.cumprod(1.0 + net)
        costs = {k: v.sum(axis=1) * equity_pre for k, v in by_cat.items()}

        weights = path.end_weights
        with np.errstate(divide="ignore", invalid="ignore"):
            units = np.nan_to_num(weights * equity[:, None] / (close * multipliers[None, :]))
        turnover = 0.5 * np.abs(cost_trades).sum(axis=1)
        bench = None
        if options.benchmark_id is not None and options.benchmark_id in data.instruments:
            j = data.instruments.index(options.benchmark_id)
            bench = np.nan_to_num(returns.close_to_close[:, j], nan=0.0)
            bench[0] = 0.0
        trades = extract_trades(
            ts, data.instruments, weights, returns.close_to_close, equity, close, units
        )
        orders, fills = self._orders(
            ts, data.instruments, path.trades, equity_pre, exec_price, multipliers, cost_frac
        )
        return BacktestResult(
            timestamps=ts,
            instruments=data.instruments,
            equity=equity,
            returns=net,
            gross_returns=path.gross,
            weights=weights,
            target_weights=path.exec_targets,
            positions=units,
            prices=close,
            turnover=turnover,
            costs=costs,
            periods_per_year=ppy,
            frequency=data.frequency,
            benchmark_returns=bench,
            benchmark_id=options.benchmark_id,
            trades=trades,
            orders=orders,
            fills=fills,
            overlay_reports=overlay_reports,
            aux=aux_panels(data.fields, data.panel),
            risk_free=risk_free_series(options),
            config=config.model_dump(mode="json"),
            metadata={
                "engine": self.name,
                "return_basis": returns.basis,
                "rebalances": int(path.rebalance.sum()),
                "instrument_meta": instrument_meta_json(data.instrument_meta),
                "execution": config.execution.model_dump(mode="json"),
                "data": {
                    k: v
                    for k, v in data.metadata.items()
                    if isinstance(v, str | int | float | bool)
                },
            },
        )

    @staticmethod
    def _orders(
        ts: TimeArray,
        instruments: tuple[str, ...],
        trades: FloatArray,
        equity_pre: FloatArray,
        prices: FloatArray,
        multipliers: FloatArray,
        cost_frac: FloatArray,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        rows, cols = np.nonzero(np.abs(trades) > TRADE_EPS)
        if not len(rows):
            return (
                pl.DataFrame({c: [] for c in ORDER_COLUMNS}),
                pl.DataFrame({c: [] for c in FILL_COLUMNS}),
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            qty = trades[rows, cols] * equity_pre[rows] / (prices[rows, cols] * multipliers[cols])
        qty = np.nan_to_num(qty)
        ids = [f"v{r}-{c}" for r, c in zip(rows.tolist(), cols.tolist(), strict=True)]
        stamps = pl_times(ts[rows])
        inst = [instruments[c] for c in cols.tolist()]
        orders = pl.DataFrame(
            {
                "order_id": ids,
                "timestamp": stamps,
                "instrument_id": inst,
                "side": np.where(qty > 0, "buy", "sell").tolist(),
                "quantity": np.abs(qty),
                "type": ["market"] * len(ids),
                "tif": ["day"] * len(ids),
                "limit_price": [None] * len(ids),
                "stop_price": [None] * len(ids),
                "status": ["filled"] * len(ids),
                "tag": ["rebalance"] * len(ids),
            },
            schema_overrides={"limit_price": pl.Float64, "stop_price": pl.Float64},
        )
        fills = pl.DataFrame(
            {
                "order_id": ids,
                "timestamp": stamps,
                "instrument_id": inst,
                "price": prices[rows, cols],
                "quantity": qty,
                "commission": cost_frac[rows, cols] * equity_pre[rows],
                "slippage": np.zeros(len(ids)),
            }
        )
        return orders, fills

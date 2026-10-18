"""Backtest result container shared by both engines, analytics, storage and the API."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Final

import numpy as np
import polars as pl

from hft_forecast.core.types import FloatArray, Frequency, TimeArray

TRADE_COLUMNS: Final = (
    "instrument_id",
    "direction",
    "entry_time",
    "exit_time",
    "entry_price",
    "exit_price",
    "quantity",
    "bars_held",
    "pnl",
    "return",
    "mae",
    "mfe",
    "is_open",
)
ORDER_COLUMNS: Final = (
    "order_id",
    "timestamp",
    "instrument_id",
    "side",
    "quantity",
    "type",
    "tif",
    "limit_price",
    "stop_price",
    "status",
    "tag",
)
FILL_COLUMNS: Final = (
    "order_id",
    "timestamp",
    "instrument_id",
    "price",
    "quantity",
    "commission",
    "slippage",
)

AUX_FIELDS: Final[tuple[str, ...]] = ()
"""Data fields engines copy into ``BacktestResult.aux`` when present (none today)."""

COST_COMMISSION: Final = "commission"
COST_SLIPPAGE: Final = "slippage"
COST_BORROW: Final = "borrow"
COST_FINANCING: Final = "financing"
COST_FEES: Final = "fees"


def empty_frame(columns: tuple[str, ...]) -> pl.DataFrame:
    """Empty frame with the given columns (all null dtype)."""
    return pl.DataFrame({c: [] for c in columns})


@dataclass(frozen=True, eq=False)
class OverlayReport:
    """What one overlay changed.

    Attributes:
        name: Overlay plugin name.
        position: 0-based position in the pipeline.
        gross_before: Gross exposure of targets entering the overlay (per decision bar).
        gross_after: Gross exposure after the overlay.
        returns_before: Zero-cost simulated returns of the incoming targets.
        returns_after: Zero-cost simulated returns after the overlay.
        mean_abs_change: Mean absolute weight change per bar.
        cells_changed: Number of (bar, instrument) cells changed.
        notes: Free-form notes the overlay recorded.
    """

    name: str
    position: int
    gross_before: FloatArray
    gross_after: FloatArray
    returns_before: FloatArray
    returns_after: FloatArray
    mean_abs_change: float
    cells_changed: int
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, eq=False)
class BacktestResult:
    """Outcome of a backtest. Identical structure for both engines.

    Arrays are aligned to ``timestamps`` (length ``T``) and ``instruments`` (length ``N``).
    ``returns[t]`` is the net return over bar ``t`` (from the close of ``t-1`` to the close of
    ``t``); ``returns[0]`` is zero. ``weights[t]`` are end-of-bar weights after trading.

    Attributes:
        timestamps: Bar timestamps.
        instruments: Instrument ids (including hedge instruments added by overlays).
        equity: Equity at the close of each bar.
        returns: Net returns.
        gross_returns: Returns before transaction costs and financing.
        weights: End-of-bar weights ``(T, N)``.
        target_weights: Targets in force at each bar (after lag) ``(T, N)``.
        positions: Units held at the end of each bar ``(T, N)``.
        prices: Close prices ``(T, N)`` used for valuation.
        turnover: One-way turnover per bar, as a fraction of equity.
        costs: Cost per bar in currency, by category (commission, slippage, borrow, ...).
        benchmark_returns: Benchmark returns aligned to ``timestamps`` or ``None``.
        benchmark_id: Benchmark instrument id.
        trades: Round-trip trades table (see ``TRADE_COLUMNS``).
        orders: Orders table (event engine; derived rebalances for the vectorized engine).
        fills: Fills table.
        overlay_reports: One report per overlay, in pipeline order.
        aux: Extra ``(T, N)`` panels carried from the data (option Greeks, strikes).
        risk_free: Per-bar risk-free returns (``config.risk_free``) or ``None``.
        periods_per_year: Annualization factor derived from frequency and calendar.
        frequency: Bar frequency.
        config: Serialized ``BacktestConfig``.
        metadata: Engine name, timings, warnings, flags (e.g. ``model_priced``).
    """

    timestamps: TimeArray
    instruments: tuple[str, ...]
    equity: FloatArray
    returns: FloatArray
    gross_returns: FloatArray
    weights: FloatArray
    target_weights: FloatArray
    positions: FloatArray
    prices: FloatArray
    turnover: FloatArray
    costs: dict[str, FloatArray]
    periods_per_year: float
    frequency: Frequency = Frequency.D1
    benchmark_returns: FloatArray | None = None
    benchmark_id: str | None = None
    trades: pl.DataFrame = field(default_factory=lambda: empty_frame(TRADE_COLUMNS))
    orders: pl.DataFrame = field(default_factory=lambda: empty_frame(ORDER_COLUMNS))
    fills: pl.DataFrame = field(default_factory=lambda: empty_frame(FILL_COLUMNS))
    overlay_reports: tuple[OverlayReport, ...] = ()
    aux: dict[str, FloatArray] = field(default_factory=dict)
    risk_free: FloatArray | None = None
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---- derived series

    @property
    def n_bars(self) -> int:
        """Number of bars."""
        return len(self.timestamps)

    @property
    def initial_capital(self) -> float:
        """Equity before the first bar."""
        return float(self.equity[0] / (1.0 + self.returns[0])) if self.n_bars else 0.0

    @property
    def long_exposure(self) -> FloatArray:
        """Sum of positive weights per bar."""
        return np.clip(self.weights, 0.0, None).sum(axis=1)

    @property
    def short_exposure(self) -> FloatArray:
        """Sum of absolute negative weights per bar."""
        return -np.clip(self.weights, None, 0.0).sum(axis=1)

    @property
    def gross_exposure(self) -> FloatArray:
        """Gross exposure (long + short) per bar."""
        return np.abs(self.weights).sum(axis=1)

    @property
    def net_exposure(self) -> FloatArray:
        """Net exposure (long - short) per bar."""
        return self.weights.sum(axis=1)

    @property
    def n_positions(self) -> FloatArray:
        """Number of non-zero positions per bar."""
        return (np.abs(self.weights) > 0).sum(axis=1).astype(np.float64)

    @property
    def total_costs(self) -> FloatArray:
        """Total cost in currency per bar across categories."""
        if not self.costs:
            return np.zeros(self.n_bars)
        return np.sum(np.vstack(list(self.costs.values())), axis=0)

    @property
    def gross_equity(self) -> FloatArray:
        """Equity path had there been no costs."""
        return self.initial_capital * np.cumprod(1.0 + self.gross_returns)

    @property
    def drawdown(self) -> FloatArray:
        """Drawdown from running peak, as a non-positive fraction."""
        peak = np.maximum.accumulate(np.concatenate(([self.initial_capital], self.equity)))[1:]
        return self.equity / peak - 1.0

    def window(self, start: int, stop: int) -> BacktestResult:
        """Sub-result over bars ``[start, stop)``, with the first bar's return re-based to 0.

        Used for date-range brushing: metrics recalculated for a selected window.
        """
        sl = slice(start, stop)
        rets = self.returns[sl].copy()
        gross = self.gross_returns[sl].copy()
        base_equity = float(self.equity[start - 1]) if start > 0 else self.initial_capital
        if len(rets):
            rets[0] = 0.0
            gross[0] = 0.0
        equity = base_equity * np.cumprod(1.0 + rets)
        bench = None
        if self.benchmark_returns is not None:
            bench = self.benchmark_returns[sl].copy()
            if len(bench):
                bench[0] = 0.0
        ts = self.timestamps[sl]
        trades = self.trades
        if trades.height and len(ts):
            lo, hi = ts[0], ts[-1]
            trades = trades.filter(
                (pl.col("entry_time") >= _to_py(lo)) & (pl.col("entry_time") <= _to_py(hi))
            )
        return replace(
            self,
            timestamps=ts,
            equity=equity,
            returns=rets,
            gross_returns=gross,
            weights=self.weights[sl],
            target_weights=self.target_weights[sl],
            positions=self.positions[sl],
            prices=self.prices[sl],
            turnover=self.turnover[sl],
            costs={k: v[sl] for k, v in self.costs.items()},
            aux={k: v[sl] for k, v in self.aux.items()},
            risk_free=self.risk_free[sl] if self.risk_free is not None else None,
            benchmark_returns=bench,
            trades=trades,
        )


def _to_py(value: np.datetime64) -> Any:
    import datetime as _dt

    micros = int(value.astype("datetime64[us]").astype(np.int64))
    return _dt.datetime(1970, 1, 1, tzinfo=_dt.UTC) + _dt.timedelta(microseconds=micros)


def aux_panels(data_fields: tuple[str, ...], panel: Any) -> dict[str, FloatArray]:
    """Aux panels for the fields in :data:`AUX_FIELDS` that exist in the data.

    Args:
        data_fields: Field names present in the data.
        panel: ``field -> (T, N) array`` accessor.
    """
    return {f: np.asarray(panel(f), dtype=np.float64) for f in AUX_FIELDS if f in data_fields}


def instrument_meta_json(meta: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Instrument metadata as JSON for ``BacktestResult.metadata``."""
    return {k: v.model_dump(mode="json") for k, v in meta.items()}

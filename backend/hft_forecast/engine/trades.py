"""Round-trip trade extraction from a weight path.

A trade is a maximal run of bars in which an instrument's end-of-bar weight is non-zero with
a constant sign. It is entered at the close of the first bar of the run and exited at the
close of the bar where the weight goes to zero or flips sign. PnL sums the bar-by-bar
contributions ``w_{t-1} * r_t * V_{t-1}`` over the holding period, so trade PnLs add up to
the portfolio's gross PnL.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt
import polars as pl

from hft_forecast.core.results import TRADE_COLUMNS
from hft_forecast.core.types import FloatArray, IntArray, TimeArray, pl_times

WEIGHT_EPS: Final = 1e-10


def _runs(sign: npt.NDArray[np.int8]) -> tuple[IntArray, IntArray]:
    """Start and end (inclusive) indices of runs of constant non-zero sign."""
    padded = np.concatenate(([0], sign, [0]))
    change = np.flatnonzero(padded[1:] != padded[:-1])
    starts = change[:-1]
    ends = change[1:] - 1
    keep = sign[starts] != 0
    return starts[keep], ends[keep]


def extract_trades(
    timestamps: TimeArray,
    instruments: tuple[str, ...],
    weights: FloatArray,
    asset_returns: FloatArray,
    equity: FloatArray,
    prices: FloatArray,
    units: FloatArray,
) -> pl.DataFrame:
    """Build the round-trip trade table.

    Args:
        timestamps: Bar timestamps ``(T,)``.
        instruments: Instrument ids ``(N,)``.
        weights: End-of-bar weights ``(T, N)``.
        asset_returns: Close-to-close returns ``(T, N)``.
        equity: Equity at each close ``(T,)``.
        prices: Close prices ``(T, N)``.
        units: Units held at each close ``(T, N)``.

    Returns:
        Trades with columns ``TRADE_COLUMNS``.
    """
    n_t = len(timestamps)
    rets = np.nan_to_num(asset_returns, nan=0.0)
    prev_eq = np.concatenate(([equity[0]], equity[:-1])) if n_t else equity
    rows: dict[str, list[object]] = {c: [] for c in TRADE_COLUMNS}
    entry_idx: list[int] = []
    exit_idx: list[int] = []
    for j, inst in enumerate(instruments):
        w = weights[:, j]
        sign = np.where(np.abs(w) > WEIGHT_EPS, np.sign(w), 0).astype(np.int8)
        starts, ends = _runs(sign)
        if not len(starts):
            continue
        held_prev = np.concatenate(([0.0], w[:-1]))
        pnl_bar = held_prev * rets[:, j] * prev_eq
        cum_pnl = np.concatenate(([0.0], np.cumsum(pnl_bar)))
        exits = np.minimum(ends + 1, n_t - 1)
        is_open = ends + 1 > n_t - 1
        pnl = cum_pnl[exits + 1] - cum_pnl[starts + 1]
        index = np.cumprod(1.0 + rets[:, j])
        direction = sign[starts].astype(np.float64)
        # excursions: path of the position's return relative to entry, bars (start, exit]
        mae = np.empty(len(starts))
        mfe = np.empty(len(starts))
        for k, (s, e) in enumerate(zip(starts, exits, strict=True)):
            rel = direction[k] * (index[s + 1 : e + 1] / index[s] - 1.0)
            mae[k] = min(0.0, float(rel.min())) if rel.size else 0.0
            mfe[k] = max(0.0, float(rel.max())) if rel.size else 0.0
        trade_ret = direction * (index[exits] / index[starts] - 1.0)
        rows["instrument_id"].extend([inst] * len(starts))
        rows["direction"].extend(["long" if d > 0 else "short" for d in direction])
        rows["entry_price"].extend(prices[starts, j].tolist())
        rows["exit_price"].extend(prices[exits, j].tolist())
        rows["quantity"].extend(np.abs(units[starts, j]).tolist())
        rows["bars_held"].extend((exits - starts).tolist())
        rows["pnl"].extend(pnl.tolist())
        rows["return"].extend(trade_ret.tolist())
        rows["mae"].extend(mae.tolist())
        rows["mfe"].extend(mfe.tolist())
        rows["is_open"].extend(is_open.tolist())
        entry_idx.extend(starts.tolist())
        exit_idx.extend(exits.tolist())
    entry = np.array(entry_idx, dtype=np.int64)
    exit_ = np.array(exit_idx, dtype=np.int64)
    rows["entry_time"] = list(pl_times(timestamps[entry]) if len(entry) else [])
    rows["exit_time"] = list(pl_times(timestamps[exit_]) if len(exit_) else [])
    schema: dict[str, pl.DataType | type[pl.DataType]] = {
        "instrument_id": pl.String,
        "direction": pl.String,
        "entry_time": pl.Datetime("us", "UTC"),
        "exit_time": pl.Datetime("us", "UTC"),
        "entry_price": pl.Float64,
        "exit_price": pl.Float64,
        "quantity": pl.Float64,
        "bars_held": pl.Int64,
        "pnl": pl.Float64,
        "return": pl.Float64,
        "mae": pl.Float64,
        "mfe": pl.Float64,
        "is_open": pl.Boolean,
    }
    return pl.DataFrame(rows, schema=schema).sort("entry_time")

"""Corporate action adjustments: raw, split-adjusted and total-return series.

Canonical frames may carry ``split_ratio`` (new shares per old share on the ex-date, 0 or 1
meaning no split) and ``dividend`` (cash per share on the ex-date) columns.

Note on lookahead: adjusted *levels* embed future corporate actions (a backward adjustment
rescales history using later events). Returns computed from adjusted series are unaffected,
which is what the engines use. Strategies that compare raw price *levels* to constants
should request ``raw`` data.
"""

from __future__ import annotations

import polars as pl

from hft_forecast.core import columns as C
from hft_forecast.core.types import Adjustment

_PRICE_COLS = (C.OPEN, C.HIGH, C.LOW, C.CLOSE)


def _ratio_expr() -> pl.Expr:
    ratio = pl.col(C.SPLIT).fill_null(1.0)
    return pl.when(ratio <= 0).then(1.0).otherwise(ratio)


def future_split_factor(frame: pl.DataFrame) -> pl.Series:
    """Product of split ratios strictly after each row, per instrument.

    Multiplying a split-adjusted price by this factor gives the raw traded price.
    """
    if C.SPLIT not in frame.columns:
        return pl.Series("f", [1.0] * frame.height)
    out = frame.sort(C.KEY_COLUMNS).with_columns(
        (_ratio_expr().reverse().cum_prod().reverse().over(C.INSTRUMENT) / _ratio_expr()).alias(
            "_f"
        )
    )
    return out.get_column("_f")


def split_to_raw(frame: pl.DataFrame) -> pl.DataFrame:
    """Convert split-adjusted OHLCV to raw traded prices and volumes."""
    frame = frame.sort(C.KEY_COLUMNS)
    factor = future_split_factor(frame)
    exprs = [(pl.col(c) * factor).alias(c) for c in _PRICE_COLS if c in frame.columns]
    if C.VOLUME in frame.columns:
        exprs.append((pl.col(C.VOLUME) / factor).alias(C.VOLUME))
    if C.DIVIDEND in frame.columns:
        exprs.append((pl.col(C.DIVIDEND) * factor).alias(C.DIVIDEND))
    return frame.with_columns(exprs)


def dividend_factor(frame: pl.DataFrame) -> pl.Series:
    """Backward total-return factor from dividends on split-adjusted closes.

    ``f_t = prod_{s > t} (1 - D_s / C_{s-1})`` per instrument.
    """
    if C.DIVIDEND not in frame.columns:
        return pl.Series("f", [1.0] * frame.height)
    out = frame.sort(C.KEY_COLUMNS).with_columns(
        (1.0 - pl.col(C.DIVIDEND).fill_null(0.0) / pl.col(C.CLOSE).shift(1).over(C.INSTRUMENT))
        .fill_null(1.0)
        .alias("_step")
    )
    out = out.with_columns(
        (pl.col("_step").reverse().cum_prod().reverse().over(C.INSTRUMENT) / pl.col("_step")).alias(
            "_f"
        )
    )
    return out.get_column("_f")


def split_to_total_return(frame: pl.DataFrame) -> pl.DataFrame:
    """Convert split-adjusted OHLC to total-return adjusted OHLC.

    Uses ``adj_close / close`` when an ``adj_close`` column is present, else dividends.
    """
    frame = frame.sort(C.KEY_COLUMNS)
    if C.ADJ_CLOSE in frame.columns:
        factor = frame.get_column(C.ADJ_CLOSE) / frame.get_column(C.CLOSE)
        factor = factor.fill_null(1.0).fill_nan(1.0)
    else:
        factor = dividend_factor(frame)
    exprs = [(pl.col(c) * factor).alias(c) for c in _PRICE_COLS if c in frame.columns]
    return frame.with_columns(exprs)


def convert(frame: pl.DataFrame, stored: Adjustment, target: Adjustment) -> pl.DataFrame:
    """Convert a canonical frame between adjustment conventions.

    Supported paths: split -> raw, split -> total_return, raw -> split (needs split ratios),
    identity. Unsupported conversions return the frame unchanged.
    """
    if stored is target:
        return frame
    if stored is Adjustment.SPLIT and target is Adjustment.RAW:
        return split_to_raw(frame)
    if stored is Adjustment.SPLIT and target is Adjustment.TOTAL_RETURN:
        return split_to_total_return(frame)
    if stored is Adjustment.RAW:
        frame = frame.sort(C.KEY_COLUMNS)
        factor = future_split_factor(frame)
        split_adj = frame.with_columns(
            [(pl.col(c) / factor).alias(c) for c in _PRICE_COLS if c in frame.columns]
        )
        return convert(split_adj, Adjustment.SPLIT, target)
    return frame

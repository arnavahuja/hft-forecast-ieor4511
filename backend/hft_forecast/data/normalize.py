"""Conversion of arbitrary frames to the canonical long schema."""

from __future__ import annotations

import polars as pl

from hft_forecast.core import columns as C
from hft_forecast.core.errors import DataValidationError
from hft_forecast.core.types import UTC_DTYPE

NUMERIC_TYPES = (
    pl.Float32,
    pl.Float64,
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Decimal,
    pl.Boolean,
)


def normalize_frame(frame: pl.DataFrame, *, dedupe: bool = True, sort: bool = True) -> pl.DataFrame:
    """Cast key columns, make numeric fields ``Float64``, sort and de-duplicate.

    Non-numeric extra columns (e.g. sector labels) are kept as strings.

    Raises:
        DataValidationError: If key columns are missing.
    """
    missing = [c for c in C.KEY_COLUMNS if c not in frame.columns]
    if missing:
        raise DataValidationError(f"Missing key columns {missing}")
    ts = frame.schema[C.TIMESTAMP]
    if isinstance(ts, pl.Datetime):
        ts_expr = (
            pl.col(C.TIMESTAMP).dt.convert_time_zone("UTC")
            if ts.time_zone
            else pl.col(C.TIMESTAMP).dt.replace_time_zone("UTC")
        )
    else:
        ts_expr = pl.col(C.TIMESTAMP).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    exprs = [ts_expr.cast(UTC_DTYPE).alias(C.TIMESTAMP), pl.col(C.INSTRUMENT).cast(pl.String)]
    for name, dtype in frame.schema.items():
        if name in C.KEY_COLUMNS:
            continue
        if isinstance(dtype, NUMERIC_TYPES) or dtype in NUMERIC_TYPES:
            exprs.append(pl.col(name).cast(pl.Float64).fill_nan(None))
        else:
            exprs.append(pl.col(name).cast(pl.String))
    out = frame.with_columns(exprs).drop_nulls(list(C.KEY_COLUMNS))
    if dedupe:
        out = out.unique(subset=list(C.KEY_COLUMNS), keep="last", maintain_order=True)
    return out.sort(C.KEY_COLUMNS) if sort else out

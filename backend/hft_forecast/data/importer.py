"""Local data import: read a file, guess a column mapping, convert to canonical form.

The mapping is saved as YAML next to the imported Parquet file so a re-import is one click.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Final

import polars as pl
from pydantic import BaseModel, Field

from hft_forecast.core import columns as C
from hft_forecast.core.errors import DataError, DataValidationError
from hft_forecast.core.types import Adjustment, Frequency
from hft_forecast.data.normalize import normalize_frame

SUPPORTED_SUFFIXES: Final = (
    ".csv",
    ".txt",
    ".parquet",
    ".pq",
    ".xlsx",
    ".xls",
    ".feather",
    ".arrow",
    ".ipc",
)
PREVIEW_ROWS: Final = 20
INFER_ROWS: Final = 10_000

_CANDIDATES: Final[dict[str, tuple[str, ...]]] = {
    C.TIMESTAMP: ("timestamp", "datetime", "date", "time", "dt", "day", "trade_date"),
    C.SYMBOL: (
        "symbol",
        "ticker",
        "instrument",
        "instrument_id",
        "permno",
        "asset",
        "code",
        "secid",
        "name",
    ),
    C.OPEN: ("open", "o", "open_price", "opening"),
    C.HIGH: ("high", "h", "high_price"),
    C.LOW: ("low", "l", "low_price"),
    C.CLOSE: ("close", "c", "close_price", "last", "price", "px_last", "prc"),
    C.ADJ_CLOSE: ("adj_close", "adjclose", "adjusted_close", "adj close"),
    C.VOLUME: ("volume", "v", "vol", "shares_traded"),
    C.DIVIDEND: ("dividend", "dividends", "div"),
    C.SPLIT: ("split", "splits", "split_ratio", "stock splits"),
}


class ImportMapping(BaseModel):
    """How to convert a user file to the canonical schema.

    Attributes:
        timestamp: Source column with dates/times.
        symbol: Source column with instrument ids (``None`` = single-instrument file).
        fixed_symbol: Instrument id to use when ``symbol`` is ``None``.
        columns: Canonical field -> source column (open, high, low, close, volume, ...).
        extra: Extra source columns to keep as fields (renamed to snake_case).
        frequency: Bar frequency.
        timezone: Timezone of naive timestamps in the file.
        datetime_format: Optional ``strptime`` format.
        adjustment: Whether prices are raw, split-adjusted or total-return adjusted.
        name: Display name of the dataset.
    """

    timestamp: str
    symbol: str | None = None
    fixed_symbol: str | None = None
    columns: dict[str, str] = Field(default_factory=dict)
    extra: list[str] = Field(default_factory=list)
    frequency: Frequency = Frequency.D1
    timezone: str = "UTC"
    datetime_format: str | None = None
    adjustment: Adjustment = Adjustment.SPLIT
    name: str = ""


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def read_any(path: Path, n_rows: int | None = None) -> pl.DataFrame:
    """Read CSV, Parquet, Excel or Feather into polars."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise DataError(
            f"Unsupported file type '{suffix}'", details={"supported": list(SUPPORTED_SUFFIXES)}
        )
    if suffix in (".csv", ".txt"):
        return pl.read_csv(
            path, n_rows=n_rows, infer_schema_length=INFER_ROWS, try_parse_dates=True
        )
    if suffix in (".parquet", ".pq"):
        frame = pl.read_parquet(path)
    elif suffix in (".xlsx", ".xls"):
        frame = pl.read_excel(path)
    else:
        frame = pl.read_ipc(path)
    return frame.head(n_rows) if n_rows is not None else frame


def guess_mapping(frame: pl.DataFrame, file_stem: str = "") -> ImportMapping:
    """Guess the column mapping from column names (the user can correct it)."""
    normalized = {_norm(c): c for c in frame.columns}
    found: dict[str, str] = {}
    for field, candidates in _CANDIDATES.items():
        for cand in candidates:
            source = normalized.get(_norm(cand))
            if source is not None and source not in found.values():
                found[field] = source
                break
    timestamp = found.pop(C.TIMESTAMP, None)
    if timestamp is None:
        temporal = [c for c, t in frame.schema.items() if t.is_temporal()]
        timestamp = temporal[0] if temporal else frame.columns[0]
    symbol = found.pop(C.SYMBOL, None)
    used = {timestamp, *found.values(), *([symbol] if symbol else [])}
    extra = [c for c in frame.columns if c not in used]
    return ImportMapping(
        timestamp=timestamp,
        symbol=symbol,
        fixed_symbol=None if symbol else (file_stem.upper() or "ASSET"),
        columns=found,
        extra=extra,
        name=file_stem,
    )


def preview(path: Path, rows: int = PREVIEW_ROWS) -> dict[str, Any]:
    """Preview a file: columns, dtypes, first rows and a guessed mapping."""
    frame = read_any(path, n_rows=max(rows, 1000))
    return {
        "columns": frame.columns,
        "dtypes": {c: str(t) for c, t in frame.schema.items()},
        "rows": frame.head(rows).to_dicts(),
        "mapping": guess_mapping(frame, path.stem).model_dump(mode="json"),
    }


def _parse_timestamps(frame: pl.DataFrame, mapping: ImportMapping) -> pl.Expr:
    col = pl.col(mapping.timestamp)
    dtype = frame.schema[mapping.timestamp]
    if dtype == pl.Date:
        expr = col.cast(pl.Datetime("us"))
    elif isinstance(dtype, pl.Datetime):
        expr = col
    elif dtype.is_integer():
        expr = col.cast(pl.String).str.to_datetime(mapping.datetime_format or "%Y%m%d")
    else:
        expr = col.cast(pl.String).str.to_datetime(
            mapping.datetime_format, strict=False, time_unit="us"
        )
    if isinstance(dtype, pl.Datetime) and dtype.time_zone:
        return expr.dt.convert_time_zone("UTC")
    return expr.dt.replace_time_zone(mapping.timezone).dt.convert_time_zone("UTC")


def apply_mapping(frame: pl.DataFrame, mapping: ImportMapping) -> pl.DataFrame:
    """Convert a raw frame to the canonical long schema.

    Raises:
        DataValidationError: If required columns are missing or timestamps do not parse.
    """
    needed = [mapping.timestamp, *mapping.columns.values(), *mapping.extra]
    if mapping.symbol:
        needed.append(mapping.symbol)
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise DataValidationError(f"Columns not found in file: {missing}")
    if C.CLOSE not in mapping.columns:
        raise DataValidationError("A 'close' column mapping is required")
    exprs = [_parse_timestamps(frame, mapping).alias(C.TIMESTAMP)]
    if mapping.symbol:
        exprs.append(pl.col(mapping.symbol).cast(pl.String).alias(C.INSTRUMENT))
    else:
        exprs.append(pl.lit(mapping.fixed_symbol or "ASSET").alias(C.INSTRUMENT))
    exprs.extend(pl.col(src).alias(dst) for dst, src in mapping.columns.items())
    taken = {C.TIMESTAMP, C.INSTRUMENT, *mapping.columns}
    for src in mapping.extra:
        dst = _norm(src)
        if dst in taken:
            dst = f"x_{dst}"
        taken.add(dst)
        exprs.append(pl.col(src).alias(dst))
    out = frame.select(exprs)
    bad = out.get_column(C.TIMESTAMP).null_count()
    if bad == out.height:
        raise DataValidationError(
            "Could not parse any timestamps; set datetime_format",
            details={"column": mapping.timestamp},
        )
    return normalize_frame(out, dedupe=False, sort=False)

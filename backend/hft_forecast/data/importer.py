"""Local data import: read a file, guess a column mapping, convert to canonical form.

The mapping is saved as YAML next to the imported Parquet file so a re-import is one click.

Level-1 quote files and level-N book files take the same path: map the columns the file
actually has and :func:`derive_quote_fields` fills in everything that follows mechanically
(level 1 of the book is the NBBO, the mid is the average of the quotes, and so on). Imports
are stored bar-end labelled like every other intraday dataset in this project (§3.4).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final

import polars as pl
from pydantic import BaseModel, Field

from hft_forecast.core import columns as C
from hft_forecast.core.errors import DataError, DataValidationError
from hft_forecast.core.types import BAR_LABEL_KEY, Adjustment, Frequency
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

MAX_BOOK_LEVELS: Final = 20
"""Deepest book an import may declare. The level-1 fields are the NBBO."""

BAR_LABEL_END: Final = "end"
BAR_LABEL_START: Final = "start"
BPS: Final = 1e4
LEVEL_ONE_TOLERANCE: Final = 1e-9

_BOOK_PRICE_ALIASES: Final = ("px", "price")
_BOOK_SIZE_ALIASES: Final = ("sz", "size", "qty", "quantity", "vol", "volume")
_LEVEL_ONE: Final = (
    (C.BID, "bid_px_1"),
    (C.ASK, "ask_px_1"),
    (C.BID_SIZE, "bid_sz_1"),
    (C.ASK_SIZE, "ask_sz_1"),
)

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
    C.BID: ("bid", "best_bid", "bid_price", "bid_px"),
    C.ASK: ("ask", "best_ask", "ask_price", "ask_px", "offer"),
    C.BID_SIZE: ("bid_size", "bid_sz", "bidsiz", "best_bidsiz", "bid_qty"),
    C.ASK_SIZE: ("ask_size", "ask_sz", "asksiz", "best_asksiz", "ask_qty"),
    C.MID: ("mid", "mid_price", "mid_px", "midpoint"),
    C.MICROPRICE: ("microprice", "micro_price", "weighted_mid"),
    C.SPREAD: ("spread",),
    C.SPREAD_BPS: ("spread_bps",),
    C.MID_TWAP: ("mid_twap", "twap"),
    C.N_QUOTES: ("n_quotes", "num_quotes", "quote_count", "updates", "n_updates"),
    C.N_TRADES: ("n_trades", "num_trades", "trade_count"),
    C.VWAP: ("vwap",),
    C.IS_STALE: ("is_stale", "stale"),
}


class ImportMapping(BaseModel):
    """How to convert a user file to the canonical schema.

    Attributes:
        timestamp: Source column with dates/times.
        symbol: Source column with instrument ids (``None`` = single-instrument file).
        fixed_symbol: Instrument id to use when ``symbol`` is ``None``.
        columns: Canonical field -> source column (open, high, low, close, bid, ask, ...).
        book_levels: Depth of the order book to import (0 for level-1 or OHLCV files).
        book_columns: Canonical book column (``bid_px_1``, ``bid_sz_1``, ...) -> source column.
        extra: Extra source columns to keep as fields (renamed to snake_case).
        frequency: Bar frequency.
        bar_label: Whether the file's timestamps sit at the bar ``end`` or ``start``; a
            start-labelled intraday file is shifted to the end convention on import.
        timezone: Timezone of naive timestamps in the file.
        datetime_format: Optional ``strptime`` format.
        adjustment: Whether prices are raw, split-adjusted or total-return adjusted.
        name: Display name of the dataset.
    """

    timestamp: str
    symbol: str | None = None
    fixed_symbol: str | None = None
    columns: dict[str, str] = Field(default_factory=dict)
    book_levels: int = Field(0, ge=0, le=MAX_BOOK_LEVELS)
    book_columns: dict[str, str] = Field(default_factory=dict)
    extra: list[str] = Field(default_factory=list)
    frequency: Frequency = Frequency.D1
    bar_label: str = Field(BAR_LABEL_END, pattern=f"^({BAR_LABEL_END}|{BAR_LABEL_START})$")
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


def guess_book_columns(columns: Iterable[str]) -> dict[str, str]:
    """Book columns present in a file, keyed by canonical name (``bid_px_1``, ``bid_sz_1``, ...).

    Levels are taken in order and stop at the first incomplete one: a book missing the ask
    size of level 4 is a 3-level book, not a 4-level book with a hole in it.
    """
    normalized = {_norm(c): c for c in columns}
    found: dict[str, str] = {}
    for level in range(1, MAX_BOOK_LEVELS + 1):
        level_map: dict[str, str] = {}
        for side in ("bid", "ask"):
            for kind, aliases in (("px", _BOOK_PRICE_ALIASES), ("sz", _BOOK_SIZE_ALIASES)):
                names = (f"{side}_{alias}_{level}" for alias in aliases)
                source = next((normalized[n] for n in names if n in normalized), None)
                if source is None:
                    return found
                level_map[f"{side}_{kind}_{level}"] = source
        found.update(level_map)
    return found


def guess_mapping(frame: pl.DataFrame, file_stem: str = "") -> ImportMapping:
    """Guess the column mapping from column names (the user can correct it)."""
    normalized = {_norm(c): c for c in frame.columns}
    book = guess_book_columns(frame.columns)
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
    used = {timestamp, *found.values(), *book.values(), *([symbol] if symbol else [])}
    extra = [c for c in frame.columns if c not in used]
    return ImportMapping(
        timestamp=timestamp,
        symbol=symbol,
        fixed_symbol=None if symbol else (file_stem.upper() or "ASSET"),
        columns=found,
        book_levels=len(book) // len(_LEVEL_ONE),
        book_columns=book,
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


def _book_sources(mapping: ImportMapping) -> dict[str, str]:
    """Canonical book column -> source column for the declared depth.

    Raises:
        DataValidationError: If a level the mapping declares has an unmapped column.
    """
    wanted = C.book_columns(mapping.book_levels)
    missing = [c for c in wanted if c not in mapping.book_columns]
    if missing:
        raise DataValidationError(
            f"book_levels is {mapping.book_levels} but these columns are not mapped: {missing}",
            details={"mapped": sorted(mapping.book_columns)},
        )
    return {c: mapping.book_columns[c] for c in wanted}


def _shift_to_bar_end(frame: pl.DataFrame, mapping: ImportMapping) -> pl.DataFrame:
    """Move start-labelled intraday timestamps to the bar end, the convention on disk (§3.4)."""
    seconds = mapping.frequency.seconds
    if mapping.bar_label == BAR_LABEL_END or seconds is None:
        return frame
    return frame.with_columns(
        (pl.col(C.TIMESTAMP) + pl.duration(seconds=seconds)).alias(C.TIMESTAMP)
    )


def _check_level_one(frame: pl.DataFrame) -> None:
    """Level 1 of the book *is* the NBBO; a file mapping both must agree.

    Raises:
        DataValidationError: If a mapped level-1 field differs from the book column.
    """
    for field_name, book in _LEVEL_ONE:
        if field_name not in frame.columns or book not in frame.columns:
            continue
        off = (pl.col(field_name) - pl.col(book)).abs() > LEVEL_ONE_TOLERANCE
        bad = int(frame.select(off.sum()).item() or 0)
        if bad:
            raise DataValidationError(
                f"'{field_name}' and '{book}' disagree on {bad} rows; level 1 of the book is "
                "the best quote, so map one or the other",
                details={"rows": bad},
            )


def _fill(frame: pl.DataFrame, exprs: dict[str, pl.Expr]) -> pl.DataFrame:
    """Add the columns of ``exprs`` that the frame does not already carry."""
    missing = [expr.alias(name) for name, expr in exprs.items() if name not in frame.columns]
    return frame.with_columns(missing) if missing else frame


def derive_quote_fields(frame: pl.DataFrame) -> pl.DataFrame:
    """Fill the quote fields that follow mechanically from the ones already present.

    Level 1 of the book gives the NBBO, the quotes give the mid, the microprice and the
    spread, and the mid gives ``close`` for a quote dataset. Nothing already in the frame is
    overwritten, so a file that carries its own mid keeps it.
    """
    out = _fill(frame, {name: pl.col(book) for name, book in _LEVEL_ONE if book in frame.columns})
    if C.BID in out.columns and C.ASK in out.columns:
        bid, ask = pl.col(C.BID), pl.col(C.ASK)
        out = _fill(out, {C.MID: (bid + ask) / 2.0, C.SPREAD: ask - bid})
        if C.BID_SIZE in out.columns and C.ASK_SIZE in out.columns:
            bid_size, ask_size = pl.col(C.BID_SIZE), pl.col(C.ASK_SIZE)
            out = _fill(
                out,
                {
                    C.MICROPRICE: (bid * ask_size + ask * bid_size) / (bid_size + ask_size),
                },
            )
    if C.MID in out.columns:
        if C.SPREAD in out.columns:
            out = _fill(out, {C.SPREAD_BPS: pl.col(C.SPREAD) / pl.col(C.MID) * BPS})
        out = _fill(out, {C.CLOSE: pl.col(C.MID)})
    return out


def quote_metadata(mapping: ImportMapping, columns: Iterable[str]) -> dict[str, str]:
    """Metadata an imported dataset carries: bar labelling and what ``close`` means.

    Always ``end``, because :func:`apply_mapping` shifts a start-labelled file. ``close`` is
    the mid whenever it was derived from the quotes rather than mapped from the file.
    """
    derived_mid = C.CLOSE not in mapping.columns and C.MID in set(columns)
    return {BAR_LABEL_KEY: BAR_LABEL_END, C.PRICE_BASIS_KEY: "mid" if derived_mid else "trade"}


def apply_mapping(frame: pl.DataFrame, mapping: ImportMapping) -> pl.DataFrame:
    """Convert a raw frame to the canonical long schema.

    Raises:
        DataValidationError: If required columns are missing, level 1 disagrees with the
            book, timestamps do not parse, or no price column can be derived.
    """
    book = _book_sources(mapping)
    needed = [mapping.timestamp, *mapping.columns.values(), *book.values(), *mapping.extra]
    if mapping.symbol:
        needed.append(mapping.symbol)
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise DataValidationError(f"Columns not found in file: {missing}")
    exprs = [_parse_timestamps(frame, mapping).alias(C.TIMESTAMP)]
    if mapping.symbol:
        exprs.append(pl.col(mapping.symbol).cast(pl.String).alias(C.INSTRUMENT))
    else:
        exprs.append(pl.lit(mapping.fixed_symbol or "ASSET").alias(C.INSTRUMENT))
    exprs.extend(pl.col(src).alias(dst) for dst, src in mapping.columns.items())
    exprs.extend(pl.col(src).alias(dst) for dst, src in book.items())
    taken = {C.TIMESTAMP, C.INSTRUMENT, *mapping.columns, *book}
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
    out = _shift_to_bar_end(out, mapping)
    _check_level_one(out)
    out = derive_quote_fields(out)
    if C.CLOSE not in out.columns:
        raise DataValidationError(
            "Map a 'close' column, or the quote (bid/ask) or book columns the mid follows from"
        )
    return normalize_frame(out, dedupe=False, sort=False)

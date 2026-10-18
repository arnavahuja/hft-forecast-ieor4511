"""Immutable domain value objects shared by every layer.

Conventions:
    * Timestamps are timezone-aware UTC. Polars columns use ``Datetime("us", "UTC")``;
      NumPy arrays use ``datetime64[us]`` and are UTC by convention (NumPy has no tz).
    * Money is ``float64``. Tests compare money with ``MONEY_RTOL``/``MONEY_ATOL``.
    * Wide numeric panels are ``float64`` arrays shaped ``(n_timestamps, n_instruments)``
      with ``NaN`` for missing values.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator

from hft_forecast.core import columns as C
from hft_forecast.core.errors import DataError

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]
TimeArray = npt.NDArray[np.datetime64]

UTC_DTYPE: Final = pl.Datetime("us", "UTC")
NP_TIME_UNIT: Final = "datetime64[us]"

MONEY_RTOL: Final = 1e-9
"""Relative tolerance used when comparing money values in tests."""
MONEY_ATOL: Final = 1e-6
"""Absolute tolerance (in currency units) used when comparing money values in tests."""


# --------------------------------------------------------------------------- enums


class AssetClass(StrEnum):
    """Broad instrument category."""

    EQUITY = "equity"
    ETF = "etf"
    FUTURE = "future"
    OPTION = "option"
    INDEX = "index"
    FX = "fx"
    OTHER = "other"


class Frequency(StrEnum):
    """Bar frequency. Minimum supported bar size is one second."""

    S1 = "1s"
    S5 = "5s"
    S10 = "10s"
    S30 = "30s"
    MIN1 = "1min"
    MIN5 = "5min"
    MIN15 = "15min"
    H1 = "1h"
    D1 = "1d"
    W1 = "1w"
    MO1 = "1mo"

    @property
    def is_intraday(self) -> bool:
        """Whether bars are shorter than one trading session."""
        return self in _INTRADAY_SECONDS

    @property
    def is_subminute(self) -> bool:
        """Whether bars are shorter than one minute."""
        seconds = self.seconds
        return seconds is not None and seconds < SECONDS_PER_MINUTE

    @property
    def seconds(self) -> int | None:
        """Bar length in seconds for intraday frequencies, else ``None``."""
        return _INTRADAY_SECONDS.get(self)

    @property
    def minutes(self) -> int | None:
        """Bar length in whole minutes, or ``None`` when not intraday or not a whole minute."""
        seconds = self.seconds
        if seconds is None or seconds % SECONDS_PER_MINUTE:
            return None
        return seconds // SECONDS_PER_MINUTE


SECONDS_PER_MINUTE: Final = 60

_INTRADAY_SECONDS: Final[dict[Frequency, int]] = {
    Frequency.S1: 1,
    Frequency.S5: 5,
    Frequency.S10: 10,
    Frequency.S30: 30,
    Frequency.MIN1: 60,
    Frequency.MIN5: 300,
    Frequency.MIN15: 900,
    Frequency.H1: 3600,
}

BAR_LABEL_KEY: Final = "bar_label"
"""MarketData metadata key: ``"end"`` (right-closed, labelled at the bar end) or ``"start"``."""


class Adjustment(StrEnum):
    """Price adjustment convention of a series."""

    RAW = "raw"
    SPLIT = "split"
    TOTAL_RETURN = "total_return"


class OptionRight(StrEnum):
    """Option right."""

    CALL = "call"
    PUT = "put"


class Side(StrEnum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    """Supported order types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(StrEnum):
    """Order time in force."""

    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"


class TargetKind(StrEnum):
    """What the numbers in a :class:`TargetFrame` mean."""

    WEIGHTS = "weights"
    SIGNALS = "signals"


# --------------------------------------------------------------------------- instruments


class Instrument(BaseModel):
    """A tradable (or reference) instrument.

    Attributes:
        id: Stable unique identifier, e.g. ``"yahoo:SPY"`` or ``"crsp:14593"``.
        symbol: Display ticker.
        asset_class: Broad category.
        currency: ISO currency code. Only USD is supported in v1.
        multiplier: Contract multiplier (1 for cash equities).
        exchange: Listing exchange code, if known.
        underlying: Underlying instrument id for derivatives.
        strike: Option strike.
        expiry: Option or future expiry date.
        right: Option right.
        root: Futures root symbol, e.g. ``"ES"``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    symbol: str
    asset_class: AssetClass = AssetClass.EQUITY
    currency: str = "USD"
    multiplier: float = 1.0
    exchange: str | None = None
    name: str | None = None
    underlying: str | None = None
    strike: float | None = None
    expiry: date | None = None
    right: OptionRight | None = None
    root: str | None = None


# --------------------------------------------------------------------------- requests


class DataRequest(BaseModel):
    """A normalized request for market data.

    Either ``instruments`` or ``universe`` must be given.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    dataset: str
    instruments: tuple[str, ...] = ()
    universe: str | None = None
    start: date
    end: date
    frequency: Frequency = Frequency.D1
    fields: tuple[str, ...] = ()
    adjustment: Adjustment = Adjustment.SPLIT
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("instruments", "fields")
    @classmethod
    def _sorted_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted(set(value)))

    def cache_key(self) -> str:
        """Stable hash of the normalized request, used as the cache key."""
        payload = self.model_dump(mode="json")
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:24]

    def key_without_dates(self) -> str:
        """Hash of the request ignoring the date range, used for partial-overlap reuse."""
        payload = self.model_dump(mode="json", exclude={"start", "end"})
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:24]


class DatasetInfo(BaseModel):
    """Description of a dataset offered by a data source."""

    model_config = ConfigDict(frozen=True)

    source: str
    dataset: str
    description: str
    frequencies: tuple[Frequency, ...] = (Frequency.D1,)
    asset_classes: tuple[AssetClass, ...] = (AssetClass.EQUITY,)
    fields: tuple[str, ...] = C.OHLCV
    survivorship_bias_free: bool = False
    quality: str = "standard"
    notes: str = ""


# --------------------------------------------------------------------------- time helpers


def to_utc_datetime(value: date | datetime) -> datetime:
    """Convert a date or naive/aware datetime to an aware UTC datetime."""
    import datetime as _dt

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=_dt.UTC)
        return value.astimezone(_dt.UTC)
    return datetime(value.year, value.month, value.day, tzinfo=_dt.UTC)


def np_times(values: pl.Series) -> TimeArray:
    """Convert a UTC polars datetime series to a ``datetime64[us]`` NumPy array."""
    return values.dt.replace_time_zone(None).cast(pl.Datetime("us")).to_numpy().astype(NP_TIME_UNIT)


def pl_times(values: TimeArray, name: str = C.TIMESTAMP) -> pl.Series:
    """Convert a ``datetime64`` array (UTC by convention) to a UTC polars series."""
    return pl.Series(name, values.astype(NP_TIME_UNIT)).dt.replace_time_zone("UTC")


# --------------------------------------------------------------------------- market data


class MarketData:
    """Long-format market data table with helper views to wide panels.

    The table has columns ``timestamp`` (UTC), ``instrument_id`` and one column per
    field. Instances are treated as immutable: every transformation returns a new object.

    Args:
        frame: Long-format polars frame.
        frequency: Bar frequency.
        instruments: Optional instrument metadata keyed by id.
        metadata: Free-form metadata (source, adjustment, quality flags, ...).
    """

    def __init__(
        self,
        frame: pl.DataFrame,
        frequency: Frequency = Frequency.D1,
        instruments: dict[str, Instrument] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        missing = [c for c in C.KEY_COLUMNS if c not in frame.columns]
        if missing:
            raise DataError(f"MarketData is missing key columns: {missing}")
        frame = frame.with_columns(pl.col(C.TIMESTAMP).cast(UTC_DTYPE)).sort(C.KEY_COLUMNS)
        self._frame = frame
        self._frequency = frequency
        self._instrument_meta: dict[str, Instrument] = dict(instruments or {})
        self._metadata: dict[str, Any] = dict(metadata or {})
        self._panel_cache: dict[str, FloatArray] = {}
        self._timestamps: TimeArray | None = None
        self._instruments: tuple[str, ...] | None = None

    # ---- basic properties

    @property
    def frame(self) -> pl.DataFrame:
        """The underlying long-format frame."""
        return self._frame

    @property
    def frequency(self) -> Frequency:
        """Bar frequency."""
        return self._frequency

    @property
    def metadata(self) -> dict[str, Any]:
        """A copy of the free-form metadata."""
        return dict(self._metadata)

    @property
    def instrument_meta(self) -> dict[str, Instrument]:
        """Instrument metadata keyed by id (may be incomplete)."""
        return dict(self._instrument_meta)

    @property
    def fields(self) -> tuple[str, ...]:
        """Data fields (all non-key columns)."""
        return tuple(c for c in self._frame.columns if c not in C.KEY_COLUMNS)

    @property
    def instruments(self) -> tuple[str, ...]:
        """Sorted unique instrument ids."""
        if self._instruments is None:
            self._instruments = tuple(
                sorted(self._frame.get_column(C.INSTRUMENT).unique().to_list())
            )
        return self._instruments

    @property
    def timestamps(self) -> TimeArray:
        """Sorted unique timestamps as ``datetime64[us]``."""
        if self._timestamps is None:
            ts = self._frame.get_column(C.TIMESTAMP).unique().sort()
            self._timestamps = np_times(ts)
        return self._timestamps

    def __len__(self) -> int:
        return self._frame.height

    def is_empty(self) -> bool:
        """Whether the table has no rows."""
        return self._frame.height == 0

    # ---- wide views

    def panel(self, field_name: str) -> FloatArray:
        """Wide ``(timestamps x instruments)`` float panel for one field.

        Missing observations are ``NaN``. The result is cached and must not be mutated.
        """
        if field_name not in self.fields:
            raise DataError(
                f"Field '{field_name}' not in market data", details={"available": list(self.fields)}
            )
        cached = self._panel_cache.get(field_name)
        if cached is not None:
            return cached
        wide = self.wide(field_name)
        values = (
            wide.select(list(self.instruments)).cast(pl.Float64).to_numpy()
            if self.instruments
            else np.empty((len(self.timestamps), 0))
        )
        arr = np.ascontiguousarray(values, dtype=np.float64)
        arr.setflags(write=False)
        self._panel_cache[field_name] = arr
        return arr

    def aligned_panel(
        self, field_name: str, timestamps: TimeArray, instruments: tuple[str, ...]
    ) -> FloatArray:
        """Panel of a field on an arbitrary grid (missing cells are ``NaN``)."""
        grid = TargetFrame(self.timestamps, self.instruments, self.panel(field_name))
        return np.array(grid.reindex(timestamps, instruments).values)

    def latest_text(self, field_name: str) -> dict[str, str]:
        """Last non-null value of a (text) field per instrument, e.g. a sector label."""
        if field_name not in self.fields:
            return {}
        last = (
            self._frame.select(C.INSTRUMENT, field_name)
            .drop_nulls(field_name)
            .group_by(C.INSTRUMENT, maintain_order=True)
            .last()
        )
        return {str(k): str(v) for k, v in last.iter_rows()}

    def has_field(self, field_name: str) -> bool:
        """Whether a field exists."""
        return field_name in self.fields

    def wide(self, field_name: str) -> pl.DataFrame:
        """Wide polars frame: ``timestamp`` column plus one column per instrument."""
        base = pl.DataFrame({C.TIMESTAMP: pl_times(self.timestamps)})
        if not self.instruments:
            return base
        pivot = self._frame.select(C.TIMESTAMP, C.INSTRUMENT, field_name).pivot(
            on=C.INSTRUMENT, index=C.TIMESTAMP, values=field_name, aggregate_function="first"
        )
        out = base.join(pivot, on=C.TIMESTAMP, how="left")
        for inst in self.instruments:
            if inst not in out.columns:
                out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias(inst))
        return out.select([C.TIMESTAMP, *self.instruments])

    # ---- transformations

    def _derive(self, frame: pl.DataFrame) -> MarketData:
        keep = set(frame.get_column(C.INSTRUMENT).unique().to_list()) if frame.height else set()
        meta = {k: v for k, v in self._instrument_meta.items() if k in keep}
        return MarketData(frame, self._frequency, meta, self._metadata)

    def truncate(self, end: np.datetime64 | datetime) -> MarketData:
        """Rows with ``timestamp <= end``."""
        end_dt = _as_datetime(end)
        return self._derive(self._frame.filter(pl.col(C.TIMESTAMP) <= end_dt))

    def between(self, start: date | datetime | None, end: date | datetime | None) -> MarketData:
        """Rows within ``[start, end]`` (inclusive; ``end`` date covers the whole day)."""
        frame = self._frame
        if start is not None:
            frame = frame.filter(pl.col(C.TIMESTAMP) >= to_utc_datetime(start))
        if end is not None:
            end_dt = to_utc_datetime(end)
            if not isinstance(end, datetime):
                end_dt = end_dt.replace(hour=23, minute=59, second=59, microsecond=999999)
            frame = frame.filter(pl.col(C.TIMESTAMP) <= end_dt)
        return self._derive(frame)

    def select_instruments(self, ids: tuple[str, ...] | list[str]) -> MarketData:
        """Restrict to a subset of instruments."""
        return self._derive(self._frame.filter(pl.col(C.INSTRUMENT).is_in(list(ids))))

    def with_metadata(self, **updates: Any) -> MarketData:
        """Return a copy with metadata entries added or replaced."""
        meta = {**self._metadata, **updates}
        return MarketData(self._frame, self._frequency, self._instrument_meta, meta)

    def concat(self, other: MarketData) -> MarketData:
        """Union of two tables (rows of ``other`` win on key collisions)."""
        cols = sorted(set(self._frame.columns) | set(other.frame.columns))
        left = self._frame.with_columns(
            [pl.lit(None).alias(c) for c in cols if c not in self._frame.columns]
        ).select(cols)
        right = other.frame.with_columns(
            [pl.lit(None).alias(c) for c in cols if c not in other.frame.columns]
        ).select(cols)
        merged = pl.concat([left, right], how="vertical_relaxed").unique(
            subset=list(C.KEY_COLUMNS), keep="last"
        )
        meta = {**self._instrument_meta, **other.instrument_meta}
        return MarketData(merged, self._frequency, meta, self._metadata)


def _as_datetime(value: np.datetime64 | datetime) -> datetime:
    if isinstance(value, np.datetime64):
        micros = int(value.astype("datetime64[us]").astype(np.int64))
        import datetime as _dt

        return datetime(1970, 1, 1, tzinfo=_dt.UTC) + _dt.timedelta(microseconds=micros)
    return to_utc_datetime(value)


# --------------------------------------------------------------------------- targets


@dataclass(frozen=True, eq=False)
class TargetFrame:
    """Per-instrument targets over time.

    ``values[t, i]`` is the target for instrument ``instruments[i]`` decided using
    information available at ``timestamps[t]`` (inclusive). ``NaN`` means "no position"
    for weights and "no opinion" for signals. The engine applies the execution lag.

    Attributes:
        timestamps: Decision timestamps, ``datetime64[us]``, strictly increasing.
        instruments: Instrument ids for the columns.
        values: ``(T, N)`` float array.
        kind: Whether values are portfolio weights or raw signals.
        extra_instruments: Hedge instruments added by overlays (subset of ``instruments``).
    """

    timestamps: TimeArray
    instruments: tuple[str, ...]
    values: FloatArray
    kind: TargetKind = TargetKind.WEIGHTS
    extra_instruments: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.values.shape != (len(self.timestamps), len(self.instruments)):
            raise DataError(
                "TargetFrame shape mismatch",
                details={
                    "values": list(self.values.shape),
                    "timestamps": len(self.timestamps),
                    "instruments": len(self.instruments),
                },
            )
        values = np.array(self.values, dtype=np.float64, copy=True)
        values.setflags(write=False)
        object.__setattr__(self, "values", values)

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_timestamps, n_instruments)``."""
        return (len(self.timestamps), len(self.instruments))

    def filled(self) -> FloatArray:
        """Values with ``NaN`` replaced by zero (a writable copy)."""
        return np.nan_to_num(self.values, nan=0.0)

    def replace(
        self,
        values: FloatArray | None = None,
        kind: TargetKind | None = None,
        instruments: tuple[str, ...] | None = None,
        extra_instruments: tuple[str, ...] | None = None,
    ) -> TargetFrame:
        """Return a copy with some attributes replaced."""
        return TargetFrame(
            timestamps=self.timestamps,
            instruments=self.instruments if instruments is None else instruments,
            values=self.values if values is None else values,
            kind=self.kind if kind is None else kind,
            extra_instruments=(
                self.extra_instruments if extra_instruments is None else extra_instruments
            ),
        )

    def reindex(self, timestamps: TimeArray, instruments: tuple[str, ...]) -> TargetFrame:
        """Align to a new grid. New cells are ``NaN``; rows are matched exactly."""
        out = np.full((len(timestamps), len(instruments)), np.nan)
        row_pos = {t: i for i, t in enumerate(self.timestamps.astype(np.int64).tolist())}
        col_pos = {s: j for j, s in enumerate(self.instruments)}
        rows_new = [i for i, t in enumerate(timestamps.astype(np.int64).tolist()) if t in row_pos]
        rows_old = [row_pos[int(t)] for t in timestamps.astype(np.int64)[rows_new]]
        cols_new = [j for j, s in enumerate(instruments) if s in col_pos]
        cols_old = [col_pos[instruments[j]] for j in cols_new]
        if rows_new and cols_new:
            out[np.ix_(rows_new, cols_new)] = self.values[np.ix_(rows_old, cols_old)]
        return TargetFrame(timestamps, instruments, out, self.kind, self.extra_instruments)

    def to_long(self) -> pl.DataFrame:
        """Long polars frame ``(timestamp, instrument_id, value)`` without ``NaN`` rows."""
        n_t, n_i = self.shape
        frame = pl.DataFrame(
            {
                C.TIMESTAMP: pl_times(np.repeat(self.timestamps, n_i)),
                C.INSTRUMENT: np.tile(np.array(self.instruments, dtype=object), n_t).tolist()
                if n_i
                else [],
                "value": self.values.reshape(-1),
            }
        )
        return frame.filter(pl.col("value").is_not_nan())

    @classmethod
    def empty_like(cls, data: MarketData, kind: TargetKind = TargetKind.WEIGHTS) -> TargetFrame:
        """All-``NaN`` targets on the grid of ``data``."""
        shape = (len(data.timestamps), len(data.instruments))
        return cls(data.timestamps, data.instruments, np.full(shape, np.nan), kind)


# --------------------------------------------------------------------------- orders & fills


@dataclass(frozen=True)
class Order:
    """An order to trade an instrument.

    Attributes:
        id: Unique order id.
        instrument: Instrument id.
        side: Buy or sell.
        quantity: Positive number of units (shares or contracts).
        type: Order type.
        tif: Time in force.
        limit_price: Limit price for limit and stop-limit orders.
        stop_price: Trigger price for stop and stop-limit orders.
        created_at: Timestamp when the order was created (``datetime64[us]``).
        tag: Free-form tag (e.g. which overlay created it).
    """

    id: str
    instrument: str
    side: Side
    quantity: float
    type: OrderType = OrderType.MARKET
    tif: TimeInForce = TimeInForce.GTC
    limit_price: float | None = None
    stop_price: float | None = None
    created_at: np.datetime64 | None = None
    tag: str = ""

    @property
    def signed_quantity(self) -> float:
        """Quantity with sign (+ buy, - sell)."""
        return self.quantity if self.side is Side.BUY else -self.quantity


@dataclass(frozen=True)
class Fill:
    """An execution of (part of) an order."""

    order_id: str
    instrument: str
    timestamp: np.datetime64
    price: float
    quantity: float
    commission: float = 0.0
    slippage: float = 0.0

    @property
    def notional(self) -> float:
        """Signed notional (quantity is signed)."""
        return self.price * self.quantity


@dataclass(frozen=True)
class Position:
    """Holding in one instrument."""

    instrument: str
    quantity: float
    avg_price: float
    market_price: float
    multiplier: float = 1.0

    @property
    def market_value(self) -> float:
        """Signed market value."""
        return self.quantity * self.market_price * self.multiplier


@dataclass(frozen=True)
class PortfolioState:
    """Snapshot of a portfolio at one timestamp."""

    timestamp: np.datetime64
    cash: float
    positions: tuple[Position, ...]
    margin_used: float = 0.0

    @property
    def equity(self) -> float:
        """Cash plus market value of all positions."""
        return self.cash + sum(p.market_value for p in self.positions)

    def weights(self) -> dict[str, float]:
        """Position weights relative to equity."""
        eq = self.equity
        if eq == 0:
            return {p.instrument: 0.0 for p in self.positions}
        return {p.instrument: p.market_value / eq for p in self.positions}

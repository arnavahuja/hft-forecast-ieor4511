"""WRDS TAQ: NBBO quote bars, trade bars and (opt-in) filtered raw quotes.

Aggregation happens server-side: one SQL statement per session returns bars, never ticks.
A liquid S&P 500 name produces millions of NBBO updates per session, so only the
``taq_nbbo_raw`` dataset -- capped and uncacheable -- brings quotes to the client.

Column names:
    The ``taqm_*`` NBBO schema varies by year, so the SQL builder takes its column names from
    :data:`NBBO_COLUMNS`. Entries are added only after they have been verified on WRDS with
    ``wrds.Connection().describe_table("taqm_<year>", "nbbom_<yyyymmdd>")``; an unverified year
    raises rather than guessing.

Conventions:
    Bars are right-closed and labelled at the bar end (§3.4): the row at ``t`` summarizes
    ``(t - delta, t]``, and its quote fields are the last valid NBBO at or before ``t``.
    ``time_m`` is the SIP timestamp in New York local time; ``part_time`` is never used.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from typing import Any, ClassVar, Final

import numpy as np
import pandas as pd
import polars as pl
from pydantic import Field

from hft_forecast.core import columns as C
from hft_forecast.core.calendar import TradingCalendar
from hft_forecast.core.errors import DataError
from hft_forecast.core.params import SourceParams
from hft_forecast.core.registry import register
from hft_forecast.core.types import (
    BAR_LABEL_KEY,
    AssetClass,
    DataRequest,
    DatasetInfo,
    Frequency,
    Instrument,
    MarketData,
)
from hft_forecast.data.normalize import normalize_frame
from hft_forecast.data.sources.wrds_connection import WrdsConnection
from hft_forecast.data.validation import QUOTE_FILTER_COUNTS_KEY

TAQ_TZ: Final = "America/New_York"
BAR_FREQUENCIES: Final = (
    Frequency.S1,
    Frequency.S5,
    Frequency.S10,
    Frequency.S30,
    Frequency.MIN1,
    Frequency.MIN5,
    Frequency.MIN15,
    Frequency.H1,
)
RAW_DATASET: Final = "taq_nbbo_raw"


@dataclass(frozen=True)
class NbboColumns:
    """Physical NBBO column names for one year of ``taqm_<year>.nbbom_*``."""

    bid: str
    bid_size: str
    ask: str
    ask_size: str
    condition: str


_TAQM_2024: Final = NbboColumns("best_bid", "best_bidsiz", "best_ask", "best_asksiz", "qu_cond")

NBBO_COLUMNS: Final[dict[int, NbboColumns]] = {
    # verified 2026-10-19 on taqm_2024.nbbom_20240702; qu_cond vocabulary there was
    # R 99.81%, Y 0.19%, then L, O, C, N and a handful of nulls
    2024: _TAQM_2024,
}


class WrdsTaqParams(SourceParams):
    """Quote filter thresholds (Holden-Jacobsen style) and pull limits."""

    quote_conditions: list[str] = Field(
        default_factory=lambda: ["R"],
        description="Accepted qu_cond codes; verify the vocabulary for the year on WRDS",
    )
    max_spread_frac: float = Field(
        0.05, gt=0, le=1, description="Drop quotes whose spread exceeds this fraction of the mid"
    )
    max_spread_dollars: float = Field(
        5.0, gt=0, description="Drop quotes whose spread exceeds this many dollars"
    )
    mid_twap: bool = Field(
        True, description="Compute the duration-weighted mid (slower); false uses no mid_twap"
    )
    count_filtered: bool = Field(
        True, description="Run a second aggregate per session counting quotes dropped per filter"
    )
    max_raw_symbols: int = Field(5, ge=1, description="Symbol cap for taq_nbbo_raw")
    max_raw_sessions: int = Field(20, ge=1, description="Session cap for taq_nbbo_raw")


def nbbo_columns(year: int) -> NbboColumns:
    """Verified NBBO column names for a year.

    Raises:
        DataError: If the year's schema has not been verified on WRDS yet.
    """
    cols = NBBO_COLUMNS.get(year)
    if cols is None:
        raise DataError(
            f"NBBO column names for {year} are not verified",
            details={
                "verified_years": sorted(NBBO_COLUMNS),
                "how": f'wrds.Connection().describe_table("taqm_{year}", "nbbom_{year}0702"), '
                "then add the entry to NBBO_COLUMNS",
            },
        )
    return cols


def _session_window(day: date, calendar: TradingCalendar) -> tuple[time, time]:
    """Regular session bounds as New York local times (early closes included)."""
    open_utc, close_utc = calendar.session_bounds(day)
    bounds = pd.to_datetime(np.array([open_utc, close_utc]), utc=True).tz_convert(TAQ_TZ)
    return bounds[0].time(), bounds[1].time()


def _quote_filter_sql(c: NbboColumns) -> str:
    mid = f"(({c.ask} + {c.bid}) / 2)"
    return (
        f"{c.bid} > 0 and {c.ask} > 0 and {c.bid_size} > 0 and {c.ask_size} > 0 "
        f"and {c.ask} > {c.bid} "
        f"and {c.ask} - {c.bid} <= %(max_abs)s "
        f"and ({c.ask} - {c.bid}) / {mid} <= %(max_frac)s "
        f"and {c.condition} in %(conds)s"
    )


def nbbo_sql(day: date, seconds: int, c: NbboColumns, *, mid_twap: bool) -> str:
    """Bar aggregation over one session of the NBBO table (filters applied before grouping)."""
    table = f"taqm_{day.year}.nbbom_{day:%Y%m%d}"
    src = (
        "select sym_root, extract(epoch from time_m)::float8 as t, "
        f"{c.bid} as bid, {c.ask} as ask, {c.bid_size} as bidsiz, {c.ask_size} as asksiz "
        f"from {table} where sym_root in %(syms)s and sym_suffix is null "
        "and time_m >= %(open)s and time_m < %(close)s "
        f"and {_quote_filter_sql(c)}"
    )
    bucketed = (
        f"select sym_root, floor(t / {seconds}) as bucket, t, bid, ask, bidsiz, asksiz, "
        "(bid + ask) / 2 as mid from q"
    )
    if mid_twap:
        # the last quote of a bucket prevails until the bucket ends
        durations = (
            f"select b.*, coalesce(lead(t) over w - t, (bucket + 1) * {seconds} - t) as dur "
            "from b window w as (partition by sym_root, bucket order by t)"
        )
        twap = "sum(mid * dur) / nullif(sum(dur), 0) as mid_twap"
        tail = f", d as ({durations}) select "
        source = "d"
    else:
        twap = "null::float8 as mid_twap"
        tail = " select "
        source = "b"
    return (
        f"with q as ({src}), b as ({bucketed}){tail}"
        "sym_root, bucket, "
        "(array_agg(bid order by t desc))[1] as bid, "
        "(array_agg(ask order by t desc))[1] as ask, "
        "(array_agg(bidsiz order by t desc))[1] as bidsiz, "
        "(array_agg(asksiz order by t desc))[1] as asksiz, "
        "(array_agg(mid order by t))[1] as open, "
        f"max(mid) as high, min(mid) as low, count(*) as n_quotes, {twap} "
        f"from {source} group by sym_root, bucket order by sym_root, bucket"
    )


def filter_counts_sql(day: date, c: NbboColumns) -> str:
    """Quotes rejected by each filter in one session, for the quality report."""
    table = f"taqm_{day.year}.nbbom_{day:%Y%m%d}"
    mid = f"(({c.ask} + {c.bid}) / 2)"
    return (
        "select "
        f"count(*) filter (where {c.bid} <= 0 or {c.ask} <= 0 or {c.bid_size} <= 0 "
        f"or {c.ask_size} <= 0) as non_positive, "
        f"count(*) filter (where {c.ask} <= {c.bid}) as locked_or_crossed, "
        f"count(*) filter (where {c.condition} is null or {c.condition} not in %(conds)s) "
        "as quote_condition, "
        f"count(*) filter (where {c.ask} > {c.bid} and ({c.ask} - {c.bid} > %(max_abs)s "
        f"or ({c.ask} - {c.bid}) / {mid} > %(max_frac)s)) as wide_spread "
        f"from {table} where sym_root in %(syms)s and sym_suffix is null "
        "and time_m >= %(open)s and time_m < %(close)s"
    )


def raw_sql(day: date, c: NbboColumns) -> str:
    """Filtered quotes of one session, no aggregation."""
    table = f"taqm_{day.year}.nbbom_{day:%Y%m%d}"
    return (
        f"select sym_root, time_m, {c.bid} as bid, {c.ask} as ask, "
        f"{c.bid_size} as bidsiz, {c.ask_size} as asksiz "
        f"from {table} where sym_root in %(syms)s and sym_suffix is null "
        "and time_m >= %(open)s and time_m < %(close)s "
        f"and {_quote_filter_sql(c)} order by sym_root, time_m"
    )


def trade_sql(day: date, seconds: int) -> str:
    """Trade bar aggregation over one session of the consolidated trade table."""
    table = f"taqm_{day.year}.ctm_{day:%Y%m%d}"
    return (
        f"select sym_root, floor(extract(epoch from time_m) / {seconds}) as bucket, "
        "(array_agg(price order by time_m))[1] as open, max(price) as high, "
        "min(price) as low, (array_agg(price order by time_m desc))[1] as close, "
        "sum(size) as volume, count(*) as n_trades, "
        "sum(price * size) / nullif(sum(size), 0) as vwap "
        f"from {table} where sym_root in %(syms)s and sym_suffix is null and tr_corr = '00' "
        "and time_m >= %(open)s and time_m < %(close)s "
        "group by sym_root, bucket order by sym_root, bucket"
    )


def _bar_end(buckets: pd.Series[float], day: date, seconds: int) -> pd.Series[pd.Timestamp]:
    """Bucket index to the bar-end timestamp in UTC (bucket b covers ``[b, b+1) * seconds``)."""
    local = pd.Timestamp(day) + pd.to_timedelta((buckets.astype(float) + 1) * seconds, unit="s")
    return local.dt.tz_localize(TAQ_TZ).dt.tz_convert("UTC")


def _num(frame: pd.DataFrame, name: str) -> pd.Series[float]:
    return pd.to_numeric(frame[name], errors="coerce").astype(float)


def nbbo_to_canonical(frame: pd.DataFrame, day: date, seconds: int) -> pl.DataFrame:
    """NBBO bar rows to the canonical quote schema (``close`` is the mid)."""
    bid, ask = _num(frame, "bid"), _num(frame, "ask")
    bid_size, ask_size = _num(frame, "bidsiz"), _num(frame, "asksiz")
    mid = (bid + ask) / 2.0
    spread = ask - bid
    out = pd.DataFrame(
        {
            C.TIMESTAMP: _bar_end(frame["bucket"], day, seconds),
            C.INSTRUMENT: frame["sym_root"].astype(str),
            C.OPEN: _num(frame, "open"),
            C.HIGH: _num(frame, "high"),
            C.LOW: _num(frame, "low"),
            C.CLOSE: mid,
            C.BID: bid,
            C.ASK: ask,
            C.BID_SIZE: bid_size,
            C.ASK_SIZE: ask_size,
            C.MID: mid,
            C.MICROPRICE: (bid * ask_size + ask * bid_size) / (bid_size + ask_size),
            C.SPREAD: spread,
            C.SPREAD_BPS: spread / mid * 1e4,
            C.MID_TWAP: _num(frame, "mid_twap"),
            C.N_QUOTES: _num(frame, "n_quotes"),
            C.IS_STALE: 0.0,
        }
    )
    return pl.from_pandas(out)


def trades_to_canonical(frame: pd.DataFrame, day: date, seconds: int) -> pl.DataFrame:
    """Trade bar rows to the canonical schema, labelled at the bar end."""
    out = pd.DataFrame(
        {
            C.TIMESTAMP: _bar_end(frame["bucket"], day, seconds),
            C.INSTRUMENT: frame["sym_root"].astype(str),
            C.OPEN: _num(frame, "open"),
            C.HIGH: _num(frame, "high"),
            C.LOW: _num(frame, "low"),
            C.CLOSE: _num(frame, "close"),
            C.VOLUME: _num(frame, "volume"),
            C.N_TRADES: _num(frame, "n_trades"),
            C.VWAP: _num(frame, "vwap"),
        }
    )
    return pl.from_pandas(out)


_FILLED: Final = (C.BID, C.ASK, C.BID_SIZE, C.ASK_SIZE, C.MID, C.MICROPRICE, C.SPREAD)


def fill_session(
    bars: pl.DataFrame, day: date, seconds: int, calendar: TradingCalendar
) -> pl.DataFrame:
    """Forward-fill missing buckets inside one session and flag them as stale.

    Buckets before an instrument's first quote of the session stay null: there is nothing to
    carry forward. ``n_quotes`` is zero on a filled bar and the bar's OHLC collapse to the mid.
    """
    if bars.is_empty():
        return bars
    open_utc, close_utc = calendar.session_bounds(day)
    step = np.timedelta64(seconds, "s")
    grid = np.arange(open_utc + step, close_utc + step, step, dtype="datetime64[us]")
    instruments = bars.get_column(C.INSTRUMENT).unique().sort().to_list()
    full = pl.DataFrame(
        {
            C.TIMESTAMP: np.tile(grid, len(instruments)),
            C.INSTRUMENT: np.repeat(instruments, len(grid)),
        }
    ).with_columns(pl.col(C.TIMESTAMP).cast(bars.schema[C.TIMESTAMP]))
    joined = full.join(bars, on=list(C.KEY_COLUMNS), how="left").sort([C.INSTRUMENT, C.TIMESTAMP])
    stale = pl.col(C.MID).is_null()
    filled = joined.with_columns(
        [pl.col(c).forward_fill().over(C.INSTRUMENT) for c in _FILLED]
        + [
            stale.cast(pl.Float64).alias(C.IS_STALE),
            pl.when(stale).then(0.0).otherwise(pl.col(C.N_QUOTES)).alias(C.N_QUOTES),
        ]
    )
    collapsed = filled.with_columns(
        [
            pl.when(pl.col(C.IS_STALE) == 1.0).then(pl.col(C.MID)).otherwise(pl.col(c)).alias(c)
            for c in (C.OPEN, C.HIGH, C.LOW, C.CLOSE, C.MID_TWAP)
        ]
        + [
            pl.when(pl.col(C.IS_STALE) == 1.0)
            .then(pl.col(C.SPREAD) / pl.col(C.MID) * 1e4)
            .otherwise(pl.col(C.SPREAD_BPS))
            .alias(C.SPREAD_BPS)
        ]
    )
    return collapsed.filter(pl.col(C.MID).is_not_null())


@register(
    "data_source",
    name="wrds_taq",
    version="1.0.0",
    tags=["taq", "nbbo", "intraday", "academic"],
    capabilities={"asset:equity", "freq:intraday", "freq:subminute"},
)
class WrdsTaqSource(WrdsConnection):
    """TAQ NBBO and trade bars, keyed by ticker (``sym_root``) rather than PERMNO."""

    Params = WrdsTaqParams
    params: WrdsTaqParams
    source_version: ClassVar[str] = "wrds-taq-1"

    def search_instruments(self, query: str) -> list[Instrument]:
        """TAQ is keyed by ticker, so a query is its own answer."""
        symbol = query.strip().upper()
        return [Instrument(id=symbol, symbol=symbol)] if symbol else []

    def describe(self) -> list[DatasetInfo]:
        """Datasets offered."""
        return [
            DatasetInfo(
                source="wrds_taq",
                dataset="taq_nbbo_bars",
                description="NBBO quote bars (last quote in the bar, mid as close), "
                "filtered and aggregated on WRDS",
                frequencies=BAR_FREQUENCIES,
                asset_classes=(AssetClass.EQUITY,),
                fields=(C.OPEN, C.HIGH, C.LOW, C.CLOSE, *C.QUOTE_FIELDS),
                quality="research",
                notes="Right-closed, labelled at the bar end. Missing buckets are "
                "forward-filled within the session and flagged by is_stale.",
            ),
            DatasetInfo(
                source="wrds_taq",
                dataset="taq_trade_bars",
                description="Trade bars aggregated from TAQ trades (tr_corr = 00)",
                frequencies=BAR_FREQUENCIES,
                asset_classes=(AssetClass.EQUITY,),
                fields=(*C.OHLCV, C.VWAP, C.N_TRADES),
                quality="research",
                notes="Right-closed, labelled at the bar end.",
            ),
        ]

    # ---- fetching

    def fetch(self, request: DataRequest) -> MarketData:
        """Dispatch on the dataset."""
        if request.dataset == "taq_nbbo_bars":
            return self._nbbo_bars(request)
        if request.dataset == "taq_trade_bars":
            return self._trade_bars(request)
        raise DataError(f"Unknown TAQ dataset '{request.dataset}'")

    def _plan(self, request: DataRequest) -> tuple[tuple[str, ...], int, list[date]]:
        seconds = request.frequency.seconds
        if seconds is None:
            raise DataError("TAQ bars need an intraday frequency")
        syms = tuple(i.upper() for i in request.instruments)
        if not syms:
            raise DataError("TAQ bars need ticker symbols")
        return syms, seconds, TradingCalendar().sessions(request.start, request.end)

    def _session_params(self, day: date, syms: tuple[str, ...]) -> dict[str, Any]:
        start, end = _session_window(day, TradingCalendar())
        p = self.params
        return {
            "syms": syms,
            "open": start,
            "close": end,
            "conds": tuple(p.quote_conditions),
            "max_abs": p.max_spread_dollars,
            "max_frac": p.max_spread_frac,
        }

    def _nbbo_bars(self, request: DataRequest) -> MarketData:
        syms, seconds, sessions = self._plan(request)
        calendar = TradingCalendar()
        p = self.params
        frames: list[pl.DataFrame] = []
        dropped: dict[str, int] = {}
        for day in sessions:
            cols = nbbo_columns(day.year)
            args = self._session_params(day, syms)
            rows = self._query(nbbo_sql(day, seconds, cols, mid_twap=p.mid_twap), params=args)
            if len(rows):
                frames.append(
                    fill_session(nbbo_to_canonical(rows, day, seconds), day, seconds, calendar)
                )
            if p.count_filtered:
                counts = self._query(filter_counts_sql(day, cols), params=args)
                for key, value in counts.iloc[0].items():
                    dropped[str(key)] = dropped.get(str(key), 0) + int(value or 0)
        if not frames:
            raise DataError("TAQ returned no quotes for the request")
        return MarketData(
            normalize_frame(pl.concat(frames)),
            request.frequency,
            {s: Instrument(id=s, symbol=s) for s in syms},
            metadata={
                "source": "wrds_taq",
                "dataset": "taq_nbbo_bars",
                "quality": "research",
                C.PRICE_BASIS_KEY: "mid",
                BAR_LABEL_KEY: "end",
                "mid_twap": "duration_weighted" if p.mid_twap else "none",
                QUOTE_FILTER_COUNTS_KEY: dropped,
            },
        )

    def _trade_bars(self, request: DataRequest) -> MarketData:
        syms, seconds, sessions = self._plan(request)
        frames = []
        for day in sessions:
            start, end = _session_window(day, TradingCalendar())
            rows = self._query(
                trade_sql(day, seconds), params={"syms": syms, "open": start, "close": end}
            )
            if len(rows):
                frames.append(trades_to_canonical(rows, day, seconds))
        if not frames:
            raise DataError("TAQ returned no trades for the request")
        return MarketData(
            normalize_frame(pl.concat(frames)),
            request.frequency,
            {s: Instrument(id=s, symbol=s) for s in syms},
            metadata={
                "source": "wrds_taq",
                "dataset": "taq_trade_bars",
                "quality": "research",
                C.PRICE_BASIS_KEY: "trade",
                BAR_LABEL_KEY: "end",
            },
        )


@register(
    "data_source",
    name="wrds_taq_raw",
    version="1.0.0",
    tags=["taq", "nbbo", "intraday", "academic"],
    capabilities={"asset:equity", "freq:intraday", "freq:subminute"},
)
class WrdsTaqRawSource(WrdsTaqSource):
    """Filtered NBBO quotes with no aggregation: a separate, uncacheable source.

    Millions of rows per symbol-session must not enter the Parquet cache, and ``cacheable`` is
    a property of the source rather than of the dataset, hence the second plugin.
    """

    source_version: ClassVar[str] = "wrds-taq-raw-1"
    cacheable: ClassVar[bool] = False

    def describe(self) -> list[DatasetInfo]:
        """The single raw dataset."""
        p = self.params
        return [
            DatasetInfo(
                source="wrds_taq_raw",
                dataset=RAW_DATASET,
                description="Filtered NBBO quotes, one row per update (not cached)",
                frequencies=(Frequency.S1,),
                asset_classes=(AssetClass.EQUITY,),
                fields=(C.BID, C.ASK, C.BID_SIZE, C.ASK_SIZE, C.MID),
                quality="research",
                notes=f"Capped at {p.max_raw_symbols} symbols x {p.max_raw_sessions} sessions; "
                "event-time experiments only.",
            )
        ]

    def fetch(self, request: DataRequest) -> MarketData:
        """Pull the raw quotes of a handful of symbol-sessions."""
        if request.dataset != RAW_DATASET:
            raise DataError(f"Unknown TAQ dataset '{request.dataset}'")
        syms = tuple(i.upper() for i in request.instruments)
        sessions = TradingCalendar().sessions(request.start, request.end)
        p = self.params
        if not syms:
            raise DataError("Raw NBBO needs ticker symbols")
        if len(syms) > p.max_raw_symbols or len(sessions) > p.max_raw_sessions:
            raise DataError(
                f"Raw NBBO is capped at {p.max_raw_symbols} symbols x "
                f"{p.max_raw_sessions} sessions",
                details={"symbols": len(syms), "sessions": len(sessions)},
            )
        frames = []
        for day in sessions:
            rows = self._query(
                raw_sql(day, nbbo_columns(day.year)), params=self._session_params(day, syms)
            )
            if len(rows):
                frames.append(_raw_to_canonical(rows, day))
        if not frames:
            raise DataError("TAQ returned no quotes for the request")
        return MarketData(
            normalize_frame(pl.concat(frames)),
            request.frequency,
            {s: Instrument(id=s, symbol=s) for s in syms},
            metadata={
                "source": "wrds_taq_raw",
                "dataset": RAW_DATASET,
                "quality": "research",
                C.PRICE_BASIS_KEY: "mid",
            },
        )


def _raw_to_canonical(frame: pd.DataFrame, day: date) -> pl.DataFrame:
    """Raw quote rows (one per update) to the canonical schema."""
    local = pd.Timestamp(day) + pd.to_timedelta(frame["time_m"].astype(str))
    bid, ask = _num(frame, "bid"), _num(frame, "ask")
    out = pd.DataFrame(
        {
            C.TIMESTAMP: local.dt.tz_localize(TAQ_TZ).dt.tz_convert("UTC"),
            C.INSTRUMENT: frame["sym_root"].astype(str),
            C.BID: bid,
            C.ASK: ask,
            C.BID_SIZE: _num(frame, "bidsiz"),
            C.ASK_SIZE: _num(frame, "asksiz"),
            C.MID: (bid + ask) / 2.0,
            C.CLOSE: (bid + ask) / 2.0,
        }
    )
    return pl.from_pandas(out)

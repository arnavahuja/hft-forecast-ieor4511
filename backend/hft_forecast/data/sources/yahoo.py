"""Yahoo Finance data source (via ``yfinance``).

Yahoo data is survivorship-biased (only currently listed tickers) and lower quality; the
dataset metadata says so. Yahoo's ``Close`` is split-adjusted; ``Adj Close`` is also
dividend-adjusted. We store split-adjusted OHLCV plus ``adj_close``, ``dividend`` and
``split_ratio`` and convert to the requested adjustment.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, ClassVar, Final

import pandas as pd
import polars as pl

from hft_forecast.core import columns as C
from hft_forecast.core.errors import DataError, DataSourceUnavailableError
from hft_forecast.core.interfaces import DataSource
from hft_forecast.core.registry import register
from hft_forecast.core.types import (
    Adjustment,
    AssetClass,
    DataRequest,
    DatasetInfo,
    Frequency,
    Instrument,
    MarketData,
)
from hft_forecast.data.adjustments import convert
from hft_forecast.data.normalize import normalize_frame

_INTERVALS: Final = {
    Frequency.MIN1: "1m",
    Frequency.MIN5: "5m",
    Frequency.MIN15: "15m",
    Frequency.H1: "1h",
    Frequency.D1: "1d",
    Frequency.W1: "1wk",
    Frequency.MO1: "1mo",
}
_MAX_INTRADAY_DAYS: Final = {
    Frequency.MIN1: 30,
    Frequency.MIN5: 60,
    Frequency.MIN15: 60,
    Frequency.H1: 730,
}
_RENAME: Final = {
    "Date": C.TIMESTAMP,
    "Datetime": C.TIMESTAMP,
    "Open": C.OPEN,
    "High": C.HIGH,
    "Low": C.LOW,
    "Close": C.CLOSE,
    "Adj Close": C.ADJ_CLOSE,
    "Volume": C.VOLUME,
    "Dividends": C.DIVIDEND,
    "Stock Splits": C.SPLIT,
}
_SEARCH_LIMIT: Final = 10


def yahoo_to_canonical(frame: pd.DataFrame, ticker: str) -> pl.DataFrame:
    """Map one ticker's yfinance frame (index = date) to the canonical long schema."""
    pdf = frame.reset_index()
    pdf.columns = pd.Index([str(c) for c in pdf.columns])
    pdf = pdf.rename(columns=_RENAME)
    keep = [c for c in pdf.columns if c in _RENAME.values()]
    pdf = pdf[keep].dropna(subset=[C.CLOSE])
    ts = pd.to_datetime(pdf[C.TIMESTAMP])
    pdf[C.TIMESTAMP] = ts.dt.tz_convert("UTC") if ts.dt.tz is not None else ts.dt.tz_localize("UTC")
    out = pl.from_pandas(pdf).with_columns(pl.lit(ticker).alias(C.INSTRUMENT))
    return out


@register(
    "data_source",
    name="yahoo",
    version="1.0.0",
    tags=["equity", "etf", "free"],
    capabilities={"asset:equity", "freq:daily", "freq:intraday"},
)
class YahooSource(DataSource):
    """Daily and limited intraday OHLCV, dividends and splits from Yahoo Finance.

    Survivorship-biased and lower quality: use for prototyping, not final research.
    """

    source_version: ClassVar[str] = "yfinance-1"

    def describe(self) -> list[DatasetInfo]:
        """Datasets offered."""
        note = "Survivorship-biased: only currently listed tickers. Adjustments may be restated."
        return [
            DatasetInfo(
                source="yahoo",
                dataset="daily",
                description="Daily/weekly/monthly OHLCV with dividends and splits",
                frequencies=(Frequency.D1, Frequency.W1, Frequency.MO1),
                asset_classes=(AssetClass.EQUITY, AssetClass.ETF, AssetClass.INDEX),
                fields=(*C.OHLCV, C.ADJ_CLOSE, C.DIVIDEND, C.SPLIT),
                quality="lower",
                notes=note,
            ),
            DatasetInfo(
                source="yahoo",
                dataset="intraday",
                description="Intraday OHLCV (1m: last 30 days, 5m/15m: 60 days, 1h: 730 days)",
                frequencies=(Frequency.MIN1, Frequency.MIN5, Frequency.MIN15, Frequency.H1),
                asset_classes=(AssetClass.EQUITY, AssetClass.ETF),
                quality="lower",
                notes=note,
            ),
        ]

    def search_instruments(self, query: str) -> list[Instrument]:
        """Search Yahoo for tickers (falls back to echoing the query offline)."""
        try:
            import yfinance as yf

            quotes: list[dict[str, Any]] = yf.Search(query, max_results=_SEARCH_LIMIT).quotes
        except Exception:
            quotes = []
        out = []
        for q in quotes:
            symbol = q.get("symbol")
            if not symbol:
                continue
            kind = str(q.get("quoteType", "")).lower()
            asset = (
                AssetClass.ETF
                if kind == "etf"
                else (AssetClass.INDEX if kind == "index" else AssetClass.EQUITY)
            )
            out.append(
                Instrument(
                    id=symbol,
                    symbol=symbol,
                    asset_class=asset,
                    exchange=q.get("exchange"),
                    name=q.get("shortname") or q.get("longname"),
                )
            )
        if not out and query.strip():
            sym = query.strip().upper()
            out.append(Instrument(id=sym, symbol=sym))
        return out

    def _download(self, tickers: list[str], start: date, end: date, interval: str) -> Any:
        """Call yfinance (isolated so tests can substitute recorded responses)."""
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - dependency is required
            raise DataSourceUnavailableError("yfinance is not installed") from exc
        return yf.download(
            tickers,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval=interval,
            auto_adjust=False,
            actions=True,
            group_by="ticker",
            progress=False,
            threads=True,
            multi_level_index=True,
        )

    def fetch(self, request: DataRequest) -> MarketData:
        """Download data for the requested tickers."""
        if not request.instruments:
            raise DataError("Yahoo requests need explicit tickers")
        interval = _INTERVALS[request.frequency]
        limit = _MAX_INTRADAY_DAYS.get(request.frequency)
        if limit is not None and (date.today() - request.start).days > limit:
            raise DataError(
                f"Yahoo only serves {request.frequency} bars for the last {limit} days",
                details={"frequency": request.frequency.value, "max_days": limit},
            )
        raw = self._download(list(request.instruments), request.start, request.end, interval)
        frames = []
        if raw is not None and len(raw):
            top = raw.columns.get_level_values(0) if raw.columns.nlevels > 1 else None
            for ticker in request.instruments:
                if top is not None and ticker in set(top):
                    sub = raw[ticker]
                elif top is None and len(request.instruments) == 1:
                    sub = raw
                else:
                    continue
                canon = yahoo_to_canonical(sub, ticker)
                if canon.height:
                    frames.append(canon)
        if not frames:
            raise DataError(
                "Yahoo returned no data", details={"tickers": list(request.instruments)}
            )
        frame = normalize_frame(pl.concat(frames, how="diagonal_relaxed"))
        if C.SPLIT in frame.columns:
            frame = frame.with_columns(
                pl.when(pl.col(C.SPLIT) > 0).then(pl.col(C.SPLIT)).otherwise(1.0).alias(C.SPLIT)
            )
        frame = convert(frame, Adjustment.SPLIT, request.adjustment)
        instruments = {t: Instrument(id=t, symbol=t) for t in request.instruments}
        return MarketData(
            frame,
            request.frequency,
            instruments,
            metadata={
                "source": "yahoo",
                "adjustment": request.adjustment.value,
                "survivorship_bias_free": False,
                "quality": "lower",
            },
        )

"""Deterministic synthetic data (geometric Brownian motion with volume).

Useful for demos without network access, tests and golden fixtures. Each instrument's path is
seeded from the request seed and the instrument id, so results are reproducible.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, time
from typing import ClassVar, Final

import numpy as np
import polars as pl
from pydantic import Field

from hft_forecast.core import columns as C
from hft_forecast.core.calendar import TradingCalendar
from hft_forecast.core.errors import DataError
from hft_forecast.core.interfaces import DataSource
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
    TimeArray,
)
from hft_forecast.data.normalize import normalize_frame

_DEFAULT_SEED: Final = 7
_BASE_PRICE: Final = 100.0
_BASE_VOLUME: Final = 1_000_000.0
_INTRADAY_RANGE: Final = 0.01
SUBMINUTE: Final = (Frequency.S1, Frequency.S5, Frequency.S10, Frequency.S30)
QUOTE_FREQUENCIES: Final = (*SUBMINUTE, Frequency.MIN1, Frequency.MIN5)
_BASE_SPREAD_BPS: Final = 2.0
_BASE_QUOTE_SIZE: Final = 500.0
_QUOTES_PER_SECOND: Final = 2.0
_HALT_SECONDS: Final = 600
"""Length of the single trading halt injected into every instrument's quote series."""
FACTORS_ID: Final = "FF"
FACTOR_NAMES: Final = ("mkt_rf", "smb", "hml", "rmw", "cma", "mom", "rf")
SMALL_FACTOR_VOL: Final = 0.08


def _instrument_seed(seed: int, instrument: str) -> int:
    digest = hashlib.sha256(f"{seed}:{instrument}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


class SyntheticParams(SourceParams):
    """Parameters of the synthetic source."""

    annual_drift: float = Field(0.07, description="Annual drift of log prices")
    annual_vol: float = Field(0.2, gt=0, description="Annual volatility")
    market_beta: float = Field(0.8, ge=0, le=1.5, description="Loading on a common factor")
    stale_rate: float = Field(
        0.02, ge=0, lt=1, description="Share of quote bars with no NBBO update (mid carried)"
    )


@register(
    "data_source",
    name="synthetic",
    version="1.0.0",
    tags=["synthetic", "offline", "demo"],
    capabilities={"asset:equity", "freq:daily", "freq:intraday", "freq:subminute"},
)
class SyntheticSource(DataSource):
    """Deterministic synthetic OHLCV for demos and tests (no network needed)."""

    Params = SyntheticParams
    params: SyntheticParams
    source_version: ClassVar[str] = "1"

    def describe(self) -> list[DatasetInfo]:
        """Datasets offered."""
        return [
            DatasetInfo(
                source="synthetic",
                dataset="gbm",
                description="Correlated geometric Brownian motion OHLCV on XNYS sessions",
                frequencies=(Frequency.D1,),
                asset_classes=(AssetClass.EQUITY,),
                survivorship_bias_free=True,
                quality="synthetic",
                notes="Any instrument id is accepted. Seed via request option 'seed'.",
            ),
            DatasetInfo(
                source="synthetic",
                dataset="ff_factors",
                description="Synthetic daily factor returns (mkt_rf, smb, hml, rmw, cma, mom, rf)",
                frequencies=(Frequency.D1,),
                asset_classes=(AssetClass.OTHER,),
                fields=FACTOR_NAMES,
                quality="synthetic",
                notes="Market factor matches the common factor of the 'gbm' dataset.",
            ),
            DatasetInfo(
                source="synthetic",
                dataset="quotes",
                description="Sub-minute NBBO quote bars (mid random walk, bar-end labelled)",
                frequencies=QUOTE_FREQUENCIES,
                asset_classes=(AssetClass.EQUITY,),
                fields=(*C.OHLCV, *C.QUOTE_FIELDS),
                quality="synthetic",
                notes="Injects stale bars at 'stale_rate' and one halt per instrument.",
            ),
        ]

    def search_instruments(self, query: str) -> list[Instrument]:
        """Any query is a valid synthetic instrument."""
        sym = query.strip().upper() or "SYN"
        return [Instrument(id=sym, symbol=sym)]

    def fetch(self, request: DataRequest) -> MarketData:
        """Generate deterministic daily bars (or factor returns)."""
        if request.dataset == "ff_factors":
            return self._factors(request)
        if request.dataset == "quotes":
            return self._quotes(request)
        if request.frequency is not Frequency.D1:
            raise DataError("The synthetic source only generates daily bars")
        if not request.instruments:
            raise DataError("Synthetic requests need instrument ids")
        seed = int(request.options.get("seed", _DEFAULT_SEED))
        sessions = TradingCalendar().sessions(request.start, request.end)
        n = len(sessions)
        if n == 0:
            raise DataError("No sessions in the requested range")
        ppy = TradingCalendar().sessions_per_year()
        p = self.params
        dt = 1.0 / ppy
        market = np.random.default_rng(seed).standard_normal(n)
        stamps = [datetime.combine(s, time(0), tzinfo=UTC) for s in sessions]
        frames = []
        for inst in request.instruments:
            rng = np.random.default_rng(_instrument_seed(seed, inst))
            idio = rng.standard_normal(n)
            beta = p.market_beta
            shock = beta * market + np.sqrt(max(1.0 - beta**2, 0.0)) * idio
            log_ret = (p.annual_drift - 0.5 * p.annual_vol**2) * dt + p.annual_vol * np.sqrt(
                dt
            ) * shock
            close = _BASE_PRICE * rng.uniform(0.5, 2.0) * np.exp(np.cumsum(log_ret))
            prev = np.concatenate(([close[0]], close[:-1]))
            gap = rng.normal(0.0, p.annual_vol * np.sqrt(dt) * 0.3, n)
            open_ = prev * np.exp(gap)
            spread = np.abs(rng.normal(0.0, _INTRADAY_RANGE, n))
            high = np.maximum(open_, close) * (1.0 + spread)
            low = np.minimum(open_, close) * (1.0 - spread)
            volume = _BASE_VOLUME * rng.lognormal(0.0, 0.3, n)
            frames.append(
                pl.DataFrame(
                    {
                        C.TIMESTAMP: stamps,
                        C.INSTRUMENT: [inst] * n,
                        C.OPEN: open_,
                        C.HIGH: high,
                        C.LOW: low,
                        C.CLOSE: close,
                        C.VOLUME: volume,
                    }
                )
            )
        frame = normalize_frame(pl.concat(frames))
        return MarketData(
            frame,
            Frequency.D1,
            {i: Instrument(id=i, symbol=i) for i in request.instruments},
            metadata={
                "source": "synthetic",
                "adjustment": request.adjustment.value,
                "survivorship_bias_free": True,
                "quality": "synthetic",
            },
        )

    def _factors(self, request: DataRequest) -> MarketData:
        seed = int(request.options.get("seed", _DEFAULT_SEED))
        sessions = TradingCalendar().sessions(request.start, request.end)
        n = len(sessions)
        ppy = TradingCalendar().sessions_per_year()
        dt = 1.0 / ppy
        p = self.params
        market = np.random.default_rng(seed).standard_normal(n)
        rng = np.random.default_rng(_instrument_seed(seed, "factors"))
        cols: dict[str, object] = {
            C.TIMESTAMP: [datetime.combine(s, time(0), tzinfo=UTC) for s in sessions],
            C.INSTRUMENT: [FACTORS_ID] * n,
            "mkt_rf": p.annual_drift * dt + p.annual_vol * np.sqrt(dt) * market,
            "rf": np.full(n, 0.0),
        }
        for name in FACTOR_NAMES:
            if name not in cols:
                cols[name] = SMALL_FACTOR_VOL * np.sqrt(dt) * rng.standard_normal(n)
        frame = normalize_frame(pl.DataFrame(cols))
        return MarketData(
            frame,
            Frequency.D1,
            {
                FACTORS_ID: Instrument(
                    id=FACTORS_ID, symbol=FACTORS_ID, asset_class=AssetClass.OTHER
                )
            },
            metadata={"source": "synthetic", "dataset": "ff_factors"},
        )

    def _quote_grid(self, request: DataRequest) -> TimeArray:
        """Bar-end timestamps of every regular-session bar in the range (right-closed)."""
        seconds = request.frequency.seconds
        if seconds is None or request.frequency not in QUOTE_FREQUENCIES:
            raise DataError(
                f"'quotes' needs a sub-minute or minute frequency, got {request.frequency}"
            )
        calendar = TradingCalendar()
        step = np.timedelta64(seconds, "s")
        blocks = []
        for session in calendar.sessions(request.start, request.end):
            open_utc, close_utc = calendar.session_bounds(session)
            # first bar ends one step after the open, last bar ends at the close
            blocks.append(
                np.arange(open_utc + step, close_utc + step, step, dtype="datetime64[us]")
            )
        if not blocks:
            raise DataError("No sessions in the requested range")
        return np.concatenate(blocks)

    def _quotes(self, request: DataRequest) -> MarketData:
        """Quote bars: a mid random walk with a log-normal spread and Poisson sizes."""
        if not request.instruments:
            raise DataError("Synthetic requests need instrument ids")
        grid = self._quote_grid(request)
        seconds = request.frequency.seconds
        assert seconds is not None  # _quote_grid rejects non-intraday frequencies
        n = len(grid)
        seed = int(request.options.get("seed", _DEFAULT_SEED))
        p = self.params
        ppy = TradingCalendar().periods_per_year(request.frequency)
        step_vol = p.annual_vol / np.sqrt(ppy)
        market = np.random.default_rng(seed).standard_normal(n)
        frames = []
        for inst in request.instruments:
            rng = np.random.default_rng(_instrument_seed(seed, inst))
            beta = p.market_beta
            shock = beta * market + np.sqrt(max(1.0 - beta**2, 0.0)) * rng.standard_normal(n)
            true_mid = _BASE_PRICE * rng.uniform(0.5, 2.0) * np.exp(np.cumsum(step_vol * shock))
            stale = rng.random(n) < p.stale_rate
            halt_bars = max(_HALT_SECONDS // seconds, 1)
            halt_at = int(rng.integers(0, max(n - halt_bars, 1)))
            stale[halt_at : halt_at + halt_bars] = True
            stale[0] = False  # the first bar must carry a quote for the forward fill
            # a stale bar carries the last observed mid forward
            last = np.maximum.accumulate(np.where(stale, 0, np.arange(n)))
            mid = true_mid[last]
            spread = mid * _BASE_SPREAD_BPS * 1e-4 * rng.lognormal(0.0, 0.3, n)
            bid_size = rng.poisson(_BASE_QUOTE_SIZE, n).astype(float) + 1.0
            ask_size = rng.poisson(_BASE_QUOTE_SIZE, n).astype(float) + 1.0
            bid, ask = mid - spread / 2.0, mid + spread / 2.0
            prev_mid = np.concatenate(([mid[0]], mid[:-1]))
            intrabar = np.abs(rng.normal(0.0, step_vol, n)) * mid
            quotes = np.where(stale, 0.0, rng.poisson(_QUOTES_PER_SECOND * seconds, n) + 1.0)
            frames.append(
                pl.DataFrame(
                    {
                        C.TIMESTAMP: grid,
                        C.INSTRUMENT: [inst] * n,
                        C.OPEN: prev_mid,
                        C.HIGH: np.maximum(prev_mid, mid) + np.where(stale, 0.0, intrabar),
                        C.LOW: np.minimum(prev_mid, mid) - np.where(stale, 0.0, intrabar),
                        C.CLOSE: mid,
                        C.BID: bid,
                        C.ASK: ask,
                        C.BID_SIZE: bid_size,
                        C.ASK_SIZE: ask_size,
                        C.MID: mid,
                        C.MICROPRICE: (bid * ask_size + ask * bid_size) / (bid_size + ask_size),
                        C.SPREAD: spread,
                        C.SPREAD_BPS: spread / mid * 1e4,
                        C.MID_TWAP: (prev_mid + mid) / 2.0,
                        C.N_QUOTES: quotes,
                        C.IS_STALE: stale.astype(float),
                    }
                )
            )
        return MarketData(
            normalize_frame(pl.concat(frames)),
            request.frequency,
            {i: Instrument(id=i, symbol=i) for i in request.instruments},
            metadata={
                "source": "synthetic",
                "dataset": "quotes",
                "quality": "synthetic",
                C.PRICE_BASIS_KEY: "mid",
                BAR_LABEL_KEY: "end",
            },
        )

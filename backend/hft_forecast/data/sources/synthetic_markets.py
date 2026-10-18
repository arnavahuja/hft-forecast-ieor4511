"""Synthetic intraday bars (offline demos and tests).

* ``gbm_intraday``: intraday OHLCV between 14:30 and 21:00 UTC on exchange sessions.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, time, timedelta
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
    AssetClass,
    DataRequest,
    DatasetInfo,
    Frequency,
    Instrument,
    MarketData,
)
from hft_forecast.data.normalize import normalize_frame

SESSION_OPEN: Final = time(14, 30)
SESSION_MINUTES: Final = 390


def _seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:4], "little")


class SyntheticMarketParams(SourceParams):
    """Parameters of the synthetic markets source."""

    annual_vol: float = Field(0.18, gt=0, le=2, description="Annual volatility")


@register(
    "data_source",
    name="synthetic_markets",
    version="2.0.0",
    tags=["synthetic", "intraday", "offline"],
    capabilities={"asset:equity", "freq:intraday"},
)
class SyntheticMarkets(DataSource):
    """Deterministic intraday bars for demos and tests."""

    Params = SyntheticMarketParams
    params: SyntheticMarketParams
    source_version: ClassVar[str] = "2"

    def describe(self) -> list[DatasetInfo]:
        """Datasets offered."""
        return [
            DatasetInfo(
                source="synthetic_markets",
                dataset="gbm_intraday",
                description="Intraday OHLCV on exchange sessions",
                frequencies=(Frequency.MIN5, Frequency.MIN15, Frequency.H1, Frequency.MIN1),
                asset_classes=(AssetClass.EQUITY,),
                quality="synthetic",
            ),
        ]

    def search_instruments(self, query: str) -> list[Instrument]:
        """Any symbol is accepted."""
        sym = query.strip().upper() or "AAA"
        return [Instrument(id=sym, symbol=sym)]

    def fetch(self, request: DataRequest) -> MarketData:
        """Dispatch on dataset."""
        if not request.instruments:
            raise DataError("Give root symbols or tickers")
        if request.dataset == "gbm_intraday":
            return self._intraday(request)
        raise DataError(f"Unknown dataset '{request.dataset}'")

    def _intraday(self, request: DataRequest) -> MarketData:
        minutes = request.frequency.minutes
        if minutes is None:
            raise DataError("gbm_intraday needs an intraday frequency")
        sessions = TradingCalendar().sessions(request.start, request.end)
        per_day = SESSION_MINUTES // minutes
        n = len(sessions) * per_day
        stamps = [
            datetime.combine(s, SESSION_OPEN, tzinfo=UTC) + timedelta(minutes=k * minutes)
            for s in sessions
            for k in range(per_day)
        ]
        step = self.params.annual_vol * np.sqrt(
            minutes / (SESSION_MINUTES * TradingCalendar().sessions_per_year())
        )
        frames = []
        for inst in request.instruments:
            rng = np.random.default_rng(_seed(inst, "intraday", request.options.get("seed", 7)))
            # a little intraday mean reversion around a slow trend
            shocks = rng.standard_normal(n) * step
            close = 50.0 * np.exp(np.cumsum(shocks - 0.3 * np.concatenate(([0], shocks[:-1]))))
            open_ = np.concatenate(([close[0]], close[:-1]))
            spread = np.abs(rng.normal(0, step, n)) * close
            frames.append(
                pl.DataFrame(
                    {
                        C.TIMESTAMP: stamps,
                        C.INSTRUMENT: inst,
                        C.OPEN: open_,
                        C.HIGH: np.maximum(open_, close) + spread,
                        C.LOW: np.minimum(open_, close) - spread,
                        C.CLOSE: close,
                        C.VOLUME: rng.lognormal(10, 0.3, n),
                    }
                )
            )
        return MarketData(
            normalize_frame(pl.concat(frames)),
            request.frequency,
            {i: Instrument(id=i, symbol=i) for i in request.instruments},
            metadata={"source": "synthetic_markets", "quality": "synthetic"},
        )

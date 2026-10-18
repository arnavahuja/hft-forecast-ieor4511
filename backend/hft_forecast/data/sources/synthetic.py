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
    AssetClass,
    DataRequest,
    DatasetInfo,
    Frequency,
    Instrument,
    MarketData,
)
from hft_forecast.data.normalize import normalize_frame

_DEFAULT_SEED: Final = 7
_BASE_PRICE: Final = 100.0
_BASE_VOLUME: Final = 1_000_000.0
_INTRADAY_RANGE: Final = 0.01
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


@register(
    "data_source",
    name="synthetic",
    version="1.0.0",
    tags=["synthetic", "offline", "demo"],
    capabilities={"asset:equity", "freq:daily", "freq:intraday"},
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
        ]

    def search_instruments(self, query: str) -> list[Instrument]:
        """Any query is a valid synthetic instrument."""
        sym = query.strip().upper() or "SYN"
        return [Instrument(id=sym, symbol=sym)]

    def fetch(self, request: DataRequest) -> MarketData:
        """Generate deterministic daily bars (or factor returns)."""
        if request.dataset == "ff_factors":
            return self._factors(request)
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

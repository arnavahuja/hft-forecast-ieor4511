"""Shared test helpers: tiny deterministic datasets and configs."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import polars as pl

from hft_forecast.core import columns as C
from hft_forecast.core.calendar import PERIODS_PER_YEAR_KEY
from hft_forecast.core.run_config import BacktestConfig, DataSpec, ExecutionSpec, PluginRef
from hft_forecast.core.types import DataRequest, MarketData
from hft_forecast.data.sources.synthetic import SyntheticSource

PPY = 252.0  # tests use an explicit annualization factor for hand-computed values


def make_market(
    closes: dict[str, list[float]],
    opens: dict[str, list[float]] | None = None,
    start: date = date(2024, 1, 1),
    extra: dict[str, dict[str, list[float]]] | None = None,
) -> MarketData:
    """MarketData from per-instrument close lists on consecutive days."""
    frames = []
    for inst, values in closes.items():
        n = len(values)
        cols: dict[str, Any] = {
            C.TIMESTAMP: [
                datetime.combine(start + timedelta(days=i), datetime.min.time(), tzinfo=UTC)
                for i in range(n)
            ],
            C.INSTRUMENT: [inst] * n,
            C.CLOSE: [float(v) for v in values],
        }
        if opens is not None:
            cols[C.OPEN] = [float(v) for v in opens[inst]]
        for field, per_inst in (extra or {}).items():
            cols[field] = [float(v) for v in per_inst[inst]]
        frames.append(pl.DataFrame(cols))
    return MarketData(pl.concat(frames), metadata={PERIODS_PER_YEAR_KEY: PPY})


def synthetic(
    instruments: tuple[str, ...] = ("AAA", "BBB", "CCC", "DDD"),
    start: date = date(2018, 1, 1),
    end: date = date(2020, 12, 31),
    seed: int = 7,
) -> MarketData:
    """Deterministic synthetic daily OHLCV."""
    request = DataRequest(
        source="synthetic",
        dataset="gbm",
        instruments=instruments,
        start=start,
        end=end,
        options={"seed": seed},
    )
    return SyntheticSource().fetch(request).with_metadata(**{PERIODS_PER_YEAR_KEY: PPY})


def config(
    strategy: str = "buy_and_hold",
    params: dict[str, Any] | None = None,
    instruments: tuple[str, ...] = ("AAA",),
    **execution: Any,
) -> BacktestConfig:
    """A synthetic-data config."""
    return BacktestConfig(
        data=DataSpec(
            source="synthetic",
            dataset="gbm",
            instruments=list(instruments),
            start=date(2018, 1, 1),
            end=date(2020, 12, 31),
        ),
        strategy=PluginRef(name=strategy, params=params or {}),
        execution=ExecutionSpec(**execution),
    )


def with_required_fields(strategy: Any, data: MarketData, seed: int = 11) -> MarketData:
    """Add random positive panels for fields a strategy requires but the data lacks."""
    import numpy as np
    import polars as pl

    missing = [f for f in strategy.data_requirements().fields if not data.has_field(f)]
    if not missing:
        return data
    rng = np.random.default_rng(seed)
    frame = data.frame.with_columns(
        [pl.Series(f, rng.uniform(0.1, 2.0, data.frame.height)) for f in missing]
    )
    return MarketData(frame, data.frequency, data.instrument_meta, data.metadata)

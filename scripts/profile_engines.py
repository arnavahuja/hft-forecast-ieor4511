"""Rough engine timings on synthetic data: python scripts/profile_engines.py"""

from __future__ import annotations

import time
from datetime import date

from hft_forecast.core.calendar import PERIODS_PER_YEAR_KEY
from hft_forecast.core.registry import registry
from hft_forecast.core.run_config import BacktestConfig
from hft_forecast.core.types import DataRequest
from hft_forecast.data.sources.synthetic import SyntheticSource
from hft_forecast.engine.base import RunOptions
from hft_forecast.engine.event.engine import EventEngine
from hft_forecast.engine.pipeline import Pipeline
from hft_forecast.engine.vectorized import VectorizedEngine
from hft_forecast.analytics.service import AnalyticsService
from hft_forecast.services.plugins import PluginService

PPY = 252.0


def data(n_inst: int, years: int):
    req = DataRequest(source="synthetic", dataset="gbm",
                      instruments=tuple(f"S{i:04d}" for i in range(n_inst)),
                      start=date(2024 - years, 1, 1), end=date(2023, 12, 31))
    return SyntheticSource().fetch(req).with_metadata(**{PERIODS_PER_YEAR_KEY: PPY})


def timed(label: str, fn) -> None:
    t0 = time.perf_counter()
    fn()
    print(f"{label:<60} {time.perf_counter() - t0:8.3f} s")


def main() -> None:
    PluginService(None).load()
    cost = registry("cost_model").get("bps_notional").create()
    cfg = BacktestConfig.model_validate({
        "data": {"source": "synthetic", "start": "2004-01-01", "end": "2023-12-31"},
        "strategy": {"name": "sma_crossover"}})
    for n_inst, years in ((10, 20), (100, 20), (500, 20)):
        d = data(n_inst, years)
        strat = registry("strategy").get("sma_crossover").create()
        opts = RunOptions(periods_per_year=PPY)
        res = None

        def vec() -> None:
            nonlocal res
            res = VectorizedEngine([cost]).run(cfg, d, strat, Pipeline(), opts)

        timed(f"vectorized  {n_inst:>4} instruments x {len(d.timestamps)} bars", vec)
        timed(f"metrics     {n_inst:>4} instruments", lambda: AnalyticsService().compute(res))
        if n_inst <= 100:
            timed(f"event       {n_inst:>4} instruments x {len(d.timestamps)} bars",
                  lambda: EventEngine([cost]).run(cfg, d, strat, Pipeline(), opts))


if __name__ == "__main__":
    main()

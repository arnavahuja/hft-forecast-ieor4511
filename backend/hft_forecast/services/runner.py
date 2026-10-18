"""Backtest runner: resolves plugins, loads data, runs the engine, computes metrics."""

from __future__ import annotations

import functools
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import structlog

from hft_forecast.analytics.service import AnalyticsService
from hft_forecast.analytics.stats import align_series
from hft_forecast.core import columns as C
from hft_forecast.core.calendar import PERIODS_PER_YEAR_KEY, TradingCalendar
from hft_forecast.core.errors import CompatibilityError, ConfigError
from hft_forecast.core.interfaces import (
    CostModel,
    FillModel,
    PortfolioConstructor,
    SlippageModel,
    Strategy,
    UniverseProvider,
)
from hft_forecast.core.registry import PluginKind, registry
from hft_forecast.core.results import BacktestResult
from hft_forecast.core.run_config import BacktestConfig, PluginRef
from hft_forecast.core.specs import MetricValue
from hft_forecast.core.types import DataRequest, FloatArray, MarketData, pl_times
from hft_forecast.data.service import DataService
from hft_forecast.engine.base import CASH_RETURNS_KEY, Engine, RunOptions
from hft_forecast.engine.event.engine import EventEngine
from hft_forecast.engine.pipeline import NamedOverlay, Pipeline
from hft_forecast.engine.vectorized import VectorizedEngine
from hft_forecast.services.compatibility import CompatibilityChecker

log = structlog.get_logger(__name__)

ProgressFn = Callable[[float, str], None]
BENCHMARK_SEPARATOR = ":"
DEFAULT_DATASET = "daily"
RISK_FREE_FIELD = "rf"


@functools.lru_cache(maxsize=1)
def git_commit() -> str | None:
    """Current git commit of the checkout (``None`` outside git)."""
    root = Path(__file__).resolve().parents[3]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


@dataclass
class PreparedRun:
    """Everything needed to execute a run."""

    config: BacktestConfig
    data: MarketData
    strategy: Strategy
    pipeline: Pipeline
    engine: Engine
    options: RunOptions
    plugin_versions: dict[str, str]
    data_refs: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Components:
    """Plugin instances of a configuration."""

    strategy: Strategy
    pipeline: Pipeline
    engine: Engine
    universe: UniverseProvider
    versions: dict[str, str]


@dataclass(frozen=True)
class RunOutcome:
    """Result of a run plus provenance."""

    result: BacktestResult
    metrics: dict[str, MetricValue]
    plugin_versions: dict[str, str]
    data_refs: dict[str, Any]
    timings: dict[str, float]


def _create(kind: PluginKind, ref: PluginRef, versions: dict[str, str]) -> Any:
    spec = registry(kind).get(ref.name)
    versions[f"{kind}:{spec.name}"] = spec.version
    return spec.create(ref.params)


def parse_benchmark(benchmark: str, default_source: str) -> tuple[str, str]:
    """Split ``"source:SYMBOL"`` (or ``"SYMBOL"``) into ``(source, symbol)``."""
    if BENCHMARK_SEPARATOR in benchmark:
        source, symbol = benchmark.split(BENCHMARK_SEPARATOR, 1)
        return source, symbol
    return default_source, benchmark


def parse_series_ref(ref: str, default_source: str, default_dataset: str) -> tuple[str, str, str]:
    """``"source:dataset:SYMBOL"``, ``"source:SYMBOL"`` or ``"SYMBOL"`` -> the three parts."""
    parts = ref.split(BENCHMARK_SEPARATOR)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], default_dataset, parts[1]
    return default_source, default_dataset, ref


def graft_series(data: MarketData, other: MarketData, instrument: str) -> MarketData:
    """Add ``instrument`` from ``other`` (another source) to ``data`` on its bar grid.

    Bars are matched by day, or by month when the two sources date monthly bars differently.
    Only close and the ``ret`` field are carried over.
    """
    sub = other.select_instruments([instrument])
    grid = data.timestamps
    cols: dict[str, Any] = {
        C.TIMESTAMP: pl_times(grid),
        C.INSTRUMENT: instrument,
        C.CLOSE: align_series(grid, sub.timestamps, sub.panel(C.CLOSE)[:, 0]),
    }
    if sub.has_field(C.RETURN):
        cols[C.RETURN] = align_series(grid, sub.timestamps, sub.panel(C.RETURN)[:, 0])
    frame = pl.DataFrame(cols).filter(pl.col(C.CLOSE).is_not_nan())
    meta = {k: v for k, v in other.instrument_meta.items() if k == instrument}
    return data.concat(MarketData(frame, data.frequency, meta))


class BacktestRunner:
    """Orchestrates a backtest run.

    Args:
        data: Data service.
        analytics: Analytics service.
        checker: Compatibility checker.
        engine_factories: Optional override of engine construction by name.
    """

    def __init__(
        self,
        data: DataService,
        analytics: AnalyticsService | None = None,
        checker: CompatibilityChecker | None = None,
        engine_factories: dict[str, Callable[..., Engine]] | None = None,
    ) -> None:
        self.data = data
        self.analytics = analytics or AnalyticsService()
        self.checker = checker or CompatibilityChecker()
        self.engine_factories = engine_factories or {}

    # ---- engine construction

    def _engine(
        self,
        config: BacktestConfig,
        costs: list[CostModel],
        slippage: SlippageModel | None,
        fill: FillModel | None,
    ) -> Engine:
        name = config.execution.engine
        if name in self.engine_factories:
            return self.engine_factories[name](costs, slippage, fill)
        if name == "vectorized":
            return VectorizedEngine(costs, slippage)
        if name == "event":
            return EventEngine(costs, slippage, fill)
        raise ConfigError(f"Unknown engine '{name}'")

    # ---- data

    def _load_data(
        self, config: BacktestConfig, instruments: list[str], universe_code: str | None = None
    ) -> tuple[MarketData, str | None, dict[str, Any]]:
        spec = config.data
        refs: dict[str, Any] = {}
        bench_id: str | None = None
        wanted = list(instruments)
        bench_source = bench_symbol = None
        bench_dataset = DEFAULT_DATASET
        foreign = False
        if config.benchmark:
            bench_source, bench_dataset, bench_symbol = parse_series_ref(
                config.benchmark, spec.source, spec.dataset
            )
            if config.benchmark.count(BENCHMARK_SEPARATOR) == 1:
                bench_dataset = spec.dataset if bench_source == spec.source else DEFAULT_DATASET
            foreign = (bench_source, bench_dataset) != (spec.source, spec.dataset)
            if not foreign and bench_symbol not in wanted:
                wanted.append(bench_symbol)
        request = spec.to_request(wanted, universe=universe_code)
        data = self.data.fetch(request)
        refs["main"] = {"request": request.cache_key(), **_cache_ref(data)}
        if bench_symbol is not None:
            bench_id = bench_symbol
            if foreign:
                breq = spec.to_request([bench_symbol]).model_copy(
                    update={"source": bench_source, "dataset": bench_dataset, "universe": None}
                )
                bdata = self.data.fetch(breq)
                refs["benchmark"] = {"request": breq.cache_key(), **_cache_ref(bdata)}
                if bench_symbol not in bdata.instruments:
                    raise ConfigError(f"Benchmark '{bench_symbol}' not found in {bench_source}")
                data = graft_series(data, bdata, bench_symbol)
            if bench_id not in data.instruments:
                bench_id = None
        return data, bench_id, refs

    def _load_risk_free(
        self, config: BacktestConfig, data: MarketData, refs: dict[str, Any]
    ) -> tuple[FloatArray | None, list[str]]:
        """Per-bar risk-free returns on the data grid (``config.risk_free``)."""
        if not config.risk_free:
            return None, []
        parts = config.risk_free.split(BENCHMARK_SEPARATOR)
        if len(parts) not in (2, 3):
            raise ConfigError("risk_free must be 'source:dataset[:field]'")
        source, dataset = parts[0], parts[1]
        field_name = parts[2] if len(parts) == 3 else RISK_FREE_FIELD
        request = DataRequest(
            source=source,
            dataset=dataset,
            start=config.data.start,
            end=config.data.end,
            frequency=config.data.frequency,
        )
        rf_data = self.data.fetch(request)
        refs["risk_free"] = {"request": request.cache_key(), **_cache_ref(rf_data)}
        if not rf_data.has_field(field_name):
            raise ConfigError(
                f"Risk-free dataset has no '{field_name}' field",
                details={"fields": list(rf_data.fields)},
            )
        first = rf_data.select_instruments([rf_data.instruments[0]])
        series = align_series(data.timestamps, first.timestamps, first.panel(field_name)[:, 0])
        warnings = []
        missing = int((~np.isfinite(series[1:])).sum())
        if missing:
            warnings.append(f"Risk-free series missing on {missing} bars (treated as 0)")
        out = np.nan_to_num(series, nan=0.0)
        if len(out):
            out[0] = 0.0
        return out, warnings

    def components(self, config: BacktestConfig) -> Components:
        """Instantiate every plugin of a configuration (no data loading)."""
        versions: dict[str, str] = {}
        strategy: Strategy = _create(PluginKind.STRATEGY, config.strategy, versions)
        constructor: PortfolioConstructor | None = (
            _create(PluginKind.PORTFOLIO_CONSTRUCTOR, config.constructor, versions)
            if config.constructor
            else None
        )
        overlays = tuple(
            NamedOverlay(ref.name, _create(PluginKind.OVERLAY, ref, versions))
            for ref in config.overlays
        )
        costs: list[CostModel] = [_create(PluginKind.COST_MODEL, r, versions) for r in config.costs]
        slippage: SlippageModel | None = (
            _create(PluginKind.SLIPPAGE_MODEL, config.slippage, versions)
            if config.slippage
            else None
        )
        fill: FillModel | None = (
            _create(PluginKind.FILL_MODEL, config.fill_model, versions)
            if config.fill_model
            else None
        )
        universe_ref = config.data.universe or PluginRef(name="static")
        universe: UniverseProvider = _create(PluginKind.UNIVERSE, universe_ref, versions)
        engine = self._engine(config, costs, slippage, fill)
        return Components(strategy, Pipeline(constructor, overlays), engine, universe, versions)

    # ---- preparation

    def prepare(self, config: BacktestConfig) -> PreparedRun:
        """Validate the config, instantiate plugins and load data.

        Raises:
            CompatibilityError: If the configuration is invalid.
        """
        report = self.checker.check(config)
        if not report.ok:
            raise CompatibilityError(
                "Invalid run configuration",
                details={"issues": [i.model_dump() for i in report.issues]},
            )
        warnings = [i.message for i in report.issues]
        comps = self.components(config)
        strategy, universe = comps.strategy, comps.universe
        candidates = universe.candidates(list(config.data.instruments))
        universe_code = universe.request_universe()

        data, bench_id, refs = self._load_data(config, candidates, universe_code)
        data, locked = self._apply_split(config, data)
        if locked:
            warnings.append(f"Test period locked: data truncated at {locked}")
        calendar = TradingCalendar(config.execution.calendar)
        ppy = calendar.periods_per_year(data.frequency)
        data = data.with_metadata(**{PERIODS_PER_YEAR_KEY: ppy})
        if data.is_empty():
            raise ConfigError("No data in the selected date range")
        if bench_id:
            data = data.with_metadata(benchmark_id=bench_id)
        if universe_code:  # the source decided the instruments (point-in-time universe)
            strat_instruments = tuple(i for i in data.instruments if i != bench_id)
        else:
            strat_instruments = tuple(i for i in data.instruments if i in set(candidates))
        needs_bench = strategy.data_requirements().needs_benchmark
        if needs_bench and bench_id and bench_id not in strat_instruments:
            strat_instruments = (*strat_instruments, bench_id)
        membership = self._membership(universe, data, strat_instruments)
        risk_free, rf_warnings = self._load_risk_free(config, data, refs)
        warnings.extend(rf_warnings)
        options = RunOptions(
            periods_per_year=ppy,
            benchmark_id=bench_id,
            strategy_instruments=tuple(sorted(strat_instruments)),
            membership=membership,
            extras={CASH_RETURNS_KEY: risk_free} if risk_free is not None else {},
        )
        return PreparedRun(
            config,
            data,
            strategy,
            comps.pipeline,
            comps.engine,
            options,
            comps.versions,
            refs,
            warnings,
        )

    @staticmethod
    def _apply_split(config: BacktestConfig, data: MarketData) -> tuple[MarketData, str | None]:
        split = config.split
        if split is None or split.test_unlocked or split.validation_end is None:
            return data, None
        cutoff = split.validation_end
        return data.between(None, cutoff), cutoff.isoformat()

    @staticmethod
    def _membership(
        universe: UniverseProvider, data: MarketData, strat_instruments: tuple[str, ...]
    ) -> FloatArray:
        sub = data.select_instruments(list(strat_instruments))
        mask_sub = universe.membership(sub)
        full = np.ones((len(data.timestamps), len(data.instruments)))
        rows = np.searchsorted(data.timestamps, sub.timestamps)
        for j, inst in enumerate(sub.instruments):
            col = data.instruments.index(inst)
            full[:, col] = 0.0
            full[rows, col] = mask_sub[:, j]
        return full

    # ---- run

    def run(
        self,
        config: BacktestConfig,
        *,
        progress: ProgressFn | None = None,
        cancelled: Callable[[], bool] | None = None,
        n_trials: int = 1,
        trial_sharpes: list[float] | None = None,
        compute_metrics: bool = True,
    ) -> RunOutcome:
        """Prepare and execute a run, then compute metrics."""
        t0 = time.perf_counter()
        prepared = self.prepare(config)
        t_prep = time.perf_counter()
        opts = prepared.options
        options = RunOptions(
            periods_per_year=opts.periods_per_year,
            benchmark_id=opts.benchmark_id,
            strategy_instruments=opts.strategy_instruments,
            membership=opts.membership,
            progress=progress,
            cancelled=cancelled,
            extras=dict(opts.extras),
        )
        result = prepared.engine.run(
            config, prepared.data, prepared.strategy, prepared.pipeline, options
        )
        result.metadata["warnings"] = prepared.warnings
        t_run = time.perf_counter()
        metrics = (
            self.analytics.compute(
                result,
                n_trials=n_trials,
                trial_sharpes=trial_sharpes or [],
            )
            if compute_metrics
            else {}
        )
        t_metrics = time.perf_counter()
        timings = {
            "prepare_seconds": round(t_prep - t0, 4),
            "engine_seconds": round(t_run - t_prep, 4),
            "metrics_seconds": round(t_metrics - t_run, 4),
            "total_seconds": round(t_metrics - t0, 4),
        }
        log.info("run_finished", strategy=config.strategy.name, **timings)
        return RunOutcome(result, metrics, prepared.plugin_versions, prepared.data_refs, timings)


def _cache_ref(data: MarketData) -> dict[str, Any]:
    meta = data.metadata
    return {k: meta[k] for k in ("cache_key", "cache_version", "source") if k in meta}

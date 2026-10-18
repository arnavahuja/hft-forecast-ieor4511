"""Rolling risk charts: Sharpe, volatility, beta, correlation to the benchmark."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import numpy as np
from pydantic import Field

from hft_forecast.analytics import chartkit as K
from hft_forecast.core.interfaces import ChartBuilder, ChartInput
from hft_forecast.core.numeric import rolling_mean, rolling_std
from hft_forecast.core.params import ChartParams
from hft_forecast.core.registry import register
from hft_forecast.core.specs import ChartSpec, ChartType, MetricFormat
from hft_forecast.core.types import FloatArray


class RollingParams(ChartParams):
    """Rolling window."""

    window: int = Field(126, ge=10, le=2520, description="Window in bars")


def rolling_sharpe(r: FloatArray, window: int, ppy: float) -> FloatArray:
    """Trailing annualized Sharpe."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return rolling_mean(r, window) / rolling_std(r, window) * np.sqrt(ppy)


def rolling_beta_corr(r: FloatArray, b: FloatArray, window: int) -> tuple[FloatArray, FloatArray]:
    """Trailing beta and correlation of ``r`` on ``b``."""
    mr, mb = rolling_mean(r, window), rolling_mean(b, window)
    cov = rolling_mean(r * b, window) - mr * mb
    var_b = rolling_mean(b * b, window) - mb**2
    var_r = rolling_mean(r * r, window) - mr**2
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = cov / var_b
        corr = cov / np.sqrt(var_b * var_r)
    return beta, corr


class _Rolling(ChartBuilder):
    """Base for single-run rolling charts."""

    Params = RollingParams
    params: RollingParams
    group: ClassVar[str] = "Rolling"
    scopes: ClassVar[frozenset[str]] = frozenset({"single", "compare"})
    chart_id: ClassVar[str] = ""
    fmt: ClassVar[MetricFormat] = MetricFormat.RATIO
    with_benchmark: ClassVar[bool] = True

    def values(self, r: FloatArray, run: ChartInput) -> FloatArray:
        """Compute the rolling statistic."""
        raise NotImplementedError

    def build(self, runs: Sequence[ChartInput]) -> ChartSpec:
        """Build the chart for one or several runs."""
        series = [
            K.time_series(
                run.label,
                run.result.timestamps,
                self.values(run.result.returns, run),
                run_id=run.run_id,
            )
            for run in runs
        ]
        first = runs[0]
        if self.with_benchmark and len(runs) == 1 and first.result.benchmark_returns is not None:
            b = K.benchmark_series(first, self.values(first.result.benchmark_returns, first))
            if b is not None:
                series.append(b)
        return ChartSpec(
            id=self.chart_id,
            title=f"{self.title} ({self.params.window} bars)",
            type=ChartType.LINE,
            x_axis=K.time_axis(),
            y_axes=[K.value_axis("", self.fmt)],
            series=series,
            time_series=True,
        )


@register("chart", name="rolling_sharpe", version="1.0.0", tags=["rolling"])
class RollingSharpe(_Rolling):
    """Trailing annualized Sharpe ratio."""

    title: ClassVar[str] = "Rolling Sharpe"
    chart_id: ClassVar[str] = "rolling_sharpe"

    def values(self, r: FloatArray, run: ChartInput) -> FloatArray:
        """Rolling Sharpe."""
        return rolling_sharpe(r, self.params.window, run.result.periods_per_year)


@register("chart", name="rolling_volatility", version="1.0.0", tags=["rolling"])
class RollingVolatility(_Rolling):
    """Trailing annualized volatility."""

    title: ClassVar[str] = "Rolling volatility"
    chart_id: ClassVar[str] = "rolling_volatility"
    fmt: ClassVar[MetricFormat] = MetricFormat.PERCENT

    def values(self, r: FloatArray, run: ChartInput) -> FloatArray:
        """Rolling volatility."""
        return rolling_std(r, self.params.window) * np.sqrt(run.result.periods_per_year)


class _BenchRolling(_Rolling):
    with_benchmark: ClassVar[bool] = False

    def applicable(self, runs: Sequence[ChartInput]) -> bool:
        """Needs benchmark returns."""
        return bool(runs) and all(r.result.benchmark_returns is not None for r in runs)


@register("chart", name="rolling_beta", version="1.0.0", tags=["rolling", "benchmark"])
class RollingBeta(_BenchRolling):
    """Trailing beta to the benchmark."""

    title: ClassVar[str] = "Rolling beta"
    chart_id: ClassVar[str] = "rolling_beta"
    fmt: ClassVar[MetricFormat] = MetricFormat.NUMBER

    def values(self, r: FloatArray, run: ChartInput) -> FloatArray:
        """Rolling beta."""
        assert run.result.benchmark_returns is not None
        return rolling_beta_corr(r, run.result.benchmark_returns, self.params.window)[0]


@register("chart", name="rolling_correlation", version="1.0.0", tags=["rolling", "benchmark"])
class RollingCorrelation(_BenchRolling):
    """Trailing correlation to the benchmark."""

    title: ClassVar[str] = "Rolling correlation"
    chart_id: ClassVar[str] = "rolling_correlation"
    fmt: ClassVar[MetricFormat] = MetricFormat.NUMBER

    def values(self, r: FloatArray, run: ChartInput) -> FloatArray:
        """Rolling correlation."""
        assert run.result.benchmark_returns is not None
        return rolling_beta_corr(r, run.result.benchmark_returns, self.params.window)[1]

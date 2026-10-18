"""Performance charts: equity curve, cumulative excess return, rolling returns."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import numpy as np
from pydantic import Field

from hft_forecast.analytics import chartkit as K
from hft_forecast.core.interfaces import ChartBuilder, ChartInput
from hft_forecast.core.numeric import rolling_sum
from hft_forecast.core.params import ChartParams
from hft_forecast.core.registry import register
from hft_forecast.core.specs import ChartSpec, ChartType, MetricFormat, SeriesRole
from hft_forecast.core.types import FloatArray


class EquityParams(ChartParams):
    """Equity curve options."""

    log_scale: bool = Field(False, description="Logarithmic y axis")


@register("chart", name="equity_curve", version="1.0.0", tags=["performance"])
class EquityCurve(ChartBuilder):
    """Growth of 1 for the strategy with the benchmark dashed."""

    Params = EquityParams
    params: EquityParams
    title: ClassVar[str] = "Equity curve"
    group: ClassVar[str] = "Performance"
    scopes: ClassVar[frozenset[str]] = frozenset({"single"})

    def build(self, runs: Sequence[ChartInput]) -> ChartSpec:
        """Build the chart."""
        run = runs[0]
        res = run.result
        series = [
            K.time_series(run.label, res.timestamps, K.wealth(res.returns), run_id=run.run_id)
        ]
        if res.benchmark_returns is not None:
            b = K.benchmark_series(run, K.wealth(res.benchmark_returns))
            if b is not None:
                series.append(b)
        return ChartSpec(
            id="equity_curve",
            title=self.title,
            type=ChartType.LINE,
            x_axis=K.time_axis(),
            y_axes=[K.value_axis("Growth of 1", MetricFormat.NUMBER, self.params.log_scale)],
            series=series,
            time_series=True,
        )


@register("chart", name="equity_log", version="1.0.0", tags=["performance"])
class EquityLog(EquityCurve):
    """Equity curve on a logarithmic scale."""

    title: ClassVar[str] = "Equity curve (log)"

    def build(self, runs: Sequence[ChartInput]) -> ChartSpec:
        """Build the chart."""
        spec = EquityCurve(EquityParams(log_scale=True)).build(runs)
        return spec.model_copy(update={"id": "equity_log", "title": self.title})


@register("chart", name="cumulative_excess", version="1.0.0", tags=["performance", "benchmark"])
class CumulativeExcess(ChartBuilder):
    """Cumulative return of the strategy minus the benchmark (geometric)."""

    title: ClassVar[str] = "Cumulative excess return"
    group: ClassVar[str] = "Performance"
    scopes: ClassVar[frozenset[str]] = frozenset({"single"})

    def applicable(self, runs: Sequence[ChartInput]) -> bool:
        """Needs a benchmark."""
        return bool(runs) and runs[0].result.benchmark_returns is not None

    def build(self, runs: Sequence[ChartInput]) -> ChartSpec:
        """Build the chart."""
        res = runs[0].result
        assert res.benchmark_returns is not None
        excess = K.wealth(res.returns) / K.wealth(res.benchmark_returns) - 1.0
        return ChartSpec(
            id="cumulative_excess",
            title=self.title,
            type=ChartType.AREA,
            x_axis=K.time_axis(),
            y_axes=[K.value_axis("Excess", MetricFormat.PERCENT)],
            series=[
                K.time_series("Excess vs benchmark", res.timestamps, excess, run_id=runs[0].run_id)
            ],
            time_series=True,
        )


class RollingReturnParams(ChartParams):
    """Rolling return window."""

    window: int = Field(252, ge=5, le=2520, description="Window in bars")


@register("chart", name="rolling_returns", version="1.0.0", tags=["performance", "rolling"])
class RollingReturns(ChartBuilder):
    """Trailing compounded return over a rolling window."""

    Params = RollingReturnParams
    params: RollingReturnParams
    title: ClassVar[str] = "Rolling returns"
    group: ClassVar[str] = "Performance"
    scopes: ClassVar[frozenset[str]] = frozenset({"single"})

    def build(self, runs: Sequence[ChartInput]) -> ChartSpec:
        """Build the chart."""
        run = runs[0]
        res = run.result
        w = self.params.window

        def roll(r: FloatArray) -> FloatArray:
            return np.expm1(rolling_sum(np.log1p(np.nan_to_num(r)), w))

        series = [K.time_series(run.label, res.timestamps, roll(res.returns), run_id=run.run_id)]
        if res.benchmark_returns is not None:
            b = K.benchmark_series(run, roll(res.benchmark_returns))
            if b is not None:
                series.append(b)
        return ChartSpec(
            id="rolling_returns",
            title=f"{self.title} ({w} bars)",
            type=ChartType.LINE,
            x_axis=K.time_axis(),
            y_axes=[K.value_axis("Return", MetricFormat.PERCENT)],
            series=series,
            time_series=True,
        )


@register("chart", name="drawdown_underwater", version="1.0.0", tags=["drawdown"])
class Underwater(ChartBuilder):
    """Underwater plot: drawdown from the running peak."""

    title: ClassVar[str] = "Underwater"
    group: ClassVar[str] = "Drawdown"
    scopes: ClassVar[frozenset[str]] = frozenset({"single"})

    def build(self, runs: Sequence[ChartInput]) -> ChartSpec:
        """Build the chart."""
        from hft_forecast.analytics.stats import drawdown_series

        run = runs[0]
        res = run.result
        series = [
            K.time_series(
                run.label,
                res.timestamps,
                drawdown_series(res.returns),
                role=SeriesRole.NEGATIVE,
                run_id=run.run_id,
            )
        ]
        if res.benchmark_returns is not None:
            b = K.benchmark_series(run, drawdown_series(res.benchmark_returns))
            if b is not None:
                series.append(b)
        return ChartSpec(
            id="drawdown_underwater",
            title=self.title,
            type=ChartType.AREA,
            x_axis=K.time_axis(),
            y_axes=[K.value_axis("Drawdown", MetricFormat.PERCENT)],
            series=series,
            time_series=True,
        )

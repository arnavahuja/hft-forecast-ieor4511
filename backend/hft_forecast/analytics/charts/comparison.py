"""Comparison charts across several runs.

Series carry ``run_id`` so the front end keeps each run's color identical across charts.
Runs are aligned on their common timestamps where a chart compares paths.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Final

import numpy as np
from pydantic import Field

from hft_forecast.analytics import chartkit as K
from hft_forecast.analytics.stats import drawdown_series
from hft_forecast.core.interfaces import ChartBuilder, ChartInput
from hft_forecast.core.params import ChartParams
from hft_forecast.core.registry import register
from hft_forecast.core.specs import (
    Axis,
    AxisType,
    ChartSeries,
    ChartSpec,
    ChartType,
    MetricFormat,
)
from hft_forecast.core.types import FloatArray, TimeArray

RADAR_METRICS: Final = (
    ("cagr", "CAGR", True),
    ("sharpe", "Sharpe", True),
    ("sortino", "Sortino", True),
    ("max_drawdown", "Max DD", True),  # less negative is better
    ("ann_vol", "Volatility", False),
    ("win_rate", "Win rate", True),
)


def aligned_returns(runs: Sequence[ChartInput]) -> tuple[TimeArray, FloatArray]:
    """Returns of all runs on their common timestamps ``(T, R)``."""
    common = runs[0].result.timestamps
    for run in runs[1:]:
        common = np.intersect1d(common, run.result.timestamps)
    cols = []
    for run in runs:
        idx = np.searchsorted(run.result.timestamps, common)
        r = run.result.returns[idx].copy()
        if len(r):
            r[0] = 0.0
        cols.append(r)
    return common, np.column_stack(cols) if cols else np.empty((0, 0))


class _Compare(ChartBuilder):
    group: ClassVar[str] = "Comparison"
    scopes: ClassVar[frozenset[str]] = frozenset({"compare"})


@register("chart", name="compare_equity", version="1.0.0", tags=["comparison"])
class CompareEquity(_Compare):
    """Growth of 1 for each run over the common period."""

    title: ClassVar[str] = "Equity curves"

    def build(self, runs: Sequence[ChartInput]) -> ChartSpec:
        """Build the chart."""
        ts, rets = aligned_returns(runs)
        series = [
            K.time_series(run.label, ts, K.wealth(rets[:, k]), run_id=run.run_id)
            for k, run in enumerate(runs)
        ]
        return ChartSpec(
            id="compare_equity",
            title=self.title,
            type=ChartType.LINE,
            x_axis=K.time_axis(),
            y_axes=[K.value_axis("Growth of 1", MetricFormat.NUMBER)],
            series=series,
            time_series=True,
        )


@register("chart", name="compare_drawdowns", version="1.0.0", tags=["comparison"])
class CompareDrawdowns(_Compare):
    """Underwater curves for each run over the common period."""

    title: ClassVar[str] = "Drawdowns"

    def build(self, runs: Sequence[ChartInput]) -> ChartSpec:
        """Build the chart."""
        ts, rets = aligned_returns(runs)
        series = [
            K.time_series(run.label, ts, drawdown_series(rets[:, k]), run_id=run.run_id)
            for k, run in enumerate(runs)
        ]
        return ChartSpec(
            id="compare_drawdowns",
            title=self.title,
            type=ChartType.LINE,
            x_axis=K.time_axis(),
            y_axes=[K.value_axis("Drawdown", MetricFormat.PERCENT)],
            series=series,
            time_series=True,
        )


class MetricBarParams(ChartParams):
    """Which metric to compare."""

    metric: str = Field("sharpe", description="Metric key")


@register("chart", name="compare_metric_bars", version="1.0.0", tags=["comparison"])
class CompareMetricBars(_Compare):
    """One metric across runs as bars."""

    Params = MetricBarParams
    params: MetricBarParams
    title: ClassVar[str] = "Metric comparison"

    def build(self, runs: Sequence[ChartInput]) -> ChartSpec:
        """Build the chart."""
        key = self.params.metric
        labels = K.label_rows(runs)
        data = [[run.label, run.metrics.get(key)] for run in runs]
        series = [
            ChartSeries(name=run.label, data=[row], run_id=run.run_id)
            for run, row in zip(runs, data, strict=True)
        ]
        return ChartSpec(
            id="compare_metric_bars",
            title=f"{key}",
            type=ChartType.BAR,
            x_axis=Axis(type=AxisType.CATEGORY, categories=labels),
            y_axes=[Axis(name=key)],
            series=series,
            options={"metric": key},
        )


@register("chart", name="risk_return_scatter", version="1.0.0", tags=["comparison"])
class RiskReturnScatter(_Compare):
    """Annualized volatility versus CAGR for each run."""

    title: ClassVar[str] = "Risk vs return"

    def build(self, runs: Sequence[ChartInput]) -> ChartSpec:
        """Build the chart."""
        series = [
            ChartSeries(
                name=run.label,
                run_id=run.run_id,
                symbol_size=14,
                data=[[run.metrics.get("ann_vol"), run.metrics.get("cagr")]],
            )
            for run in runs
        ]
        return ChartSpec(
            id="risk_return_scatter",
            title=self.title,
            type=ChartType.SCATTER,
            x_axis=Axis(type=AxisType.VALUE, name="Volatility", format=MetricFormat.PERCENT),
            y_axes=[Axis(name="CAGR", format=MetricFormat.PERCENT)],
            series=series,
        )


@register("chart", name="correlation_matrix", version="1.0.0", tags=["comparison"])
class CorrelationMatrix(_Compare):
    """Correlation of run returns over the common period."""

    title: ClassVar[str] = "Return correlation"

    def build(self, runs: Sequence[ChartInput]) -> ChartSpec:
        """Build the chart."""
        _, rets = aligned_returns(runs)
        labels = K.label_rows(runs)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(rets[1:].T) if rets.shape[0] > 2 else np.eye(len(runs))
        corr = np.atleast_2d(corr)
        cells = [
            [i, j, float(corr[i, j]) if np.isfinite(corr[i, j]) else None]
            for i in range(len(runs))
            for j in range(len(runs))
        ]
        return ChartSpec(
            id="correlation_matrix",
            title=self.title,
            type=ChartType.HEATMAP,
            x_axis=Axis(type=AxisType.CATEGORY, categories=labels),
            y_axes=[Axis(type=AxisType.CATEGORY, categories=labels)],
            series=[ChartSeries(name="Correlation", data=cells)],
            options={
                "value_format": MetricFormat.NUMBER.value,
                "diverging": True,
                "min": -1,
                "max": 1,
            },
        )


@register("chart", name="radar_metrics", version="1.0.0", tags=["comparison"])
class RadarMetrics(_Compare):
    """Normalized key metrics per run (1 = best among the selected runs)."""

    title: ClassVar[str] = "Metric profile"

    def build(self, runs: Sequence[ChartInput]) -> ChartSpec:
        """Build the chart."""
        indicators = []
        matrix = []
        for key, label, higher in RADAR_METRICS:
            vals = np.array(
                [np.nan if run.metrics.get(key) is None else run.metrics[key] for run in runs],
                dtype=np.float64,
            )
            if not np.isfinite(vals).any():
                continue
            lo, hi = np.nanmin(vals), np.nanmax(vals)
            span = hi - lo if hi > lo else 1.0
            norm = (vals - lo) / span if higher else (hi - vals) / span
            if hi == lo:
                norm = np.ones_like(vals)
            indicators.append({"name": label, "max": 1.0})
            matrix.append(np.nan_to_num(norm, nan=0.0))
        values = np.array(matrix).T if matrix else np.zeros((len(runs), 0))
        series = [
            ChartSeries(name=run.label, run_id=run.run_id, data=[[float(v) for v in values[k]]])
            for k, run in enumerate(runs)
        ]
        return ChartSpec(
            id="radar_metrics",
            title=self.title,
            type=ChartType.RADAR,
            series=series,
            options={"indicators": indicators},
        )


class RelativeParams(ChartParams):
    """Rolling relative performance window."""

    window: int = Field(126, ge=5, le=2520, description="Window in bars")


@register("chart", name="rolling_relative", version="1.0.0", tags=["comparison", "rolling"])
class RollingRelative(_Compare):
    """Each run's trailing return minus the first run's (the reference)."""

    Params = RelativeParams
    params: RelativeParams
    title: ClassVar[str] = "Rolling relative performance"

    def applicable(self, runs: Sequence[ChartInput]) -> bool:
        """Needs two or more runs."""
        return len(runs) >= 2

    def build(self, runs: Sequence[ChartInput]) -> ChartSpec:
        """Build the chart."""
        from hft_forecast.core.numeric import rolling_sum

        ts, rets = aligned_returns(runs)
        roll = np.expm1(rolling_sum(np.log1p(rets), self.params.window))
        series = [
            K.time_series(
                f"{run.label} vs {runs[0].label}", ts, roll[:, k] - roll[:, 0], run_id=run.run_id
            )
            for k, run in enumerate(runs)
            if k > 0
        ]
        return ChartSpec(
            id="rolling_relative",
            title=f"{self.title} ({self.params.window} bars)",
            type=ChartType.LINE,
            x_axis=K.time_axis(),
            y_axes=[K.value_axis("Relative return", MetricFormat.PERCENT)],
            series=series,
            time_series=True,
        )

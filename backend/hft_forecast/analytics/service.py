"""Metric computation and chart building over registered plugins."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, Final

import numpy as np
import polars as pl
import structlog

from hft_forecast.analytics.downsample import lttb_indices
from hft_forecast.core.interfaces import ChartBuilder, ChartInput, Metric, MetricContext
from hft_forecast.core.registry import PluginKind, registry
from hft_forecast.core.results import BacktestResult
from hft_forecast.core.specs import (
    AxisType,
    ChartSpec,
    ChartType,
    MetricDescriptor,
    MetricKind,
    MetricValue,
)
from hft_forecast.core.types import IntArray

log = structlog.get_logger(__name__)

DEFAULT_MAX_POINTS: Final = 2000
_DOWNSAMPLE_TYPES: Final = {ChartType.LINE, ChartType.AREA, ChartType.STACKED_AREA}


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value) if np.isfinite(value) else None
    return None


class AnalyticsService:
    """Computes metrics and builds charts.

    Args:
        metric_params: Optional parameters per metric plugin name.
        risk_free_rate: Annual risk-free rate for Sharpe-type metrics.
    """

    def __init__(
        self, metric_params: dict[str, dict[str, Any]] | None = None, risk_free_rate: float = 0.0
    ) -> None:
        self.metric_params = dict(metric_params or {})
        self.risk_free_rate = risk_free_rate

    # ---- metrics

    def _metrics(self) -> list[tuple[str, Metric]]:
        out = []
        for spec in registry(PluginKind.METRIC).specs():
            out.append((spec.name, spec.create(self.metric_params.get(spec.name))))
        return out

    def descriptors(self) -> list[MetricDescriptor]:
        """All metric descriptors, in plugin order."""
        return [d for spec in registry(PluginKind.METRIC).specs() for d in spec.cls.descriptors]

    def compute(
        self,
        result: BacktestResult,
        *,
        n_trials: int = 1,
        trial_sharpes: Sequence[float] = (),
        factors: pl.DataFrame | None = None,
    ) -> dict[str, MetricValue]:
        """Compute every metric for the strategy and (where meaningful) the benchmark."""
        values: dict[str, MetricValue] = {}
        base = MetricContext(
            result=result,
            returns=result.returns,
            benchmark_returns=result.benchmark_returns,
            periods_per_year=result.periods_per_year,
            risk_free_rate=self.risk_free_rate,
            factors=factors,
            n_trials=n_trials,
            trial_sharpes=tuple(trial_sharpes),
        )
        bench_ctx = None
        if result.benchmark_returns is not None:
            bench_ctx = MetricContext(
                result=result,
                returns=result.benchmark_returns,
                benchmark_returns=None,
                periods_per_year=result.periods_per_year,
                is_benchmark=True,
                risk_free_rate=self.risk_free_rate,
                n_trials=1,
            )
        for name, metric in self._metrics():
            try:
                out = metric.compute(base)
                bout = metric.compute(bench_ctx) if bench_ctx is not None else {}
            except Exception as exc:
                log.warning("metric_failed", metric=name, error=str(exc))
                continue
            for d in metric.descriptors:
                raw = out.get(d.key)
                if d.kind is MetricKind.TABLE:
                    values[d.key] = MetricValue(
                        key=d.key, table=raw if isinstance(raw, list) else None
                    )
                else:
                    values[d.key] = MetricValue(
                        key=d.key,
                        value=_as_float(raw),
                        benchmark=_as_float(bout.get(d.key)) if d.benchmark else None,
                    )
        return values

    @staticmethod
    def scalars(values: dict[str, MetricValue]) -> dict[str, float | None]:
        """Scalar values only, keyed by metric."""
        return {k: v.value for k, v in values.items() if v.table is None}

    # ---- charts

    def chart_specs(self, scope: str | None = None) -> list[dict[str, Any]]:
        """Descriptions of chart plugins, optionally filtered by scope."""
        out = []
        for spec in registry(PluginKind.CHART).specs():
            cls: type[ChartBuilder] = spec.cls
            if scope is not None and scope not in cls.scopes:
                continue
            out.append(
                {
                    "name": spec.name,
                    "title": cls.title or spec.name,
                    "group": cls.group,
                    "scopes": sorted(cls.scopes),
                    "description": spec.description,
                }
            )
        return out

    def build_chart(
        self,
        name: str,
        inputs: Sequence[ChartInput],
        params: dict[str, Any] | None = None,
        max_points: int | None = DEFAULT_MAX_POINTS,
    ) -> ChartSpec:
        """Build a chart and downsample long time series with LTTB."""
        builder: ChartBuilder = registry(PluginKind.CHART).get(name).create(params)
        if not builder.applicable(inputs):
            return ChartSpec(
                id=name,
                title=builder.title or name,
                type=ChartType.LINE,
                warnings=["Not applicable to the selected runs"],
            )
        spec = builder.build(inputs)
        if max_points and spec.x_axis.type is AxisType.TIME and spec.type in _DOWNSAMPLE_TYPES:
            spec = downsample_spec(spec, max_points)
        return spec


def _epoch(value: object) -> float:
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    return float(value)  # type: ignore[arg-type]  # numeric x values


def downsample_spec(spec: ChartSpec, max_points: int) -> ChartSpec:
    """Downsample each series of a time chart to at most ``max_points`` points.

    Stacked series share one index set so stacks stay aligned.
    """
    series = []
    shared: IntArray | None = None
    for s in spec.series:
        if len(s.data) <= max_points or s.render_as is ChartType.SCATTER:
            series.append(s)
            continue
        if s.stack is not None and shared is not None and len(shared) and len(s.data) > shared[-1]:
            idx = shared
        else:
            x = np.array([_epoch(row[0]) for row in s.data])
            y = np.array([np.nan if row[1] is None else float(row[1]) for row in s.data])
            idx = lttb_indices(x, y, max_points)
            if s.stack is not None:
                shared = idx
        series.append(s.model_copy(update={"data": [s.data[i] for i in idx]}))
    return spec.model_copy(update={"series": series})

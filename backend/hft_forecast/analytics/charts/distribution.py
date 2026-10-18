"""Return distribution charts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Final

import numpy as np
from pydantic import Field
from scipy import stats as sps

from hft_forecast.analytics.stats import aggregate_returns
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
    SeriesRole,
)

MONTHS: Final = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
WEEKDAYS: Final = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
EPOCH_WEEKDAY: Final = 3  # 1970-01-01 was a Thursday


class HistParams(ChartParams):
    """Histogram options."""

    bins: int = Field(60, ge=10, le=300, description="Number of bins")


@register("chart", name="return_histogram", version="1.0.0", tags=["distribution"])
class ReturnHistogram(ChartBuilder):
    """Histogram of bar returns with a fitted normal density overlay."""

    Params = HistParams
    params: HistParams
    title: ClassVar[str] = "Return distribution"
    group: ClassVar[str] = "Distribution"
    scopes: ClassVar[frozenset[str]] = frozenset({"single"})

    def build(self, runs: Sequence[ChartInput]) -> ChartSpec:
        """Build the chart."""
        r = runs[0].result.returns[1:]
        r = r[np.isfinite(r)]
        counts, edges = np.histogram(r, bins=self.params.bins, density=True)
        mids = (edges[:-1] + edges[1:]) / 2
        mu, sd = (float(r.mean()), float(r.std(ddof=1))) if r.size > 1 else (0.0, 1.0)
        normal = sps.norm.pdf(mids, mu, sd) if sd > 0 else np.zeros_like(mids)
        return ChartSpec(
            id="return_histogram",
            title=self.title,
            type=ChartType.BAR,
            x_axis=Axis(type=AxisType.VALUE, name="Return", format=MetricFormat.PERCENT),
            y_axes=[Axis(name="Density")],
            series=[
                ChartSeries(
                    name="Returns",
                    data=[[float(m), float(c)] for m, c in zip(mids, counts, strict=True)],
                    run_id=runs[0].run_id,
                ),
                ChartSeries(
                    name="Normal fit",
                    data=[[float(m), float(n)] for m, n in zip(mids, normal, strict=True)],
                    render_as=ChartType.LINE,
                    role=SeriesRole.REFERENCE,
                ),
            ],
        )


@register("chart", name="qq_plot", version="1.0.0", tags=["distribution"])
class QQPlot(ChartBuilder):
    """Quantile-quantile plot of standardized returns against the normal distribution."""

    title: ClassVar[str] = "QQ plot"
    group: ClassVar[str] = "Distribution"
    scopes: ClassVar[frozenset[str]] = frozenset({"single"})

    def build(self, runs: Sequence[ChartInput]) -> ChartSpec:
        """Build the chart."""
        r = runs[0].result.returns[1:]
        r = np.sort(r[np.isfinite(r)])
        n = r.size
        sd = r.std(ddof=1) if n > 1 else 1.0
        z = (r - r.mean()) / sd if sd > 0 else r
        theo = sps.norm.ppf((np.arange(1, n + 1) - 0.5) / max(n, 1))
        step = max(n // 1500, 1)
        pts = [[float(a), float(b)] for a, b in zip(theo[::step], z[::step], strict=True)]
        lim = float(np.nanmax(np.abs(theo))) if n else 3.0
        return ChartSpec(
            id="qq_plot",
            title=self.title,
            type=ChartType.SCATTER,
            x_axis=Axis(type=AxisType.VALUE, name="Normal quantile"),
            y_axes=[Axis(name="Sample quantile")],
            series=[
                ChartSeries(name="Returns", data=pts, run_id=runs[0].run_id, symbol_size=4),
                ChartSeries(
                    name="y = x",
                    data=[[-lim, -lim], [lim, lim]],
                    render_as=ChartType.LINE,
                    role=SeriesRole.REFERENCE,
                ),
            ],
        )


@register("chart", name="monthly_heatmap", version="1.0.0", tags=["distribution", "calendar"])
class MonthlyHeatmap(ChartBuilder):
    """Monthly returns by year and month."""

    title: ClassVar[str] = "Monthly returns"
    group: ClassVar[str] = "Distribution"
    scopes: ClassVar[frozenset[str]] = frozenset({"single"})

    def build(self, runs: Sequence[ChartInput]) -> ChartSpec:
        """Build the chart."""
        res = runs[0].result
        codes, monthly = aggregate_returns(res.returns, res.timestamps, "M")
        years = (codes // 12 + 1970).astype(int)
        months = (codes % 12).astype(int)
        year_labels = [str(y) for y in sorted(set(years.tolist()))]
        cells = [
            [int(m), year_labels.index(str(y)), float(v)]
            for y, m, v in zip(years, months, monthly, strict=True)
        ]
        return ChartSpec(
            id="monthly_heatmap",
            title=self.title,
            type=ChartType.HEATMAP,
            x_axis=Axis(type=AxisType.CATEGORY, categories=list(MONTHS)),
            y_axes=[Axis(type=AxisType.CATEGORY, categories=year_labels)],
            series=[ChartSeries(name="Monthly return", data=cells, run_id=runs[0].run_id)],
            options={"value_format": MetricFormat.PERCENT.value, "diverging": True},
        )


@register("chart", name="annual_returns", version="1.0.0", tags=["distribution", "calendar"])
class AnnualReturns(ChartBuilder):
    """Calendar-year returns, strategy versus benchmark."""

    title: ClassVar[str] = "Annual returns"
    group: ClassVar[str] = "Distribution"
    scopes: ClassVar[frozenset[str]] = frozenset({"single"})

    def build(self, runs: Sequence[ChartInput]) -> ChartSpec:
        """Build the chart."""
        res = runs[0].result
        codes, yearly = aggregate_returns(res.returns, res.timestamps, "Y")
        labels = [str(int(c) + 1970) for c in codes]
        series = [
            ChartSeries(
                name=runs[0].label,
                run_id=runs[0].run_id,
                data=[[lab, float(v)] for lab, v in zip(labels, yearly, strict=True)],
            )
        ]
        if res.benchmark_returns is not None:
            _, byearly = aggregate_returns(res.benchmark_returns, res.timestamps, "Y")
            series.append(
                ChartSeries(
                    name="Benchmark",
                    role=SeriesRole.BENCHMARK,
                    data=[[lab, float(v)] for lab, v in zip(labels, byearly, strict=True)],
                )
            )
        return ChartSpec(
            id="annual_returns",
            title=self.title,
            type=ChartType.BAR,
            x_axis=Axis(type=AxisType.CATEGORY, categories=labels),
            y_axes=[Axis(name="Return", format=MetricFormat.PERCENT)],
            series=series,
        )


class BoxParams(ChartParams):
    """Grouping for the box plot."""

    by: str = Field("month", pattern="^(month|weekday)$", description="month or weekday")


@register("chart", name="returns_boxplot", version="1.0.0", tags=["distribution", "calendar"])
class ReturnsBoxplot(ChartBuilder):
    """Distribution of bar returns grouped by calendar month or weekday."""

    Params = BoxParams
    params: BoxParams
    title: ClassVar[str] = "Returns by month / weekday"
    group: ClassVar[str] = "Distribution"
    scopes: ClassVar[frozenset[str]] = frozenset({"single"})

    def build(self, runs: Sequence[ChartInput]) -> ChartSpec:
        """Build the chart."""
        res = runs[0].result
        r = res.returns
        if self.params.by == "month":
            keys = (res.timestamps.astype("datetime64[M]").astype(np.int64) % 12).astype(int)
            labels = list(MONTHS)
        else:
            days = res.timestamps.astype("datetime64[D]").astype(np.int64)
            keys = ((days + EPOCH_WEEKDAY) % 7).astype(int)
            labels = list(WEEKDAYS)
        rows, used = [], []
        for k, lab in enumerate(labels):
            vals = r[(keys == k) & np.isfinite(r)]
            if vals.size == 0:
                continue
            q = np.quantile(vals, [0.0, 0.25, 0.5, 0.75, 1.0])
            rows.append([float(x) for x in q])
            used.append(lab)
        return ChartSpec(
            id="returns_boxplot",
            title=f"Returns by {self.params.by}",
            type=ChartType.BOXPLOT,
            x_axis=Axis(type=AxisType.CATEGORY, categories=used),
            y_axes=[Axis(name="Return", format=MetricFormat.PERCENT)],
            series=[ChartSeries(name=runs[0].label, data=rows, run_id=runs[0].run_id)],
        )

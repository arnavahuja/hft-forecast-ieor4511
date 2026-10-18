"""Before/after overlay charts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from hft_forecast.analytics import chartkit as K
from hft_forecast.core.interfaces import ChartBuilder, ChartInput
from hft_forecast.core.registry import register
from hft_forecast.core.specs import ChartSeries, ChartSpec, ChartType, MetricFormat, SeriesRole


class _OverlayChart(ChartBuilder):
    group: ClassVar[str] = "Overlays"
    scopes: ClassVar[frozenset[str]] = frozenset({"single"})

    def applicable(self, runs: Sequence[ChartInput]) -> bool:
        """Needs at least one overlay."""
        return bool(runs) and bool(runs[0].result.overlay_reports)


@register("chart", name="overlay_equity", version="1.0.0", tags=["overlays"])
class OverlayEquity(_OverlayChart):
    """Zero-cost equity before the first overlay and after each overlay in turn."""

    title: ClassVar[str] = "Equity before and after overlays"

    def build(self, runs: Sequence[ChartInput]) -> ChartSpec:
        """Build the chart."""
        res = runs[0].result
        reps = res.overlay_reports
        series: list[ChartSeries] = [
            K.time_series(
                "Before overlays",
                res.timestamps,
                K.wealth(reps[0].returns_before),
                role=SeriesRole.REFERENCE,
                dashed=True,
            )
        ]
        series += [
            K.time_series(f"+ {rep.name}", res.timestamps, K.wealth(rep.returns_after))
            for rep in reps
        ]
        return ChartSpec(
            id="overlay_equity",
            title=self.title,
            type=ChartType.LINE,
            x_axis=K.time_axis(),
            y_axes=[K.value_axis("Growth of 1 (zero cost)", MetricFormat.NUMBER)],
            series=series,
            time_series=True,
        )


@register("chart", name="overlay_exposure", version="1.0.0", tags=["overlays"])
class OverlayExposure(_OverlayChart):
    """Gross target exposure before the first overlay and after each overlay."""

    title: ClassVar[str] = "Gross exposure before and after overlays"

    def build(self, runs: Sequence[ChartInput]) -> ChartSpec:
        """Build the chart."""
        res = runs[0].result
        reps = res.overlay_reports
        series = [
            K.time_series(
                "Before overlays",
                res.timestamps,
                reps[0].gross_before,
                role=SeriesRole.REFERENCE,
                dashed=True,
            )
        ]
        series += [K.time_series(f"+ {rep.name}", res.timestamps, rep.gross_after) for rep in reps]
        return ChartSpec(
            id="overlay_exposure",
            title=self.title,
            type=ChartType.LINE,
            x_axis=K.time_axis(),
            y_axes=[K.value_axis("Gross exposure", MetricFormat.PERCENT)],
            series=series,
            time_series=True,
        )

"""Drawdown periods shaded on the equity curve."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from pydantic import Field

from hft_forecast.analytics import chartkit as K
from hft_forecast.analytics.stats import drawdown_periods
from hft_forecast.core.interfaces import ChartBuilder, ChartInput
from hft_forecast.core.params import ChartParams
from hft_forecast.core.registry import register
from hft_forecast.core.specs import (
    Annotation,
    AnnotationKind,
    ChartSpec,
    ChartType,
    MetricFormat,
    SeriesRole,
)


class TopDrawdownParams(ChartParams):
    """How many drawdowns to shade."""

    top_n: int = Field(5, ge=1, le=20, description="Number of drawdowns to shade")


@register("chart", name="top_drawdowns", version="1.0.0", tags=["drawdown"])
class TopDrawdowns(ChartBuilder):
    """Equity curve with the deepest drawdown periods shaded."""

    Params = TopDrawdownParams
    params: TopDrawdownParams
    title: ClassVar[str] = "Top drawdown periods"
    group: ClassVar[str] = "Drawdown"
    scopes: ClassVar[frozenset[str]] = frozenset({"single"})

    def build(self, runs: Sequence[ChartInput]) -> ChartSpec:
        """Build the chart."""
        run = runs[0]
        res = run.result
        ts = K.iso(res.timestamps)
        bands = []
        for k, p in enumerate(drawdown_periods(res.returns)[: self.params.top_n]):
            end = ts[p.end] if p.end is not None else ts[-1]
            bands.append(
                Annotation(
                    kind=AnnotationKind.BAND,
                    start=ts[p.start],
                    end=end,
                    label=f"#{k + 1} {p.depth:.1%}",
                    role=SeriesRole.NEGATIVE,
                )
            )
        return ChartSpec(
            id="top_drawdowns",
            title=self.title,
            type=ChartType.LINE,
            x_axis=K.time_axis(),
            y_axes=[K.value_axis("Growth of 1", MetricFormat.NUMBER)],
            series=[
                K.time_series(run.label, res.timestamps, K.wealth(res.returns), run_id=run.run_id)
            ],
            annotations=bands,
            time_series=True,
        )

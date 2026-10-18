"""Cost charts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import numpy as np

from hft_forecast.analytics import chartkit as K
from hft_forecast.core.interfaces import ChartBuilder, ChartInput
from hft_forecast.core.registry import register
from hft_forecast.core.specs import ChartSpec, ChartType, MetricFormat, SeriesRole


@register("chart", name="gross_vs_net", version="1.0.0", tags=["costs"])
class GrossVsNet(ChartBuilder):
    """Equity before and after costs."""

    title: ClassVar[str] = "Gross vs net equity"
    group: ClassVar[str] = "Costs"
    scopes: ClassVar[frozenset[str]] = frozenset({"single"})

    def build(self, runs: Sequence[ChartInput]) -> ChartSpec:
        """Build the chart."""
        res = runs[0].result
        return ChartSpec(
            id="gross_vs_net",
            title=self.title,
            type=ChartType.LINE,
            x_axis=K.time_axis(),
            y_axes=[K.value_axis("Equity", MetricFormat.CURRENCY)],
            series=[
                K.time_series(
                    "Gross",
                    res.timestamps,
                    res.gross_equity,
                    role=SeriesRole.REFERENCE,
                    dashed=True,
                ),
                K.time_series("Net", res.timestamps, res.equity, run_id=runs[0].run_id),
            ],
            time_series=True,
        )


@register("chart", name="cumulative_costs", version="1.0.0", tags=["costs"])
class CumulativeCosts(ChartBuilder):
    """Cumulative costs by category, stacked."""

    title: ClassVar[str] = "Cumulative costs"
    group: ClassVar[str] = "Costs"
    scopes: ClassVar[frozenset[str]] = frozenset({"single"})

    def applicable(self, runs: Sequence[ChartInput]) -> bool:
        """Needs at least one cost category."""
        return bool(runs) and bool(runs[0].result.costs)

    def build(self, runs: Sequence[ChartInput]) -> ChartSpec:
        """Build the chart."""
        res = runs[0].result
        series = [
            K.time_series(cat.capitalize(), res.timestamps, np.cumsum(v), stack="c")
            for cat, v in sorted(res.costs.items())
        ]
        return ChartSpec(
            id="cumulative_costs",
            title=self.title,
            type=ChartType.STACKED_AREA,
            x_axis=K.time_axis(),
            y_axes=[K.value_axis("Cost", MetricFormat.CURRENCY)],
            series=series,
            time_series=True,
        )

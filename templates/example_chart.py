"""Example chart: a user chart (uses an existing chart type, so no front-end change)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import numpy as np

from hft_forecast.analytics import chartkit as K
from hft_forecast.core import ChartBuilder, ChartInput, register
from hft_forecast.core.specs import ChartSpec, ChartType, MetricFormat


@register("chart", name="example_chart", version="0.1.0", tags=["user"])
class ExampleChart(ChartBuilder):
    """Cumulative number of up bars."""

    title: ClassVar[str] = "Example chart"
    group: ClassVar[str] = "Performance"
    scopes: ClassVar[frozenset[str]] = frozenset({"single"})

    def build(self, runs: Sequence[ChartInput]) -> ChartSpec:
        """Return a library-neutral chart spec."""
        res = runs[0].result
        return ChartSpec(
            id="example_chart", title=self.title, type=ChartType.LINE, x_axis=K.time_axis(),
            y_axes=[K.value_axis("Up bars", MetricFormat.INTEGER)], time_series=True,
            series=[K.time_series(runs[0].label, res.timestamps,
                                  np.cumsum(res.returns > 0).astype(float))],
        )

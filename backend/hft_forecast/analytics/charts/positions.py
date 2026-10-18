"""Position and exposure charts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import numpy as np
from pydantic import Field

from hft_forecast.analytics import chartkit as K
from hft_forecast.core.interfaces import ChartBuilder, ChartInput
from hft_forecast.core.params import ChartParams
from hft_forecast.core.registry import register
from hft_forecast.core.specs import ChartSpec, ChartType, MetricFormat, SeriesRole


class WeightsParams(ChartParams):
    """Stacked weights options."""

    max_instruments: int = Field(15, ge=1, le=100, description="Largest positions shown")


@register("chart", name="weights_over_time", version="1.0.0", tags=["positions"])
class WeightsOverTime(ChartBuilder):
    """End-of-bar weights as a stacked area (largest positions, rest grouped as Other)."""

    Params = WeightsParams
    params: WeightsParams
    title: ClassVar[str] = "Weights over time"
    group: ClassVar[str] = "Positions"
    scopes: ClassVar[frozenset[str]] = frozenset({"single"})

    def build(self, runs: Sequence[ChartInput]) -> ChartSpec:
        """Build the chart."""
        res = runs[0].result
        w = res.weights
        order = np.argsort(-np.abs(w).mean(axis=0)) if w.size else np.array([], dtype=int)
        keep = order[: self.params.max_instruments]
        rest = order[self.params.max_instruments :]
        series = [
            K.time_series(res.instruments[j], res.timestamps, w[:, j], stack="w") for j in keep
        ]
        if rest.size:
            series.append(K.time_series("Other", res.timestamps, w[:, rest].sum(axis=1), stack="w"))
        return ChartSpec(
            id="weights_over_time",
            title=self.title,
            type=ChartType.STACKED_AREA,
            x_axis=K.time_axis(),
            y_axes=[K.value_axis("Weight", MetricFormat.PERCENT)],
            series=series,
            time_series=True,
        )


@register("chart", name="exposure", version="1.0.0", tags=["positions"])
class Exposure(ChartBuilder):
    """Gross, net, long and short exposure."""

    title: ClassVar[str] = "Gross and net exposure"
    group: ClassVar[str] = "Positions"
    scopes: ClassVar[frozenset[str]] = frozenset({"single"})

    def build(self, runs: Sequence[ChartInput]) -> ChartSpec:
        """Build the chart."""
        res = runs[0].result
        ts = res.timestamps
        return ChartSpec(
            id="exposure",
            title=self.title,
            type=ChartType.LINE,
            x_axis=K.time_axis(),
            y_axes=[K.value_axis("Exposure", MetricFormat.PERCENT)],
            series=[
                K.time_series("Gross", ts, res.gross_exposure),
                K.time_series("Net", ts, res.net_exposure),
                K.time_series("Long", ts, res.long_exposure, role=SeriesRole.POSITIVE),
                K.time_series("Short", ts, -res.short_exposure, role=SeriesRole.NEGATIVE),
            ],
            time_series=True,
        )


@register("chart", name="leverage", version="1.0.0", tags=["positions"])
class Leverage(ChartBuilder):
    """Leverage (gross exposure) and number of holdings."""

    title: ClassVar[str] = "Leverage and holdings"
    group: ClassVar[str] = "Positions"
    scopes: ClassVar[frozenset[str]] = frozenset({"single"})

    def build(self, runs: Sequence[ChartInput]) -> ChartSpec:
        """Build the chart."""
        res = runs[0].result
        ts = res.timestamps
        holdings = K.time_series("Holdings", ts, res.n_positions)
        holdings.y_axis = 1
        return ChartSpec(
            id="leverage",
            title=self.title,
            type=ChartType.LINE,
            x_axis=K.time_axis(),
            y_axes=[
                K.value_axis("Leverage", MetricFormat.RATIO),
                K.value_axis("Holdings", MetricFormat.INTEGER),
            ],
            series=[K.time_series("Leverage", ts, res.gross_exposure), holdings],
            time_series=True,
        )


@register("chart", name="turnover", version="1.0.0", tags=["positions", "costs"])
class Turnover(ChartBuilder):
    """One-way turnover per bar."""

    title: ClassVar[str] = "Turnover"
    group: ClassVar[str] = "Positions"
    scopes: ClassVar[frozenset[str]] = frozenset({"single"})

    def build(self, runs: Sequence[ChartInput]) -> ChartSpec:
        """Build the chart."""
        res = runs[0].result
        return ChartSpec(
            id="turnover",
            title=self.title,
            type=ChartType.BAR,
            x_axis=K.time_axis(),
            y_axes=[K.value_axis("Turnover", MetricFormat.PERCENT)],
            series=[K.time_series("Turnover", res.timestamps, res.turnover)],
            time_series=True,
        )

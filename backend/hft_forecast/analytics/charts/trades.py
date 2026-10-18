"""Trade charts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import polars as pl
from pydantic import Field

from hft_forecast.analytics import chartkit as K
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
from hft_forecast.core.types import np_times


class _TradeChart(ChartBuilder):
    group: ClassVar[str] = "Trades"
    scopes: ClassVar[frozenset[str]] = frozenset({"single"})

    def applicable(self, runs: Sequence[ChartInput]) -> bool:
        """Needs at least one trade."""
        return bool(runs) and runs[0].result.trades.height > 0


class PriceParams(ChartParams):
    """Which instrument to plot."""

    instrument: str = Field("", description="Instrument id (default: most traded)")


@register("chart", name="price_with_trades", version="1.0.0", tags=["trades"])
class PriceWithTrades(_TradeChart):
    """Close price with trade entry and exit markers."""

    Params = PriceParams
    params: PriceParams
    title: ClassVar[str] = "Price with entries and exits"

    def build(self, runs: Sequence[ChartInput]) -> ChartSpec:
        """Build the chart."""
        res = runs[0].result
        trades = res.trades
        inst = self.params.instrument
        if not inst or inst not in res.instruments:
            inst = str(
                trades.group_by("instrument_id")
                .len()
                .sort("len", descending=True)
                .get_column("instrument_id")[0]
            )
        j = res.instruments.index(inst)
        t = trades.filter(pl.col("instrument_id") == inst)

        def markers(time_col: str, price_col: str, name: str, role: SeriesRole) -> ChartSeries:
            sub = t.filter(pl.col(time_col).is_not_null())
            times = K.iso(np_times(sub.get_column(time_col)))
            prices = sub.get_column(price_col).to_list()
            return ChartSeries(
                name=name,
                data=[[a, b] for a, b in zip(times, prices, strict=True)],
                render_as=ChartType.SCATTER,
                role=role,
                symbol_size=9,
            )

        closed = t.filter(~pl.col("is_open"))
        exits_series = markers("exit_time", "exit_price", "Exit", SeriesRole.NEGATIVE)
        if closed.height == 0:
            exits_series.data = []
        return ChartSpec(
            id="price_with_trades",
            title=f"{inst}: price with entries and exits",
            type=ChartType.LINE,
            x_axis=K.time_axis(),
            y_axes=[K.value_axis("Price")],
            series=[
                K.time_series(inst, res.timestamps, res.prices[:, j]),
                markers("entry_time", "entry_price", "Entry", SeriesRole.POSITIVE),
                exits_series,
            ],
            time_series=True,
        )


@register("chart", name="trade_pnl_distribution", version="1.0.0", tags=["trades"])
class TradePnlDistribution(_TradeChart):
    """Histogram of trade returns."""

    title: ClassVar[str] = "Trade return distribution"

    def build(self, runs: Sequence[ChartInput]) -> ChartSpec:
        """Build the chart."""
        rets = runs[0].result.trades.get_column("return").to_numpy()
        rets = rets[np.isfinite(rets)]
        counts, edges = np.histogram(rets, bins=min(40, max(rets.size, 1)))
        mids = (edges[:-1] + edges[1:]) / 2
        return ChartSpec(
            id="trade_pnl_distribution",
            title=self.title,
            type=ChartType.BAR,
            x_axis=Axis(type=AxisType.VALUE, name="Trade return", format=MetricFormat.PERCENT),
            y_axes=[Axis(name="Trades", format=MetricFormat.INTEGER)],
            series=[
                ChartSeries(
                    name="Trades",
                    data=[[float(m), int(c)] for m, c in zip(mids, counts, strict=True)],
                )
            ],
        )


@register("chart", name="mae_mfe", version="1.0.0", tags=["trades"])
class MaeMfe(_TradeChart):
    """Maximum adverse versus favourable excursion per trade."""

    title: ClassVar[str] = "MAE vs MFE"

    def build(self, runs: Sequence[ChartInput]) -> ChartSpec:
        """Build the chart."""
        t = runs[0].result.trades
        win = t.filter(pl.col("pnl") > 0)
        loss = t.filter(pl.col("pnl") <= 0)

        def pts(frame: pl.DataFrame) -> list[list[float]]:
            return [
                [float(a), float(b)]
                for a, b in zip(
                    frame.get_column("mae").to_list(),
                    frame.get_column("mfe").to_list(),
                    strict=True,
                )
            ]

        return ChartSpec(
            id="mae_mfe",
            title=self.title,
            type=ChartType.SCATTER,
            x_axis=Axis(type=AxisType.VALUE, name="MAE", format=MetricFormat.PERCENT),
            y_axes=[Axis(name="MFE", format=MetricFormat.PERCENT)],
            series=[
                ChartSeries(name="Winners", data=pts(win), role=SeriesRole.POSITIVE, symbol_size=6),
                ChartSeries(name="Losers", data=pts(loss), role=SeriesRole.NEGATIVE, symbol_size=6),
            ],
        )


@register("chart", name="holding_period_histogram", version="1.0.0", tags=["trades"])
class HoldingPeriodHistogram(_TradeChart):
    """Distribution of trade holding periods in bars."""

    title: ClassVar[str] = "Holding periods"

    def build(self, runs: Sequence[ChartInput]) -> ChartSpec:
        """Build the chart."""
        bars = runs[0].result.trades.get_column("bars_held").to_numpy()
        counts, edges = np.histogram(bars, bins=min(30, max(len(np.unique(bars)), 1)))
        mids = (edges[:-1] + edges[1:]) / 2
        return ChartSpec(
            id="holding_period_histogram",
            title=self.title,
            type=ChartType.BAR,
            x_axis=Axis(type=AxisType.VALUE, name="Bars held"),
            y_axes=[Axis(name="Trades", format=MetricFormat.INTEGER)],
            series=[
                ChartSeries(
                    name="Trades",
                    data=[[float(m), int(c)] for m, c in zip(mids, counts, strict=True)],
                )
            ],
        )


@register("chart", name="pnl_by_instrument", version="1.0.0", tags=["trades"])
class PnlByInstrument(_TradeChart):
    """Cumulative trade PnL by instrument."""

    title: ClassVar[str] = "PnL by instrument"

    def build(self, runs: Sequence[ChartInput]) -> ChartSpec:
        """Build the chart."""
        agg = (
            runs[0]
            .result.trades.group_by("instrument_id")
            .agg(pl.col("pnl").sum())
            .sort("pnl", descending=True)
        )
        names = agg.get_column("instrument_id").to_list()
        pnl = agg.get_column("pnl").to_list()
        return ChartSpec(
            id="pnl_by_instrument",
            title=self.title,
            type=ChartType.BAR,
            x_axis=Axis(type=AxisType.CATEGORY, categories=names),
            y_axes=[Axis(name="PnL", format=MetricFormat.CURRENCY)],
            series=[
                ChartSeries(
                    name="PnL", data=[[n, float(p)] for n, p in zip(names, pnl, strict=True)]
                )
            ],
            options={"color_by_sign": True},
        )

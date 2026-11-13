"""Chart specs for an :class:`~hft_forecast.ml.eda.EdaReport`.

Built here and stored with the report rather than registered as chart plugins: the plugin
registry serves charts computed from a :class:`BacktestResult`, and an EDA report has no
equity curve. The specs are the same library-neutral :class:`ChartSpec` the front end
already renders, so the UI needs no new chart code.
"""

from __future__ import annotations

from typing import Final

import polars as pl

from hft_forecast.core import columns as C
from hft_forecast.core.specs import (
    Annotation,
    AnnotationKind,
    Axis,
    AxisType,
    ChartSeries,
    ChartSpec,
    ChartType,
    MetricFormat,
    SeriesRole,
)
from hft_forecast.ml.eda import BUCKET, FEATURE, HORIZON_SECONDS, EdaReport

TOP_FEATURES: Final = 8
"""Features drawn in the per-feature line charts. More than this and the legend is unreadable;
the tables keep every feature."""


def eda_charts(report: EdaReport) -> list[ChartSpec]:
    """Every chart of a report, in display order."""
    top = _top_features(report.ranking, TOP_FEATURES)
    charts = [
        _ic_ranking(report.ranking),
        _bucket_profile(report.buckets, top),
        _decay(report.decay, top, report.horizon_seconds),
        _coverage(report.coverage),
        _correlation(report.correlation),
        _missing(report.missing),
    ]
    return [c for c in charts if c.series]


def _top_features(ranking: pl.DataFrame, n: int) -> list[str]:
    """The strongest features by absolute mean IC (the ranking is already sorted)."""
    return ranking.head(n).get_column(FEATURE).to_list() if not ranking.is_empty() else []


def _ic_ranking(ranking: pl.DataFrame) -> ChartSpec:
    """Mean information coefficient per feature, with the t-statistic as a second axis."""
    if ranking.is_empty():
        return _empty("eda_ic_ranking", "Feature IC ranking")
    features = ranking.get_column(FEATURE).to_list()
    return ChartSpec(
        id="eda_ic_ranking",
        title="Feature IC ranking (train sessions only)",
        type=ChartType.BAR,
        x_axis=Axis(type=AxisType.CATEGORY, categories=features, name="Feature"),
        y_axes=[
            Axis(name="Mean IC across stock-days", format=MetricFormat.NUMBER),
            Axis(name="t-statistic", format=MetricFormat.NUMBER),
        ],
        series=[
            ChartSeries(
                name="Mean IC",
                data=_rows(ranking, FEATURE, "mean_ic"),
            ),
            ChartSeries(
                name="t-statistic",
                data=_rows(ranking, FEATURE, "t_stat"),
                render_as=ChartType.SCATTER,
                role=SeriesRole.REFERENCE,
                y_axis=1,
                symbol_size=7,
            ),
        ],
        annotations=[Annotation(kind=AnnotationKind.HLINE, value=0, label="")],
    )


def _bucket_profile(buckets: pl.DataFrame, top: list[str]) -> ChartSpec:
    """Median stock-day mean forward return per rank bucket, one line per feature."""
    if buckets.is_empty():
        return _empty("eda_bucket_profile", "Monotonicity profile")
    labels = [
        str(b)
        for b in buckets.get_column(BUCKET).unique().sort().to_list()  # 0 = lowest
    ]
    series = [
        ChartSeries(
            name=name,
            data=_rows(buckets.filter(pl.col(FEATURE) == name), BUCKET, "median_block_mean_bps"),
        )
        for name in top
    ]
    return ChartSpec(
        id="eda_bucket_profile",
        title="Forward return by feature rank bucket (bps)",
        type=ChartType.LINE,
        x_axis=Axis(type=AxisType.CATEGORY, categories=labels, name="Rank bucket (0 = lowest)"),
        y_axes=[Axis(name="Median stock-day mean (bps)", format=MetricFormat.NUMBER)],
        series=[s for s in series if s.data],
        annotations=[Annotation(kind=AnnotationKind.HLINE, value=0, label="")],
    )


def _decay(decay: pl.DataFrame, top: list[str], configured: int) -> ChartSpec:
    """IC against horizon, with the configured horizon marked."""
    if decay.is_empty():
        return _empty("eda_ic_decay", "IC by horizon")
    series = [
        ChartSeries(
            name=name,
            data=_rows(decay.filter(pl.col(FEATURE) == name), HORIZON_SECONDS, "mean_ic"),
        )
        for name in top
    ]
    return ChartSpec(
        id="eda_ic_decay",
        title="Signal decay: mean IC by forecast horizon",
        type=ChartType.LINE,
        x_axis=Axis(type=AxisType.VALUE, name="Horizon (seconds)"),
        y_axes=[Axis(name="Mean IC across stock-days", format=MetricFormat.NUMBER)],
        series=[s for s in series if s.data],
        annotations=[
            Annotation(
                kind=AnnotationKind.VLINE,
                value=configured,
                label=f"configured ({configured}s)",
                role=SeriesRole.REFERENCE,
            ),
            Annotation(kind=AnnotationKind.HLINE, value=0, label=""),
        ],
    )


def _coverage(frame: pl.DataFrame) -> ChartSpec:
    """Usable share and typical spread per instrument."""
    if frame.is_empty():
        return _empty("eda_coverage", "Coverage by instrument")
    names = frame.get_column(C.INSTRUMENT).to_list()
    series = [
        ChartSeries(
            name="Model-ready bars",
            data=_rows(frame, C.INSTRUMENT, "model_ready_share"),
        ),
        ChartSeries(
            name="Valid labels",
            data=_rows(frame, C.INSTRUMENT, "label_valid_share"),
            role=SeriesRole.BAND,
        ),
    ]
    axes = [Axis(name="Share of bars", format=MetricFormat.PERCENT, min=0, max=1)]
    if "median_spread_bps" in frame.columns:
        series.append(
            ChartSeries(
                name="Median spread (bps)",
                data=_rows(frame, C.INSTRUMENT, "median_spread_bps"),
                render_as=ChartType.SCATTER,
                role=SeriesRole.REFERENCE,
                y_axis=1,
                symbol_size=7,
            )
        )
        axes.append(Axis(name="Spread (bps)", format=MetricFormat.NUMBER))
    return ChartSpec(
        id="eda_coverage",
        title="Coverage and spread by instrument",
        type=ChartType.BAR,
        x_axis=Axis(type=AxisType.CATEGORY, categories=names, name="Instrument"),
        y_axes=axes,
        series=[s for s in series if s.data],
    )


def _correlation(frame: pl.DataFrame) -> ChartSpec:
    """Feature-by-feature correlation heatmap."""
    if frame.is_empty():
        return _empty("eda_correlation", "Feature correlation")
    features = frame.get_column(FEATURE).unique(maintain_order=True).to_list()
    index = {name: i for i, name in enumerate(features)}
    cells = [
        [index[row[FEATURE]], index[row["other"]], row["correlation"]]
        for row in frame.iter_rows(named=True)
        if row["other"] in index and row["correlation"] is not None
    ]
    return ChartSpec(
        id="eda_correlation",
        title="Feature correlation (redundancy check)",
        type=ChartType.HEATMAP,
        x_axis=Axis(type=AxisType.CATEGORY, categories=features),
        y_axes=[Axis(type=AxisType.CATEGORY, categories=features)],
        series=[ChartSeries(name="Correlation", data=cells)],
        options={
            "value_format": MetricFormat.NUMBER.value,
            "diverging": True,
            "min": -1,
            "max": 1,
        },
    )


def _missing(frame: pl.DataFrame) -> ChartSpec:
    """Coverage heatmap: instruments across, features down.

    Drawn as the share *present* rather than the share missing, because the shared heatmap
    palette runs low-red to high-green: a fully covered feature should not look alarming.
    The ``missing`` table keeps the number the other way round.
    """
    if frame.is_empty():
        return _empty("eda_missing", "Missing features")
    names = frame.get_column(C.INSTRUMENT).unique(maintain_order=True).to_list()
    features = frame.get_column(FEATURE).unique(maintain_order=True).to_list()
    columns = {name: i for i, name in enumerate(names)}
    rows = {name: i for i, name in enumerate(features)}
    cells = [
        [columns[row[C.INSTRUMENT]], rows[row[FEATURE]], 1.0 - row["missing_share"]]
        for row in frame.iter_rows(named=True)
        if row["missing_share"] is not None
    ]
    return ChartSpec(
        id="eda_missing",
        title="Share of bars a feature is present",
        type=ChartType.HEATMAP,
        x_axis=Axis(type=AxisType.CATEGORY, categories=names),
        y_axes=[Axis(type=AxisType.CATEGORY, categories=features)],
        series=[ChartSeries(name="Missing share", data=cells)],
        options={"value_format": MetricFormat.PERCENT.value, "min": 0, "max": 1},
    )


def _rows(frame: pl.DataFrame, x: str, y: str) -> list[list[object]]:
    """``[x, y]`` pairs, skipping rows whose value is null."""
    if frame.is_empty() or y not in frame.columns:
        return []
    return [[row[x], row[y]] for row in frame.iter_rows(named=True) if row.get(y) is not None]


def _empty(chart_id: str, title: str) -> ChartSpec:
    """A spec with no series, dropped by :func:`eda_charts`."""
    return ChartSpec(id=chart_id, title=title, type=ChartType.BAR)

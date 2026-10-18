"""Library-neutral chart specs and metric descriptors.

The backend describes charts and metrics with these models; the front end renders them
through registries keyed by :class:`ChartType` and :class:`MetricFormat`. Adding a chart that
uses an existing type therefore needs backend code only.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------- metrics


class MetricFormat(StrEnum):
    """How a metric value is displayed."""

    PERCENT = "percent"
    RATIO = "ratio"
    NUMBER = "number"
    INTEGER = "integer"
    CURRENCY = "currency"
    BARS = "bars"
    DAYS = "days"


class MetricKind(StrEnum):
    """Shape of a metric value."""

    SCALAR = "scalar"
    TABLE = "table"


OUTPUT_CONFIG = ConfigDict(json_schema_serialization_defaults_required=True)
"""Response models mark defaulted fields as required in the serialization schema."""


class MetricDescriptor(BaseModel):
    """Static description of one metric output.

    Attributes:
        key: Unique key across all metrics.
        label: Display label.
        group: Group name for the UI (Return, Risk, ...).
        format: Display format.
        higher_is_better: Direction used for best/worst highlighting (``None`` = neutral).
        kind: Scalar or table.
        description: Tooltip text.
        benchmark: Whether the metric is also computed for the benchmark.
    """

    model_config = ConfigDict(frozen=True, json_schema_serialization_defaults_required=True)

    key: str
    label: str
    group: str
    format: MetricFormat = MetricFormat.NUMBER
    higher_is_better: bool | None = None
    kind: MetricKind = MetricKind.SCALAR
    description: str = ""
    benchmark: bool = True


class MetricValue(BaseModel):
    """A computed metric for one run.

    Attributes:
        key: Metric key.
        value: Scalar value (``None`` when undefined).
        benchmark: Same metric computed on the benchmark, if meaningful.
        table: Rows for table metrics.
    """

    model_config = OUTPUT_CONFIG

    key: str
    value: float | None = None
    benchmark: float | None = None
    table: list[dict[str, Any]] | None = None


# --------------------------------------------------------------------------- charts


class ChartType(StrEnum):
    """Chart types the front end knows how to render."""

    LINE = "line"
    AREA = "area"
    STACKED_AREA = "stacked_area"
    BAR = "bar"
    HISTOGRAM = "histogram"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    BOXPLOT = "boxplot"
    RADAR = "radar"


class AxisType(StrEnum):
    """Axis scale."""

    TIME = "time"
    VALUE = "value"
    LOG = "log"
    CATEGORY = "category"


class SeriesRole(StrEnum):
    """Semantic role of a series, mapped to theme colors by the front end."""

    SERIES = "series"
    BENCHMARK = "benchmark"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    BAND = "band"
    REFERENCE = "reference"


class Axis(BaseModel):
    """Axis description."""

    model_config = OUTPUT_CONFIG

    type: AxisType = AxisType.VALUE
    name: str = ""
    format: MetricFormat | None = None
    categories: list[str] | None = None
    min: float | None = None
    max: float | None = None


class ChartSeries(BaseModel):
    """One data series.

    ``data`` rows are ``[x, y]`` for line/area/bar/scatter (x may be an ISO timestamp or a
    category), ``[x, y, value]`` for heatmaps, ``[min, q1, median, q3, max]`` for box plots
    and a list of values for radar charts.

    Attributes:
        name: Legend name.
        data: Data rows.
        role: Semantic role (drives colors and dashes).
        run_id: Run this series belongs to (keeps colors consistent across charts).
        render_as: Override the chart type for this series (e.g. ``scatter`` markers on a line).
        stack: Stack group.
        y_axis: Index of the y axis.
        dashed: Draw dashed.
        symbol_size: Marker size for scatter series.
    """

    model_config = OUTPUT_CONFIG

    name: str
    data: list[list[Any]] = Field(default_factory=list)
    role: SeriesRole = SeriesRole.SERIES
    run_id: str | None = None
    render_as: ChartType | None = None
    stack: str | None = None
    y_axis: int = 0
    dashed: bool = False
    symbol_size: float | None = None
    labels: list[str] | None = None


class AnnotationKind(StrEnum):
    """Annotation kinds."""

    BAND = "band"
    VLINE = "vline"
    HLINE = "hline"


class Annotation(BaseModel):
    """Shaded band or reference line."""

    model_config = OUTPUT_CONFIG

    kind: AnnotationKind
    start: Any = None
    end: Any = None
    value: Any = None
    label: str = ""
    role: SeriesRole = SeriesRole.NEGATIVE


class ChartSpec(BaseModel):
    """Library-neutral chart description.

    Attributes:
        id: Chart plugin name.
        title: Title.
        type: Default chart type for all series.
        x_axis: X axis.
        y_axes: One or more y axes.
        series: Series.
        annotations: Bands and reference lines.
        time_series: Whether the chart participates in synchronized zoom and brushing.
        options: Type-specific options (e.g. radar ``indicators``, heatmap ``categories``).
    """

    model_config = OUTPUT_CONFIG

    id: str
    title: str
    type: ChartType
    x_axis: Axis = Field(default_factory=Axis)
    y_axes: list[Axis] = Field(default_factory=lambda: [Axis()])
    series: list[ChartSeries] = Field(default_factory=list)
    annotations: list[Annotation] = Field(default_factory=list)
    time_series: bool = False
    options: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

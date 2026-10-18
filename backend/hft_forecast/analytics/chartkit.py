"""Helpers shared by chart builders (not a plugin module)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from hft_forecast.core.interfaces import ChartInput
from hft_forecast.core.specs import Axis, AxisType, ChartSeries, MetricFormat, SeriesRole
from hft_forecast.core.types import FloatArray, TimeArray


def iso(timestamps: TimeArray) -> list[str]:
    """ISO-8601 strings (UTC) for a timestamp array."""
    return [f"{t}Z" for t in np.datetime_as_string(timestamps.astype("datetime64[s]"))]


def clean(values: FloatArray) -> list[float | None]:
    """Floats with NaN/inf replaced by ``None``."""
    return [float(v) if np.isfinite(v) else None for v in np.asarray(values, dtype=np.float64)]


def time_series(
    name: str,
    timestamps: TimeArray,
    values: FloatArray,
    *,
    role: SeriesRole = SeriesRole.SERIES,
    run_id: str | None = None,
    stack: str | None = None,
    dashed: bool = False,
) -> ChartSeries:
    """A ``[timestamp, value]`` series."""
    xs = iso(timestamps)
    ys = clean(values)
    return ChartSeries(
        name=name,
        data=[[x, y] for x, y in zip(xs, ys, strict=True)],
        role=role,
        run_id=run_id,
        stack=stack,
        dashed=dashed or role is SeriesRole.BENCHMARK,
    )


def time_axis() -> Axis:
    """Time x axis."""
    return Axis(type=AxisType.TIME)


def value_axis(name: str = "", fmt: MetricFormat | None = None, log: bool = False) -> Axis:
    """Value (or log) y axis."""
    return Axis(type=AxisType.LOG if log else AxisType.VALUE, name=name, format=fmt)


def benchmark_series(
    run: ChartInput, values: FloatArray, name: str | None = None
) -> ChartSeries | None:
    """Benchmark series for a run (or ``None``)."""
    if run.result.benchmark_returns is None:
        return None
    label = name or f"Benchmark ({run.result.benchmark_id})"
    return time_series(label, run.result.timestamps, values, role=SeriesRole.BENCHMARK)


def wealth(returns: FloatArray) -> FloatArray:
    """Growth of 1."""
    return np.cumprod(1.0 + np.nan_to_num(returns, nan=0.0))


def label_rows(runs: Sequence[ChartInput]) -> list[str]:
    """Run labels."""
    return [r.label for r in runs]


def common_window(runs: Sequence[ChartInput]) -> tuple[Any, Any] | None:
    """Overlapping timestamp range of several runs (``None`` if disjoint)."""
    starts = [r.result.timestamps[0] for r in runs if r.result.n_bars]
    ends = [r.result.timestamps[-1] for r in runs if r.result.n_bars]
    if not starts:
        return None
    lo, hi = max(starts), min(ends)
    return (lo, hi) if lo <= hi else None

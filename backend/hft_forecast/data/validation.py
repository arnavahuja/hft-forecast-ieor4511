"""Data quality checks run on ingest.

Checks: duplicate rows, non-monotonic timestamps, negative or zero prices, gaps versus the
trading calendar (daily data), and extreme jumps. The result is a :class:`QualityReport`
shown on the UI's Data page.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Final

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from hft_forecast.core import columns as C
from hft_forecast.core.calendar import TradingCalendar
from hft_forecast.core.types import Frequency, MarketData

EXTREME_JUMP_THRESHOLD: Final = 0.5
"""Absolute close-to-close return above which a bar is flagged as an extreme jump."""
MAX_EXAMPLES: Final = 5
GAP_WARNING_FRACTION: Final = 0.02
"""Fraction of missing sessions above which a gap issue becomes a warning."""


class Severity(StrEnum):
    """Issue severity."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class QualityIssue(BaseModel):
    """One data quality finding."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    check: str
    severity: Severity
    message: str
    instrument: str | None = None
    count: int = 0
    examples: list[str] = Field(default_factory=list)


class QualityReport(BaseModel):
    """Data quality report for a dataset."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    rows: int
    instruments: int
    start: str | None
    end: str | None
    issues: list[QualityIssue] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no error-level issue was found."""
        return not any(i.severity is Severity.ERROR for i in self.issues)

    @property
    def summary(self) -> dict[str, int]:
        """Issue counts per severity."""
        out = {s.value: 0 for s in Severity}
        for issue in self.issues:
            out[issue.severity.value] += 1
        return out


def _fmt_ts(values: list[object]) -> list[str]:
    return [str(v) for v in values[:MAX_EXAMPLES]]


def check_duplicates(frame: pl.DataFrame) -> list[QualityIssue]:
    """Rows sharing ``(timestamp, instrument_id)``."""
    dup = frame.group_by(list(C.KEY_COLUMNS)).len().filter(pl.col("len") > 1)
    if dup.height == 0:
        return []
    return [
        QualityIssue(
            check="duplicates",
            severity=Severity.ERROR,
            message=f"{dup.height} duplicate (timestamp, instrument) keys",
            count=dup.height,
            examples=_fmt_ts(dup.get_column(C.TIMESTAMP).to_list()),
        )
    ]


def check_monotonic(frame: pl.DataFrame) -> list[QualityIssue]:
    """Timestamps that decrease within an instrument in the original row order."""
    bad = (
        frame.with_row_index("_row")
        .with_columns(
            (pl.col(C.TIMESTAMP).diff().over(C.INSTRUMENT, order_by="_row") < pl.duration())
            .fill_null(False)
            .alias("_back")
        )
        .filter(pl.col("_back"))
    )
    if bad.height == 0:
        return []
    return [
        QualityIssue(
            check="monotonic",
            severity=Severity.WARNING,
            message=f"{bad.height} rows out of time order in the source (sorted on ingest)",
            count=bad.height,
            examples=_fmt_ts(bad.get_column(C.TIMESTAMP).to_list()),
        )
    ]


def check_prices(frame: pl.DataFrame) -> list[QualityIssue]:
    """Non-positive prices."""
    issues: list[QualityIssue] = []
    for col in C.PRICE_FIELDS:
        if col not in frame.columns:
            continue
        bad = frame.filter(pl.col(col) <= 0)
        if bad.height:
            issues.append(
                QualityIssue(
                    check="non_positive_price",
                    severity=Severity.ERROR,
                    message=f"{bad.height} non-positive values in '{col}'",
                    count=bad.height,
                    examples=_fmt_ts(bad.get_column(C.TIMESTAMP).to_list()),
                )
            )
    if C.VOLUME in frame.columns:
        neg = frame.filter(pl.col(C.VOLUME) < 0)
        if neg.height:
            issues.append(
                QualityIssue(
                    check="negative_volume",
                    severity=Severity.ERROR,
                    message=f"{neg.height} negative volumes",
                    count=neg.height,
                )
            )
    return issues


def check_jumps(data: MarketData, threshold: float = EXTREME_JUMP_THRESHOLD) -> list[QualityIssue]:
    """Close-to-close moves larger than ``threshold`` in absolute value."""
    if not data.has_field(C.CLOSE) or len(data.timestamps) < 2:
        return []
    close = data.panel(C.CLOSE)
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = close[1:] / close[:-1] - 1.0
    issues: list[QualityIssue] = []
    for j, inst in enumerate(data.instruments):
        idx = np.flatnonzero(np.abs(np.nan_to_num(rets[:, j])) > threshold)
        if idx.size:
            issues.append(
                QualityIssue(
                    check="extreme_jump",
                    severity=Severity.WARNING,
                    instrument=inst,
                    message=f"{idx.size} close-to-close moves above {threshold:.0%}",
                    count=int(idx.size),
                    examples=[str(data.timestamps[i + 1])[:10] for i in idx[:MAX_EXAMPLES]],
                )
            )
    return issues


def check_calendar_gaps(data: MarketData, calendar: TradingCalendar) -> list[QualityIssue]:
    """Missing sessions versus the exchange calendar (daily data only)."""
    if data.frequency is not Frequency.D1 or data.is_empty():
        return []
    issues: list[QualityIssue] = []
    per_inst = data.frame.group_by(C.INSTRUMENT).agg(pl.col(C.TIMESTAMP).dt.date().alias("dates"))
    for inst, dates in per_inst.iter_rows():
        present: set[date] = set(dates)
        sessions = calendar.sessions(min(present), max(present))
        missing = [s for s in sessions if s not in present]
        if not missing:
            continue
        frac = len(missing) / max(len(sessions), 1)
        issues.append(
            QualityIssue(
                check="calendar_gap",
                severity=Severity.WARNING if frac > GAP_WARNING_FRACTION else Severity.INFO,
                instrument=inst,
                message=f"{len(missing)} missing sessions ({frac:.1%}) vs {calendar.name}",
                count=len(missing),
                examples=[d.isoformat() for d in missing[:MAX_EXAMPLES]],
            )
        )
    return issues


def validate(
    data: MarketData, calendar: TradingCalendar | None = None, raw: pl.DataFrame | None = None
) -> QualityReport:
    """Run all checks.

    Args:
        data: Canonical market data.
        calendar: Calendar for gap checks (skipped if ``None``).
        raw: The frame in original row order, for the monotonicity check.

    Returns:
        The quality report.
    """
    frame = data.frame
    issues = [
        *check_duplicates(raw if raw is not None else frame),
        *check_monotonic(raw if raw is not None else frame),
        *check_prices(frame),
        *check_jumps(data),
    ]
    if calendar is not None:
        issues.extend(check_calendar_gaps(data, calendar))
    ts = data.timestamps
    return QualityReport(
        rows=frame.height,
        instruments=len(data.instruments),
        start=str(ts[0]) if len(ts) else None,
        end=str(ts[-1]) if len(ts) else None,
        issues=issues,
    )

"""Data quality checks run on ingest.

Checks: duplicate rows, non-monotonic timestamps, negative or zero prices, gaps versus the
trading calendar (daily data), and extreme jumps. Intraday data gets a second set: session
coverage, missing sessions, stale runs and halts, spread outliers, the bar-end labelling
convention, session bounds and mid consistency. The result is a :class:`QualityReport` shown
on the UI's Data page.
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
from hft_forecast.core.types import BAR_LABEL_KEY, BoolArray, Frequency, MarketData

EXTREME_JUMP_THRESHOLD: Final = 0.5
"""Absolute close-to-close return above which a bar is flagged as an extreme jump."""
MAX_EXAMPLES: Final = 5
GAP_WARNING_FRACTION: Final = 0.02
"""Fraction of missing sessions above which a gap issue becomes a warning."""
MIN_SESSION_COVERAGE: Final = 0.98
"""Share of a session's expected bars below which coverage becomes a warning."""
STALE_WARNING_FRACTION: Final = 0.05
"""Share of stale bars above which staleness becomes a warning."""
HALT_SECONDS: Final = 300
"""A stale run at least this long is reported separately as a probable halt."""
SPREAD_OUTLIER_QUANTILE: Final = 0.999
MID_TOLERANCE: Final = 1e-9
QUOTE_FILTER_COUNTS_KEY: Final = "quote_filter_counts"
"""MarketData metadata key: ``filter name -> quotes dropped`` reported by a quote source."""


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


def check_bar_label(data: MarketData) -> list[QualityIssue]:
    """Intraday datasets must be right-closed and labelled at the bar end."""
    label = data.metadata.get(BAR_LABEL_KEY)
    if label == "end":
        return []
    return [
        QualityIssue(
            check="bar_label",
            severity=Severity.ERROR,
            message=(
                f"metadata['{BAR_LABEL_KEY}'] is {label!r}, expected 'end': every intraday bar "
                "must summarize (t - step, t]"
            ),
        )
    ]


def check_mid_consistency(data: MarketData) -> list[QualityIssue]:
    """``mid`` is the average of the quotes, and ``close`` is the mid on quote datasets."""
    issues: list[QualityIssue] = []
    if data.has_field(C.MID) and data.has_field(C.BID) and data.has_field(C.ASK):
        off = np.abs(data.panel(C.MID) - (data.panel(C.BID) + data.panel(C.ASK)) / 2.0)
        bad = int(np.nansum(off > MID_TOLERANCE))
        if bad:
            issues.append(
                QualityIssue(
                    check="mid_consistency",
                    severity=Severity.ERROR,
                    message=f"{bad} bars where mid != (bid + ask) / 2",
                    count=bad,
                )
            )
    if data.metadata.get(C.PRICE_BASIS_KEY) == "mid" and data.has_field(C.MID):
        off = np.abs(data.panel(C.CLOSE) - data.panel(C.MID))
        bad = int(np.nansum(off > MID_TOLERANCE))
        if bad:
            issues.append(
                QualityIssue(
                    check="mid_consistency",
                    severity=Severity.ERROR,
                    message=f"{bad} bars where close != mid although price_basis is 'mid'",
                    count=bad,
                )
            )
    return issues


def check_session_coverage(data: MarketData, calendar: TradingCalendar) -> list[QualityIssue]:
    """Bars present versus bars expected per session, and sessions with no rows at all."""
    step = data.frequency.seconds
    if step is None or data.is_empty():
        return []
    stamps = data.timestamps
    sessions = calendar.sessions(
        stamps.min().astype("datetime64[D]").astype(date),
        stamps.max().astype("datetime64[D]").astype(date),
    )
    ids = calendar.session_ids(stamps)
    present = {int(i) for i in np.unique(ids) if i >= 0}
    issues: list[QualityIssue] = []
    outside = int((ids < 0).sum())
    if outside:
        issues.append(
            QualityIssue(
                check="session_bounds",
                severity=Severity.ERROR,
                message=f"{outside} bars outside any regular session of {calendar.name}",
                count=outside,
                examples=[str(t) for t in stamps[ids < 0][:MAX_EXAMPLES]],
            )
        )
    missing, thin, early = [], [], []
    for day in sessions:
        key = int(np.datetime64(day, "D").astype(np.int64))
        if key not in present:
            missing.append(day)
            continue
        open_utc, close_utc = calendar.session_bounds(day)
        expected = int((close_utc - open_utc).astype("timedelta64[s]").astype(np.int64)) // step
        counted = int((ids == key).sum()) / max(len(data.instruments), 1)
        if expected and counted / expected < MIN_SESSION_COVERAGE:
            thin.append(f"{day}: {counted:.0f}/{expected}")
        if str(close_utc)[11:16] != _regular_close_utc(calendar, sessions):
            early.append(day)
    if missing:
        issues.append(
            QualityIssue(
                check="missing_sessions",
                severity=Severity.WARNING,
                message=f"{len(missing)} calendar sessions with no rows",
                count=len(missing),
                examples=[d.isoformat() for d in missing[:MAX_EXAMPLES]],
            )
        )
    if thin:
        issues.append(
            QualityIssue(
                check="session_coverage",
                severity=Severity.WARNING,
                message=f"{len(thin)} sessions below {MIN_SESSION_COVERAGE:.0%} of expected bars",
                count=len(thin),
                examples=thin[:MAX_EXAMPLES],
            )
        )
    if early:
        issues.append(
            QualityIssue(
                check="early_close",
                severity=Severity.INFO,
                message=f"{len(early)} sessions close early",
                count=len(early),
                examples=[d.isoformat() for d in early[:MAX_EXAMPLES]],
            )
        )
    return issues


def _regular_close_utc(calendar: TradingCalendar, sessions: list[date]) -> str:
    """UTC HH:MM of the most common close in ``sessions`` (DST-correct, early closes aside)."""
    closes = [str(calendar.session_bounds(d)[1])[11:16] for d in sessions]
    return max(set(closes), key=closes.count) if closes else ""


def check_staleness(data: MarketData, calendar: TradingCalendar) -> list[QualityIssue]:
    """Share of bars with no quote update, and stale runs long enough to be halts."""
    step = data.frequency.seconds
    if step is None or not data.has_field(C.IS_STALE) or data.is_empty():
        return []
    stale = data.panel(C.IS_STALE) == 1.0
    halt_bars = max(HALT_SECONDS // step, 1)
    issues: list[QualityIssue] = []
    for j, inst in enumerate(data.instruments):
        column = stale[:, j]
        share = float(column.mean())
        longest, halts = _longest_run(column, halt_bars)
        if share:
            issues.append(
                QualityIssue(
                    check="stale_run",
                    severity=Severity.WARNING if share > STALE_WARNING_FRACTION else Severity.INFO,
                    instrument=inst,
                    message=(
                        f"{share:.1%} of bars carry a stale mid; longest run {longest} bars "
                        f"({longest * step}s)"
                    ),
                    count=int(column.sum()),
                )
            )
        if halts:
            issues.append(
                QualityIssue(
                    check="probable_halt",
                    severity=Severity.WARNING,
                    instrument=inst,
                    message=f"{len(halts)} stale runs of at least {HALT_SECONDS}s",
                    count=len(halts),
                    examples=[str(data.timestamps[i]) for i in halts[:MAX_EXAMPLES]],
                )
            )
    return issues


def _longest_run(flags: BoolArray, min_run: int) -> tuple[int, list[int]]:
    """Longest run of ``True`` and the start index of every run of at least ``min_run``."""
    padded = np.concatenate(([False], flags, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts, ends = edges[0::2], edges[1::2]
    if not len(starts):
        return 0, []
    lengths = ends - starts
    return int(lengths.max()), [int(s) for s in starts[lengths >= min_run]]


def check_spread_outliers(data: MarketData) -> list[QualityIssue]:
    """Bars above an instrument's 99.9th percentile spread.

    The count is ~0.1% of bars by construction; the examples are the point, since a quote
    that survived the source's filters but sits far above the tail is usually a bad print.
    """
    if not data.has_field(C.SPREAD_BPS) or data.is_empty():
        return []
    panel = data.panel(C.SPREAD_BPS)
    issues: list[QualityIssue] = []
    for j, inst in enumerate(data.instruments):
        column = panel[:, j]
        finite = column[np.isfinite(column)]
        if not len(finite):
            continue
        cutoff = float(np.quantile(finite, SPREAD_OUTLIER_QUANTILE))
        idx = np.flatnonzero(np.isfinite(column) & (column > cutoff))
        if idx.size:
            issues.append(
                QualityIssue(
                    check="spread_outliers",
                    severity=Severity.WARNING,
                    instrument=inst,
                    message=(
                        f"{idx.size} bars above the {SPREAD_OUTLIER_QUANTILE:.1%} spread of "
                        f"{cutoff:.1f} bps (max {column[idx].max():.1f} bps)"
                    ),
                    count=int(idx.size),
                    examples=[str(data.timestamps[i]) for i in idx[:MAX_EXAMPLES]],
                )
            )
    return issues


def check_quote_filters(data: MarketData) -> list[QualityIssue]:
    """Quotes the source dropped, reported per filter (e.g. locked and crossed markets)."""
    counts = data.metadata.get(QUOTE_FILTER_COUNTS_KEY)
    if not isinstance(counts, dict) or not counts:
        return []
    total = sum(int(v) for v in counts.values())
    return [
        QualityIssue(
            check="quote_filters",
            severity=Severity.INFO,
            message=f"{total} quotes dropped before aggregation",
            count=total,
            examples=[f"{k}: {v}" for k, v in sorted(counts.items())[:MAX_EXAMPLES]],
        )
    ]


def check_intraday(data: MarketData, calendar: TradingCalendar) -> list[QualityIssue]:
    """Every intraday-only check (§3.6 of the design document)."""
    return [
        *check_bar_label(data),
        *check_mid_consistency(data),
        *check_session_coverage(data, calendar),
        *check_staleness(data, calendar),
        *check_spread_outliers(data),
        *check_quote_filters(data),
    ]


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
        if data.frequency.is_intraday:
            issues.extend(check_intraday(data, calendar))
        else:
            issues.extend(check_calendar_gaps(data, calendar))
    ts = data.timestamps
    return QualityReport(
        rows=frame.height,
        instruments=len(data.instruments),
        start=str(ts[0]) if len(ts) else None,
        end=str(ts[-1]) if len(ts) else None,
        issues=issues,
    )

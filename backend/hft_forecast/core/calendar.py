"""Trading calendars and annualization.

Annualization factors are derived from the exchange calendar and the bar frequency, never
hard-coded. ``exchange_calendars`` supplies sessions and holidays.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Final

import numpy as np
import pandas as pd

from hft_forecast.core.errors import ConfigError
from hft_forecast.core.types import Frequency, MarketData, TimeArray

_CALENDAR_SAMPLE_START: Final = date(2010, 1, 1)
_CALENDAR_SAMPLE_END: Final = date(2019, 12, 31)
_MINUTES_PER_HOUR: Final = 60
WEEKS_PER_YEAR: Final = 52.0
MONTHS_PER_YEAR: Final = 12.0
DAYS_PER_YEAR: Final = 365.25
SECONDS_PER_DAY: Final = 86_400.0


@lru_cache(maxsize=16)
def _load(name: str) -> object:
    import exchange_calendars as xcals

    try:
        return xcals.get_calendar(name)
    except Exception as exc:
        raise ConfigError(f"Unknown exchange calendar '{name}'") from exc


class TradingCalendar:
    """Wrapper around an ``exchange_calendars`` calendar.

    Args:
        name: Calendar code, e.g. ``"XNYS"``.
    """

    def __init__(self, name: str = "XNYS") -> None:
        self.name = name

    @property
    def _cal(self) -> object:
        return _load(self.name)

    def sessions(self, start: date, end: date) -> list[date]:
        """Trading session dates in ``[start, end]`` (clipped to the calendar bounds)."""
        cal = self._cal
        first = max(pd.Timestamp(start), cal.first_session)  # type: ignore[attr-defined]
        last = min(pd.Timestamp(end), cal.last_session)  # type: ignore[attr-defined]
        if first > last:
            return []
        idx = cal.sessions_in_range(first, last)  # type: ignore[attr-defined]
        return [ts.date() for ts in idx]

    def sessions_per_year(self) -> float:
        """Average sessions per year over a fixed reference decade."""
        sessions = self.sessions(_CALENDAR_SAMPLE_START, _CALENDAR_SAMPLE_END)
        years = _CALENDAR_SAMPLE_END.year - _CALENDAR_SAMPLE_START.year + 1
        return len(sessions) / years

    def session_minutes(self) -> float:
        """Typical regular-session length in minutes (median over the reference period)."""
        cal = self._cal
        sessions = cal.sessions_in_range(  # type: ignore[attr-defined]
            pd.Timestamp(_CALENDAR_SAMPLE_START), pd.Timestamp(_CALENDAR_SAMPLE_END)
        )
        opens = cal.schedule.loc[sessions, "open"]  # type: ignore[attr-defined]
        closes = cal.schedule.loc[sessions, "close"]  # type: ignore[attr-defined]
        return float(((closes - opens).dt.total_seconds() / _MINUTES_PER_HOUR).median())

    def periods_per_year(self, frequency: Frequency) -> float:
        """Number of bars per year for a frequency on this calendar."""
        if frequency is Frequency.W1:
            return WEEKS_PER_YEAR
        if frequency is Frequency.MO1:
            return MONTHS_PER_YEAR
        sessions = self.sessions_per_year()
        if frequency is Frequency.D1:
            return sessions
        minutes = frequency.minutes
        if minutes is None:  # pragma: no cover - exhaustive above
            raise ConfigError(f"Unsupported frequency {frequency}")
        return sessions * np.ceil(self.session_minutes() / minutes)


def infer_periods_per_year(timestamps: TimeArray) -> float | None:
    """Estimate bars per year from observed timestamps (``None`` if too few)."""
    min_obs = 3
    if len(timestamps) < min_obs:
        return None
    span_days = (
        float((timestamps[-1] - timestamps[0]).astype("timedelta64[s]").astype(np.int64))
        / SECONDS_PER_DAY
    )
    if span_days <= 0:
        return None
    return (len(timestamps) - 1) / (span_days / DAYS_PER_YEAR)


PERIODS_PER_YEAR_KEY: Final = "periods_per_year"
"""MarketData metadata key under which the runner stores the annualization factor."""


def periods_per_year_for(data: MarketData, calendar: str = "XNYS") -> float:
    """Annualization factor for market data.

    Uses the value the runner stored in the metadata, else derives it from the data
    frequency and the exchange calendar. Never inferred from the data's own timestamps, so
    the value does not change when data is truncated (lookahead-safe).
    """
    value = data.metadata.get(PERIODS_PER_YEAR_KEY)
    if isinstance(value, int | float) and value > 0:
        return float(value)
    return TradingCalendar(calendar).periods_per_year(data.frequency)

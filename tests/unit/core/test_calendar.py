"""Session bounds and session ids: early closes, DST, and sub-minute annualization."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from hft_forecast.core.calendar import TradingCalendar
from hft_forecast.core.errors import ConfigError
from hft_forecast.core.types import Frequency

CAL = TradingCalendar("XNYS")
BARS_PER_SESSION_5S = 390 * 12


def test_periods_per_year_scales_with_bar_seconds():
    sessions = CAL.sessions_per_year()
    assert CAL.periods_per_year(Frequency.D1) == sessions
    assert CAL.periods_per_year(Frequency.MIN1) == pytest.approx(sessions * 390)
    assert CAL.periods_per_year(Frequency.S5) == pytest.approx(sessions * BARS_PER_SESSION_5S)
    assert CAL.periods_per_year(Frequency.S5) == 12 * CAL.periods_per_year(Frequency.MIN1)


@pytest.mark.parametrize(
    ("day", "open_utc", "close_utc"),
    [
        # EST: 09:30-16:00 New York is 14:30-21:00 UTC
        (date(2024, 11, 15), "2024-11-15T14:30", "2024-11-15T21:00"),
        # EDT: the same local session is an hour earlier in UTC
        (date(2024, 3, 15), "2024-03-15T13:30", "2024-03-15T20:00"),
        # the day after Thanksgiving closes at 13:00 local
        (date(2024, 11, 29), "2024-11-29T14:30", "2024-11-29T18:00"),
    ],
)
def test_session_bounds_handle_dst_and_early_closes(day, open_utc, close_utc):
    assert CAL.session_bounds(day) == (
        np.datetime64(open_utc, "us"),
        np.datetime64(close_utc, "us"),
    )


def test_session_bounds_rejects_a_non_session():
    with pytest.raises(ConfigError):
        CAL.session_bounds(date(2024, 12, 25))


def test_session_ids_group_by_local_session_not_utc_date():
    stamps = np.array(
        [
            "2024-11-29T14:30",  # open of the early-close session
            "2024-11-29T18:00",  # its close
            "2024-11-29T18:00:01",  # after the close
            "2024-12-02T15:00",  # next session
            "2024-11-30T15:00",  # a Saturday
        ],
        dtype="datetime64[us]",
    )
    ids = CAL.session_ids(stamps)
    assert ids[0] == ids[1]
    assert ids[2] == -1 and ids[4] == -1
    assert ids[3] != ids[0]
    assert np.array(ids[[0, 3]], dtype="datetime64[D]").tolist() == [
        date(2024, 11, 29),
        date(2024, 12, 2),
    ]


def test_session_ids_is_order_insensitive_and_handles_an_empty_input():
    stamps = np.array(
        ["2024-12-02T15:00", "2024-11-29T15:00", "2024-12-02T16:00"], dtype="datetime64[us]"
    )
    ids = CAL.session_ids(stamps)
    assert ids[0] == ids[2] and ids[1] != ids[0]
    assert len(CAL.session_ids(np.array([], dtype="datetime64[us]"))) == 0

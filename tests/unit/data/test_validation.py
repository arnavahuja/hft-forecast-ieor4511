from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from hft_forecast.core import columns as C
from hft_forecast.core.calendar import TradingCalendar
from hft_forecast.core.types import MarketData
from hft_forecast.data.validation import validate


def test_gap_and_jump_detection():
    days = [2, 3, 4, 8, 9]  # 2024-01-05 missing
    frame = pl.DataFrame(
        {
            C.TIMESTAMP: [datetime(2024, 1, d, tzinfo=UTC) for d in days],
            C.INSTRUMENT: ["A"] * 5,
            C.CLOSE: [10.0, 10.1, 30.0, 30.1, 30.2],
        }
    )
    report = validate(MarketData(frame), TradingCalendar())
    by_check = {i.check: i for i in report.issues}
    assert by_check["calendar_gap"].examples == ["2024-01-05"]
    assert by_check["extreme_jump"].count == 1
    assert report.ok

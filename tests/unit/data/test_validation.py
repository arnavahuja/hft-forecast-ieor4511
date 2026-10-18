from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import polars as pl

from hft_forecast.core import columns as C
from hft_forecast.core.calendar import TradingCalendar
from hft_forecast.core.types import BAR_LABEL_KEY, DataRequest, Frequency, MarketData
from hft_forecast.data.sources.synthetic import SyntheticSource
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


def _quotes(frequency: Frequency = Frequency.S5) -> MarketData:
    request = DataRequest(
        source="synthetic",
        dataset="quotes",
        instruments=("AAA",),
        start=date(2024, 11, 26),
        end=date(2024, 12, 2),
        frequency=frequency,
    )
    return SyntheticSource().fetch(request)


def test_clean_quote_bars_pass_every_intraday_check():
    report = validate(_quotes(), TradingCalendar())
    checks = {i.check for i in report.issues}
    assert report.ok, [i.message for i in report.issues if i.severity == "error"]
    # the generator injects one halt and some stale bars, and 2024-11-29 closes early
    assert {"stale_run", "probable_halt", "early_close"} <= checks
    assert "missing_sessions" not in checks and "session_coverage" not in checks


def test_bar_label_and_mid_consistency_are_errors():
    data = _quotes()
    unlabelled = MarketData(
        data.frame, data.frequency, data.instrument_meta, {**data.metadata, BAR_LABEL_KEY: "start"}
    )
    report = validate(unlabelled, TradingCalendar())
    assert not report.ok
    assert any(i.check == "bar_label" for i in report.issues)

    broken = MarketData(
        data.frame.with_columns(pl.col(C.MID) * 1.01),
        data.frequency,
        data.instrument_meta,
        data.metadata,
    )
    issues = [i for i in validate(broken, TradingCalendar()).issues if i.check == "mid_consistency"]
    assert len(issues) == 2  # mid != (bid + ask) / 2, and close != mid


def test_bars_outside_a_session_are_an_error():
    data = _quotes()
    shifted = data.frame.with_columns(pl.col(C.TIMESTAMP) + timedelta(hours=6))
    report = validate(
        MarketData(shifted, data.frequency, data.instrument_meta, data.metadata), TradingCalendar()
    )
    assert not report.ok
    assert any(i.check == "session_bounds" for i in report.issues)


def test_a_dropped_session_is_reported():
    data = _quotes()
    without = data.frame.filter(pl.col(C.TIMESTAMP).dt.date() != date(2024, 11, 27))
    report = validate(
        MarketData(without, data.frequency, data.instrument_meta, data.metadata), TradingCalendar()
    )
    assert [i.examples for i in report.issues if i.check == "missing_sessions"] == [["2024-11-27"]]

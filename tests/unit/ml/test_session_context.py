"""The session-context set: the two families that cross a session boundary, and the range.

The shared causality check only sees one session of the fixture, so the overnight and
historical columns are null there and effectively untested by it. These tests give them
several sessions and pin down what each one must equal.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from hft_forecast.core import columns as C
from hft_forecast.ml.base import SECONDS_SINCE_OPEN, SECONDS_TO_CLOSE, SESSION_ID
from hft_forecast.ml.features import FEATURE_SETS, check_feature_lookahead

BUCKET = 60
SESSION = 240


def _set(**params):
    return FEATURE_SETS.get("session_context").create(
        {"long_bars": 2, "bucket_seconds": BUCKET, "hist_sessions": 2, "edge_minutes": 1, **params}
    )


def _frame(mids_per_session: list[list[float]], instrument: str = "AAA") -> pl.DataFrame:
    """One row per bar on a 5 s grid, ``SESSION`` seconds long, with the session columns."""
    rows = []
    start = datetime(2024, 1, 2, 14, 30)
    for day, mids in enumerate(mids_per_session):
        step = SESSION // len(mids)
        for i, mid in enumerate(mids):
            since = (i + 1) * step
            rows.append(
                {
                    C.TIMESTAMP: start + timedelta(days=day, seconds=since),
                    C.INSTRUMENT: instrument,
                    C.MID: mid,
                    C.HIGH: mid,
                    C.LOW: mid,
                    SESSION_ID: day,
                    SECONDS_SINCE_OPEN: float(since),
                    SECONDS_TO_CLOSE: float(SESSION - since),
                }
            )
    return pl.DataFrame(rows)


def test_the_overnight_gap_is_measured_from_the_previous_close():
    # session 0 closes at 110, session 1 opens at 121: a gap of log(121/110)
    frame = _frame([[100.0, 110.0], [121.0, 130.0]])
    out = _set().compute(frame)
    gap = out["f_overnight_gap_bps"]
    assert gap[0] is None and gap[1] is None  # the first session has no previous close
    want = np.log(121.0 / 110.0) * 1e4
    assert gap[2] == pytest.approx(want) and gap[3] == pytest.approx(want)


def test_the_previous_session_return_is_that_session_not_this_one():
    frame = _frame([[100.0, 110.0], [121.0, 130.0]])
    out = _set().compute(frame)
    want = np.log(110.0 / 100.0) * 1e4
    assert out["f_prev_session_logret_bps"][2] == pytest.approx(want)


def test_the_gap_reversion_is_positive_when_the_day_gives_the_gap_back():
    # gaps up (110 -> 121) and then falls back through the session: reverting
    out = _set().compute(_frame([[100.0, 110.0], [121.0, 115.0]]))
    assert out["f_gap_reversion_bps"][3] > 0
    # gaps up and keeps going: not reverting
    out = _set().compute(_frame([[100.0, 110.0], [121.0, 130.0]]))
    assert out["f_gap_reversion_bps"][3] < 0


def test_the_time_of_day_history_excludes_the_current_session():
    """The whole point of the feature: today's own bucket must not be in its own average."""
    # a bucket wider than the session puts every bar of a day in one bucket, so the bucket
    # return is the day's: log(2), log(3), log(4), log(5)
    frame = _frame([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]])
    out = _set(bucket_seconds=SESSION * 10).compute(frame)
    hist = out["f_tod_hist_logret_bps"]
    returns = [np.log(k) * 1e4 for k in (2.0, 3.0, 4.0, 5.0)]
    assert hist[0] is None and hist[1] is None  # session 0 has no history
    # session 1 sees session 0; session 2 sees the mean of 0 and 1 (hist_sessions=2)
    assert hist[2] == pytest.approx(returns[0])
    assert hist[4] == pytest.approx(np.mean(returns[:2]))
    # session 3 sees only the last two, not session 0
    assert hist[6] == pytest.approx(np.mean(returns[1:3]))


def test_the_range_position_is_zero_at_the_low_and_one_at_the_high():
    out = _set().compute(_frame([[10.0, 12.0, 8.0, 11.0]]))
    position = out["f_range_position"].to_list()
    assert position[1] == pytest.approx(1.0)  # 12 is the running high
    assert position[2] == pytest.approx(0.0)  # 8 is the running low
    assert 0.0 < position[3] < 1.0
    drawdown, runup = out["f_drawdown_bps"].to_numpy(), out["f_runup_bps"].to_numpy()
    assert (drawdown <= 1e-9).all() and (runup >= -1e-9).all()


def test_the_set_is_causal_across_several_sessions(quote_bars):
    check_feature_lookahead(_set(long_bars=20, bucket_seconds=1800), quote_bars)


def test_rows_come_back_in_the_order_they_went_in(quote_bars):
    out = _set(long_bars=20, bucket_seconds=1800).compute(quote_bars)
    assert out.height == quote_bars.height
    assert out[C.TIMESTAMP].to_list() == quote_bars[C.TIMESTAMP].to_list()
    assert out[C.INSTRUMENT].to_list() == quote_bars[C.INSTRUMENT].to_list()


def test_the_clock_runs_from_zero_to_one_within_every_session(quote_bars):
    out = _set(long_bars=20).compute(quote_bars)
    phase = out["f_session_phase"]
    assert phase.min() > 0.0 and phase.max() == pytest.approx(1.0)
    assert np.allclose(out["f_tod_sin"].to_numpy() ** 2 + out["f_tod_cos"].to_numpy() ** 2, 1.0)

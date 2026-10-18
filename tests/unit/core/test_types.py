from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import polars as pl
import pytest

from hft_forecast.core import columns as C
from hft_forecast.core.errors import DataError
from hft_forecast.core.types import (
    DataRequest,
    Frequency,
    MarketData,
    TargetFrame,
    TargetKind,
)


def _md() -> MarketData:
    ts = [datetime(2024, 1, d, tzinfo=UTC) for d in (2, 3, 4)]
    frame = pl.DataFrame(
        {
            C.TIMESTAMP: ts * 2,
            C.INSTRUMENT: ["A"] * 3 + ["B"] * 3,
            C.CLOSE: [1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
        }
    ).filter(~((pl.col(C.INSTRUMENT) == "B") & (pl.col(C.CLOSE) == 11.0)))
    return MarketData(frame)


def test_panel_aligns_and_fills_nan():
    md = _md()
    assert md.instruments == ("A", "B")
    panel = md.panel(C.CLOSE)
    assert panel.shape == (3, 2)
    assert np.isnan(panel[1, 1])
    assert panel[2, 1] == 12.0
    with pytest.raises(ValueError):
        panel[0, 0] = 5.0  # read-only


def test_truncate_excludes_future():
    md = _md()
    cut = md.truncate(md.timestamps[1])
    assert len(cut.timestamps) == 2


def test_request_cache_key_is_order_insensitive():
    a = DataRequest(
        source="s",
        dataset="d",
        instruments=("B", "A"),
        start=date(2020, 1, 1),
        end=date(2020, 2, 1),
    )
    b = DataRequest(
        source="s",
        dataset="d",
        instruments=("A", "B", "A"),
        start=date(2020, 1, 1),
        end=date(2020, 2, 1),
    )
    assert a.cache_key() == b.cache_key()
    assert (
        a.key_without_dates() == a.model_copy(update={"end": date(2021, 1, 1)}).key_without_dates()
    )


def test_target_frame_validates_shape_and_reindexes():
    md = _md()
    with pytest.raises(DataError):
        TargetFrame(md.timestamps, ("A",), np.zeros((3, 2)))
    tf = TargetFrame(md.timestamps, ("A", "B"), np.ones((3, 2)), TargetKind.SIGNALS)
    re = tf.reindex(md.timestamps[1:], ("B", "C"))
    assert re.shape == (2, 2)
    assert np.all(re.values[:, 0] == 1.0)
    assert np.all(np.isnan(re.values[:, 1]))
    assert tf.to_long().height == 6


@pytest.mark.parametrize(
    ("frequency", "seconds", "minutes"),
    [
        (Frequency.S1, 1, None),
        (Frequency.S5, 5, None),
        (Frequency.S30, 30, None),
        (Frequency.MIN1, 60, 1),
        (Frequency.MIN15, 900, 15),
        (Frequency.H1, 3600, 60),
        (Frequency.D1, None, None),
        (Frequency.MO1, None, None),
    ],
)
def test_frequency_seconds_and_whole_minutes(frequency, seconds, minutes):
    assert frequency.seconds == seconds
    assert frequency.minutes == minutes  # None whenever the bar is not a whole minute
    assert frequency.is_intraday == (seconds is not None)
    assert frequency.is_subminute == (seconds is not None and seconds < 60)

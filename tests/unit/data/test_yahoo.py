from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from hft_forecast.core import columns as C
from hft_forecast.core.types import Adjustment, DataRequest
from hft_forecast.data.sources.yahoo import YahooSource
from tests.conftest import recorded_yahoo


@pytest.fixture
def yahoo(monkeypatch) -> YahooSource:
    source = YahooSource()
    monkeypatch.setattr(
        source, "_download", lambda tickers, start, end, interval: recorded_yahoo(tickers)
    )
    return source


def _req(adj: Adjustment) -> DataRequest:
    return DataRequest(
        source="yahoo",
        dataset="daily",
        instruments=("AAPL", "SPY"),
        start=date(2020, 6, 1),
        end=date(2020, 9, 29),
        adjustment=adj,
    )


def test_fetch_maps_to_canonical(yahoo):
    data = yahoo.fetch(_req(Adjustment.SPLIT))
    assert data.instruments == ("AAPL", "SPY")
    for col in (C.OPEN, C.HIGH, C.LOW, C.CLOSE, C.VOLUME, C.ADJ_CLOSE, C.DIVIDEND, C.SPLIT):
        assert col in data.fields
    assert str(data.frame.schema[C.TIMESTAMP]) == "Datetime(time_unit='us', time_zone='UTC')"
    assert data.metadata["survivorship_bias_free"] is False


def test_raw_adjustment_undoes_aapl_split(yahoo):
    split = yahoo.fetch(_req(Adjustment.SPLIT)).panel(C.CLOSE)[:, 0]
    raw_md = yahoo.fetch(_req(Adjustment.RAW))
    raw = raw_md.panel(C.CLOSE)[:, 0]
    ts = raw_md.timestamps.astype("datetime64[D]")
    before = ts < np.datetime64("2020-08-31")
    np.testing.assert_allclose(raw[before], split[before] * 4.0)
    np.testing.assert_allclose(raw[~before], split[~before])


def test_total_return_uses_adj_close(yahoo):
    tr = yahoo.fetch(_req(Adjustment.TOTAL_RETURN))
    np.testing.assert_allclose(tr.panel(C.CLOSE), tr.panel(C.ADJ_CLOSE), rtol=1e-6)


def test_requires_tickers():
    from hft_forecast.core.errors import DataError

    with pytest.raises(DataError):
        YahooSource().fetch(
            DataRequest(
                source="yahoo", dataset="daily", start=date(2020, 1, 1), end=date(2020, 2, 1)
            )
        )


@pytest.mark.live
def test_live_yahoo_download():
    data = YahooSource().fetch(
        DataRequest(
            source="yahoo",
            dataset="daily",
            instruments=("SPY",),
            start=date(2023, 1, 1),
            end=date(2023, 3, 1),
        )
    )
    assert len(data.timestamps) > 30

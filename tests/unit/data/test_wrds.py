from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from hft_forecast.core import columns as C
from hft_forecast.core.interfaces import SourceEnvironment
from hft_forecast.core.types import Adjustment, DataRequest
from hft_forecast.data.sources.wrds_crsp import UNIVERSE_FIELD, WrdsSource
from tests.fixtures.fake_wrds import DELISTED, JOINS_LATE, FakeWrds


@pytest.fixture
def fake() -> FakeWrds:
    return FakeWrds()


@pytest.fixture
def source(fake, monkeypatch, tmp_path: Path) -> WrdsSource:
    src = WrdsSource(env=SourceEnvironment(tmp_path, tmp_path, wrds_username="tester"))
    monkeypatch.setattr(src, "_query", fake.query)
    return src


def _req(dataset="crsp_daily", **kw) -> DataRequest:
    base = {
        "source": "wrds",
        "dataset": dataset,
        "start": date(2018, 1, 1),
        "end": date(2020, 12, 31),
    }
    return DataRequest(**{**base, **kw})


def test_unavailable_without_credentials(tmp_path, monkeypatch):
    src = WrdsSource(env=SourceEnvironment(tmp_path, tmp_path, wrds_username=None))
    ok, reason = src.available()
    assert not ok and "WRDS_USERNAME" in reason
    monkeypatch.setenv("PGPASSFILE", str(tmp_path / "missing"))
    src = WrdsSource(env=SourceEnvironment(tmp_path, tmp_path, wrds_username="x"))
    ok, reason = src.available()
    assert not ok and "pgpass" in reason


def test_crsp_tickers_map_to_permnos_and_prices_are_positive(source):
    data = source.fetch(_req(instruments=("AAA", "BBB")))
    assert data.instruments == ("10001", "10002")
    assert data.instrument_meta["10001"].symbol == "AAA"
    assert np.nanmin(data.panel(C.CLOSE)) > 0  # negative CRSP midpoints made positive
    raw = source.fetch(_req(instruments=("AAA",), adjustment=Adjustment.RAW))
    np.testing.assert_allclose(raw.panel(C.CLOSE), data.panel(C.CLOSE)[:, :1] * 2.0)


def test_delisting_return_is_merged(source, fake):
    data = source.fetch(_req(instruments=(str(DELISTED),)))
    ret = data.panel(C.RETURN)[:, 0]
    last = np.flatnonzero(np.isfinite(ret))[-1]
    daily = fake.dsf[fake.dsf.permno == DELISTED].ret.iloc[-1]
    assert ret[last] == pytest.approx((1 + daily) * (1 - 0.30) - 1)
    assert data.metadata["survivorship_bias_free"] is True


def test_sp500_universe_is_point_in_time(source):
    data = source.fetch(_req(universe="sp500"))
    assert len(data.instruments) == 6  # includes the delisted member
    flag = data.panel(UNIVERSE_FIELD)
    ts = data.timestamps.astype("datetime64[D]")
    j = data.instruments.index(str(JOINS_LATE))
    assert flag[ts < np.datetime64("2019-01-02"), j].max() == 0
    assert flag[ts >= np.datetime64("2019-01-02"), j].min() == 1


def test_compustat_uses_availability_date(source):
    data = source.fetch(_req("compustat_annual", instruments=("AAA",)))
    first = data.frame.sort("timestamp").row(0, named=True)
    fiscal = first["fiscal_period_end"]
    lag = (first["timestamp"].date() - date.fromisoformat(fiscal)).days
    assert lag == 180
    assert "book_equity" in data.fields


def test_factors(source):
    data = source.fetch(_req("ff_factors"))
    assert data.instruments == ("FF",)
    assert {"mkt_rf", "smb", "hml", "rmw", "cma", "mom", "rf"} <= set(data.fields)
    assert source.test_connection()["ok"]


@pytest.mark.live
def test_live_wrds_crsp(tmp_path):
    """Opt-in: needs WRDS_USERNAME and ~/.pgpass (pytest -m live)."""
    import os

    src = WrdsSource(
        env=SourceEnvironment(tmp_path, tmp_path, wrds_username=os.environ.get("WRDS_USERNAME"))
    )
    ok, reason = src.available()
    if not ok:
        pytest.skip(reason)
    data = src.fetch(_req(instruments=("AAPL",), start=date(2020, 1, 1), end=date(2020, 3, 1)))
    assert len(data.timestamps) > 30

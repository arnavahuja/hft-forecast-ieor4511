"""Offline WRDS acceptance: CRSP daily with a survivorship-free, point-in-time S&P 500
universe (fake WRDS). This is how the team picks the names it then pulls TAQ quotes for."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hft_forecast.core.config import Settings
from hft_forecast.core.run_config import BacktestConfig
from hft_forecast.data.sources.wrds_crsp import WrdsSource
from hft_forecast.services.container import Services
from tests.fixtures.fake_wrds import DELISTED, JOINS_LATE, FakeWrds


@pytest.fixture
def services(tmp_path: Path, monkeypatch) -> Services:
    fake = FakeWrds()
    monkeypatch.setattr(
        WrdsSource,
        "_query",
        lambda self, sql, params=None, date_cols=None: fake.query(sql, params, date_cols),
    )
    monkeypatch.setattr(WrdsSource, "available", lambda self: (True, ""))
    return Services.create(Settings(data_dir=tmp_path / "data", user_plugins_dir=tmp_path / "up"))


CONFIG = {
    "name": "buy and hold crsp sp500",
    "data": {
        "source": "wrds",
        "dataset": "crsp_daily",
        "instruments": [],
        "universe": {"name": "index_membership", "params": {"index": "sp500"}},
        "start": "2017-06-01",
        "end": "2020-12-31",
        "adjustment": "total_return",
    },
    "strategy": {"name": "buy_and_hold"},
    "costs": [{"name": "bps_notional", "params": {"bps": 5}}],
}


def test_crsp_with_point_in_time_universe_is_survivorship_free(services):
    record = services.execute_run(BacktestConfig.model_validate(CONFIG))
    res = services.store.result(record.id)
    assert res.metadata["return_basis"].startswith("ret field")  # CRSP returns incl. dlret
    assert str(DELISTED) in res.instruments  # the delisted name is still in the panel
    j = res.instruments.index(str(JOINS_LATE))
    before = res.timestamps.astype("datetime64[D]") < np.datetime64("2019-01-02")
    assert np.abs(res.target_weights[before, j]).max() == 0  # not tradable before joining
    d = res.instruments.index(str(DELISTED))
    after = res.timestamps.astype("datetime64[D]") > np.datetime64("2019-07-03")
    assert np.abs(res.weights[after, d]).max() == 0

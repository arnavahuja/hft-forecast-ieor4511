from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hft_forecast.data.service import DataPaths, DataService
from hft_forecast.services.plugins import PluginService

FIXTURES = Path(__file__).parent / "fixtures"


def pytest_configure(config: pytest.Config) -> None:
    """Discover built-in plugins before collection (test modules parametrize over them)."""
    report = PluginService(user_dir=None).load()
    if report.failures:
        raise RuntimeError(f"Plugin import failures: {report.failures}")


@pytest.fixture
def data_service(tmp_path: Path) -> DataService:
    return DataService(DataPaths.under(tmp_path / "data"))


def recorded_yahoo(tickers: list[str]) -> pd.DataFrame:
    """Rebuild a yfinance-shaped (ticker, field) MultiIndex frame from the fixture CSV."""
    long = pd.read_csv(FIXTURES / "yahoo_spy_aapl_2020.csv", parse_dates=["Date"])
    long = long[long["Ticker"].isin(tickers)]
    wide = long.set_index(["Date", "Ticker"]).unstack("Ticker")
    wide.columns = wide.columns.swaplevel(0, 1)
    return wide.sort_index(axis=1)

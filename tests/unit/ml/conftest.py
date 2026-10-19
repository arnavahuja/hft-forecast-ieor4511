"""A small synthetic 5 s quote dataset shared by the ml unit tests."""

from __future__ import annotations

from datetime import date

import pytest

from hft_forecast.core.calendar import TradingCalendar
from hft_forecast.core.run_config import DataSpec
from hft_forecast.core.types import DataRequest, Frequency
from hft_forecast.data.sources.synthetic import SyntheticSource
from hft_forecast.ml.base import H30S
from hft_forecast.ml.config import ExperimentConfig
from hft_forecast.ml.datasets import with_session_columns

START, END = date(2024, 7, 1), date(2024, 7, 3)
INSTRUMENTS = ("AAA", "BBB")


@pytest.fixture(scope="session")
def quote_bars():
    """Three sessions of 5 s synthetic quote bars with session columns attached."""
    data = SyntheticSource().fetch(
        DataRequest(
            source="synthetic",
            dataset="quotes",
            instruments=INSTRUMENTS,
            start=START,
            end=END,
            frequency=Frequency.S5,
        )
    )
    return with_session_columns(data.frame, TradingCalendar())


@pytest.fixture
def experiment() -> ExperimentConfig:
    return ExperimentConfig(
        experiment="h30s",
        horizon=H30S,
        data=DataSpec(
            source="synthetic",
            dataset="quotes",
            instruments=list(INSTRUMENTS),
            start=START,
            end=END,
            frequency=Frequency.S5,
        ),
    )

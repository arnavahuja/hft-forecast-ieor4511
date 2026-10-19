"""Dataset building: session columns, caching by hash and the sequence windows."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from hft_forecast.core.errors import ConfigError
from hft_forecast.core.types import BAR_LABEL_KEY, MarketData
from hft_forecast.ml.base import LABEL_VALID, SECONDS_SINCE_OPEN, SESSION_ID
from hft_forecast.ml.datasets import (
    build_dataset,
    design_matrix,
    load_dataset,
    usable_rows,
    windowed,
)


def test_session_columns_come_from_the_calendar(quote_bars):
    assert (quote_bars[SESSION_ID] >= 0).all()
    assert quote_bars[SESSION_ID].n_unique() == 3
    per_session = quote_bars.group_by(SESSION_ID).agg(pl.col(SECONDS_SINCE_OPEN).min())
    assert (per_session[SECONDS_SINCE_OPEN] == 5).all()  # the first bar ends one step in


def test_bars_must_be_labelled_at_the_bar_end(data_service, experiment, monkeypatch):
    from hft_forecast.ml import datasets

    original = datasets.DataService.fetch

    def fetch(self, request, **kwargs):
        got = original(self, request, **kwargs)
        return MarketData(got.frame, got.frequency, got.instrument_meta, {BAR_LABEL_KEY: "start"})

    monkeypatch.setattr(datasets.DataService, "fetch", fetch)
    with pytest.raises(ConfigError, match="bar end"):
        build_dataset(experiment, data_service)


def test_build_dataset_writes_one_file_per_instrument_and_caches(data_service, experiment):
    directory = build_dataset(experiment, data_service)
    assert sorted(p.stem for p in directory.glob("*.parquet")) == ["AAA", "BBB"]
    written = {p: p.stat().st_mtime_ns for p in directory.glob("*.parquet")}
    assert build_dataset(experiment, data_service) == directory
    assert {p: p.stat().st_mtime_ns for p in directory.glob("*.parquet")} == written

    frame = load_dataset(directory)
    matrix, columns = design_matrix(frame)
    assert matrix.shape == (frame.height, len(columns))
    rows = usable_rows(frame)
    assert 0 < rows.height < frame.height
    assert rows[LABEL_VALID].all()
    assert np.isfinite(design_matrix(rows)[0]).all()


def test_a_different_label_config_builds_a_different_dataset(data_service, experiment):
    other = experiment.model_copy(
        update={"labels": experiment.labels.model_copy(update={"dead_zone_bps": 7.0})}
    )
    assert build_dataset(experiment, data_service) != build_dataset(other, data_service)


def test_windows_never_span_a_session():
    x = np.arange(12, dtype=float).reshape(6, 2)
    sessions = np.array([1, 1, 1, 2, 2, 2])
    out = windowed(x, lookback=2, session_id=sessions)
    assert np.isnan(out[0]).all() and np.isnan(out[3]).all()  # first bar of each session
    assert np.array_equal(out[1], x[0:2])
    assert np.array_equal(out[4], x[3:5])

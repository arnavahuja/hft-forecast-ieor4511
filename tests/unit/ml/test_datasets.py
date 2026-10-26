"""Dataset building: session columns, caching by hash and the sequence windows."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from hft_forecast.core.errors import ConfigError
from hft_forecast.core.types import BAR_LABEL_KEY, Frequency, MarketData
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


@pytest.mark.parametrize(
    ("grid", "seconds", "vol_windows"),
    [
        (Frequency.S30, 300, [20, 100]),
        (Frequency.MIN1, 300, [20, 100]),
        (Frequency.MIN5, 1800, [5, 20]),
    ],
)
def test_every_grid_produces_usable_rows(data_service, experiment, grid, seconds, vol_windows):
    """One pipeline, three grids: the horizon and the window lengths are all that move.

    The windows are counted in bars, so a 100-bar volatility feature is eight minutes on a
    5 s grid and more than a whole session on a 5 min one; a coarse grid shrinks them.
    """
    from hft_forecast.core.run_config import PluginRef
    from hft_forecast.ml.base import HorizonSpec
    from hft_forecast.ml.datasets import null_report

    config = experiment.model_copy(
        update={
            "horizon": HorizonSpec(name=f"h{seconds}s", seconds=seconds, grid=grid),
            "data": experiment.data.model_copy(update={"frequency": grid}),
            "features": [PluginRef(name="basic", params={"vol_windows": vol_windows})],
        }
    )
    frame = load_dataset(build_dataset(config, data_service))
    assert null_report(frame)["f_vol_coarse_completed"] < 0.5
    assert usable_rows(frame).height > 0


def test_bars_on_the_wrong_grid_are_refused(quote_bars, experiment):
    """An imported file carries its own bar size, so the grid is checked, not assumed."""
    from hft_forecast.ml.datasets import _check_bars

    data = MarketData(quote_bars, Frequency.MIN1, metadata={BAR_LABEL_KEY: "end"})
    with pytest.raises(ConfigError, match="1min"):
        _check_bars(data, experiment.horizon)

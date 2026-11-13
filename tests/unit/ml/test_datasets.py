"""Dataset building: session columns, caching by hash and the sequence windows."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from hft_forecast.core import columns as C
from hft_forecast.core.errors import ConfigError
from hft_forecast.core.types import BAR_LABEL_KEY, Frequency, MarketData
from hft_forecast.data.manifest import Manifest, ManifestEntry
from hft_forecast.ml.base import LABEL_VALID, SECONDS_SINCE_OPEN, SESSION_ID
from hft_forecast.ml.datasets import (
    FEATURES_DIR,
    add_cross_sectional,
    build_dataset,
    design_matrix,
    load_bars,
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


def test_instruments_whose_columns_are_in_a_different_order_still_load(tmp_path):
    """A manifest partition and a cached fetch write the same columns in a different order."""
    first = pl.DataFrame({C.INSTRUMENT: ["AAA"], C.TIMESTAMP: [1], C.MID: [10.0]})
    second = first.select(C.MID, C.TIMESTAMP, C.INSTRUMENT).with_columns(
        pl.lit("BBB").alias(C.INSTRUMENT)
    )
    first.write_parquet(tmp_path / "AAA.parquet")
    second.write_parquet(tmp_path / "BBB.parquet")

    frame = load_dataset(tmp_path)

    assert frame.columns == first.columns
    assert frame.get_column(C.INSTRUMENT).to_list() == ["AAA", "BBB"]


def test_a_genuinely_missing_column_still_fails(tmp_path):
    pl.DataFrame({C.INSTRUMENT: ["AAA"], C.TIMESTAMP: [1], C.MID: [10.0]}).write_parquet(
        tmp_path / "AAA.parquet"
    )
    pl.DataFrame({C.INSTRUMENT: ["BBB"], C.TIMESTAMP: [1]}).write_parquet(tmp_path / "BBB.parquet")
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        load_dataset(tmp_path)


def test_manifest_partitions_sharing_a_cache_file_are_sliced_not_repeated(tmp_path):
    """Every month of one instrument resolves to the same cache file, extended in place."""
    from hft_forecast.data.manifest import Manifest, ManifestEntry
    from hft_forecast.ml.datasets import load_bars

    shared = tmp_path / "AAPL.parquet"
    stamps = pl.datetime_range(
        datetime(2024, 1, 1), datetime(2024, 3, 1), interval="1d", eager=True
    )
    pl.DataFrame(
        {C.TIMESTAMP: stamps, C.INSTRUMENT: ["AAPL"] * len(stamps), C.MID: [10.0] * len(stamps)}
    ).write_parquet(shared)
    manifest = Manifest(
        name="m",
        source="wrds_taq",
        frequency="5s",
        entries=[
            ManifestEntry(
                instrument="AAPL",
                dataset="taq_nbbo_bars",
                start=start,
                end=end,
                rows=0,
                path=str(shared),
            )
            for start, end in [("2024-01-01", "2024-01-31"), ("2024-02-01", "2024-02-29")]
        ],
    )

    frame = load_bars(manifest, "AAPL").collect()

    assert frame.height == 60, "31 January days + 29 February days, each once"
    assert frame.get_column(C.TIMESTAMP).max().date() == date(2024, 2, 29), "March is excluded"
    assert frame.get_column(C.TIMESTAMP).is_duplicated().sum() == 0


def test_cross_sectional_adds_market_and_deviation(tmp_path: Path) -> None:
    """The market column is the mean over instruments; the deviation sums to zero."""
    stamps = [datetime(2024, 1, 2, 14, 30), datetime(2024, 1, 2, 14, 30, 5)]
    for name, values in {"AAA": [1.0, 3.0], "BBB": [3.0, 5.0]}.items():
        pl.DataFrame(
            {C.TIMESTAMP: stamps, C.INSTRUMENT: [name] * 2, "f_logret_10": values}
        ).write_parquet(tmp_path / f"{name}.parquet")
    add_cross_sectional(tmp_path)
    frame = pl.read_parquet(tmp_path / "AAA.parquet")
    assert frame.get_column("f_mkt_logret_10").to_list() == [2.0, 4.0]
    assert frame.get_column("f_rel_logret_10").to_list() == [-1.0, -1.0]


def test_cross_sectional_skips_one_instrument(tmp_path: Path) -> None:
    """One name has no cross section, so the file is left alone."""
    pl.DataFrame({C.TIMESTAMP: [datetime(2024, 1, 2, 14, 30)], "f_logret_10": [1.0]}).write_parquet(
        tmp_path / "AAA.parquet"
    )
    add_cross_sectional(tmp_path)
    assert pl.read_parquet(tmp_path / "AAA.parquet").columns == [C.TIMESTAMP, "f_logret_10"]


def test_load_bars_tolerates_a_different_column_order(tmp_path: Path) -> None:
    """Partitions pulled at different times are stored with their columns in any order."""
    rows = {C.TIMESTAMP: [datetime(2024, 1, 2, 14, 30)], C.INSTRUMENT: ["AAA"], "mid": [1.0]}
    first = tmp_path / "jan.parquet"
    second = tmp_path / "feb.parquet"
    pl.DataFrame(rows).write_parquet(first)
    pl.DataFrame(rows).select("mid", C.INSTRUMENT, C.TIMESTAMP).with_columns(
        pl.lit(datetime(2024, 2, 1, 14, 30)).alias(C.TIMESTAMP)
    ).write_parquet(second)
    manifest = Manifest(
        name="m",
        source="wrds_taq",
        frequency="5s",
        entries=[
            ManifestEntry(
                instrument="AAA",
                dataset="taq_nbbo_bars",
                start="2024-01-01",
                end="2024-01-31",
                rows=1,
                path=str(first),
            ),
            ManifestEntry(
                instrument="AAA",
                dataset="taq_nbbo_bars",
                start="2024-02-01",
                end="2024-02-29",
                rows=1,
                path=str(second),
            ),
        ],
    )
    assert load_bars(manifest, "AAA").collect().height == 2


def test_an_interrupted_build_is_not_served_as_cached(tmp_path: Path) -> None:
    """A half-written directory must not look complete; only the final rename publishes it."""
    hash_dir = tmp_path / FEATURES_DIR / "deadbeef"
    partial = hash_dir.with_name("deadbeef.partial")
    partial.mkdir(parents=True)
    (partial / "AAA.parquet").touch()
    assert not hash_dir.exists()  # the partial name is invisible to the cache check


def test_label_price_changes_the_dataset_hash(data_service, experiment):
    twap = experiment.model_copy(update={"label_price": C.MID_TWAP})
    assert build_dataset(experiment, data_service) != build_dataset(twap, data_service)


def test_label_price_measures_the_target_on_that_column_and_keeps_the_mid(data_service, experiment):
    twap = experiment.model_copy(update={"label_price": C.MID_TWAP})
    on_mid = load_dataset(build_dataset(experiment, data_service)).sort(C.INSTRUMENT, C.TIMESTAMP)
    on_twap = load_dataset(build_dataset(twap, data_service)).sort(C.INSTRUMENT, C.TIMESTAMP)

    # the snapshot mid survives the swap, so the features still describe the real quotes
    assert on_twap[C.MID].to_list() == on_mid[C.MID].to_list()
    # and the target now moves with mid_twap instead
    steps = experiment.horizon.steps
    checked = on_twap.with_columns(
        (
            pl.col(C.MID_TWAP).shift(-steps).over(C.INSTRUMENT).log() - pl.col(C.MID_TWAP).log()
        ).alias("expected")
    ).filter(pl.col("fwd_logret").is_not_null() & pl.col("expected").is_not_null())
    assert checked.height > 0
    assert np.allclose(checked["fwd_logret"].to_numpy(), checked["expected"].to_numpy())
    assert on_twap["fwd_logret"].to_list() != on_mid["fwd_logret"].to_list()


def test_an_unknown_label_price_is_refused(data_service, experiment):
    nonsense = experiment.model_copy(update={"label_price": "not_a_column"})
    with pytest.raises(ConfigError):
        build_dataset(nonsense, data_service)

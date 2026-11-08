"""Labels: the session guard, the dead zone and the drop rules."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from hft_forecast.core import columns as C
from hft_forecast.ml.base import (
    DIRECTION,
    FWD_LOGRET,
    H30S,
    LABEL_END,
    LABEL_VALID,
    MAGNITUDE_BPS,
    SESSION_ID,
)
from hft_forecast.ml.config import LabelParams
from hft_forecast.ml.labels import label_report, make_labels

STEPS = H30S.steps  # 6 bars of 5 s


@pytest.fixture
def labelled(quote_bars):
    return make_labels(quote_bars, H30S, LabelParams())


def test_the_forward_return_is_the_log_mid_six_bars_ahead(labelled):
    one = labelled.filter(
        (pl.col(C.INSTRUMENT) == "AAA") & (pl.col(SESSION_ID) == labelled[SESSION_ID].min())
    ).sort(C.TIMESTAMP)
    mid = one.get_column(C.MID).to_numpy()
    got = one.get_column(FWD_LOGRET).to_numpy()
    want = np.log(mid[STEPS:]) - np.log(mid[:-STEPS])
    assert np.allclose(got[:-STEPS], want, atol=1e-12, equal_nan=True)
    assert np.isnan(got[-STEPS:]).all()  # the last bars of a session have no target


def test_no_label_crosses_a_session(labelled):
    valid = labelled.filter(pl.col(LABEL_VALID))
    span = (valid[LABEL_END] - valid[C.TIMESTAMP]).dt.total_seconds().to_numpy()
    assert (span == H30S.seconds).all()
    last_bar = labelled.group_by([C.INSTRUMENT, SESSION_ID]).agg(
        pl.col(C.TIMESTAMP).max().alias("last_bar")
    )
    joined = (
        valid.group_by([C.INSTRUMENT, SESSION_ID])
        .agg(pl.col(LABEL_END).max().alias("last_target"))
        .join(last_bar, on=[C.INSTRUMENT, SESSION_ID])
    )
    assert (joined["last_target"] <= joined["last_bar"]).all()


def test_the_dead_zone_widens_the_flat_class(quote_bars):
    narrow = make_labels(quote_bars, H30S, LabelParams(dead_zone_bps=0.0))
    wide = make_labels(quote_bars, H30S, LabelParams(dead_zone_bps=50.0))
    flat = lambda f: (f.filter(pl.col(LABEL_VALID))[DIRECTION].to_numpy() == 0).mean()  # noqa: E731
    assert flat(narrow) < flat(wide)
    assert flat(wide) > 0.5


def test_warmup_and_cooldown_mask_the_crosses(quote_bars):
    labelled = make_labels(quote_bars, H30S, LabelParams(warmup_seconds=120, cooldown_seconds=120))
    valid = labelled.filter(pl.col(LABEL_VALID))
    assert valid["seconds_since_open"].min() >= 120
    assert valid["seconds_to_close"].min() >= 120


def test_stale_targets_are_dropped_unless_asked_for(quote_bars):
    kept = make_labels(quote_bars, H30S, LabelParams(drop_stale_targets=False))
    dropped = make_labels(quote_bars, H30S, LabelParams(drop_stale_targets=True))
    assert dropped[LABEL_VALID].sum() < kept[LABEL_VALID].sum()


def test_the_report_adds_up(labelled):
    report = label_report(labelled)
    assert report["n_valid"] <= report["n_rows"]
    assert report["class_down"] + report["class_flat"] + report["class_up"] == pytest.approx(1.0)
    assert report["dropped_closing_cross"] > 0
    assert (labelled[MAGNITUDE_BPS].drop_nulls() >= 0).all()


def test_a_start_lag_shifts_the_labelled_return(quote_bars):
    lag = 2
    lagged = make_labels(quote_bars, H30S, LabelParams(start_lag_bars=lag))
    one = lagged.filter(
        (pl.col(C.INSTRUMENT) == "AAA") & (pl.col(SESSION_ID) == lagged[SESSION_ID].min())
    ).sort(C.TIMESTAMP)
    mid = one.get_column(C.MID).to_numpy()
    got = one.get_column(FWD_LOGRET).to_numpy()
    want = np.log(mid[lag + STEPS :]) - np.log(mid[lag : -STEPS])
    assert np.allclose(got[: -(lag + STEPS)], want, atol=1e-12, equal_nan=True)
    valid = lagged.filter(pl.col(LABEL_VALID))
    span = (valid[LABEL_END] - valid[C.TIMESTAMP]).dt.total_seconds().to_numpy()
    assert (span == H30S.seconds + lag * 5).all()  # the purge sees the whole lagged interval


def test_a_zero_lag_keeps_the_original_label_and_hashes(quote_bars):
    from pathlib import Path

    from hft_forecast.ml import config as cfgmod

    base = make_labels(quote_bars, H30S, LabelParams())
    zero = make_labels(quote_bars, H30S, LabelParams(start_lag_bars=0))
    assert base.equals(zero)

    path = Path(__file__).parents[3] / "backend/hft_forecast/ml/horizons/h5m/config.yaml"
    cfg = cfgmod.ExperimentConfig.load(path)
    # the hash a config had before start_lag_bars existed
    old = cfg.model_dump(mode="json", include={"horizon", "data", "manifest", "labels"})
    del old["labels"]["start_lag_bars"]
    old["features"] = [f.model_dump(mode="json") for f in cfg.features]
    assert cfg.dataset_hash() == cfgmod._hash(old)
    lagged = cfg.model_copy(update={"labels": LabelParams(start_lag_bars=1)})
    assert lagged.dataset_hash() != cfg.dataset_hash()
    assert lagged.config_hash() != cfg.config_hash()

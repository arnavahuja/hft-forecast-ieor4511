"""h30s group: the order-flow feature set and the LightGBM model."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from hft_forecast.core import columns as C
from hft_forecast.core.errors import ConfigError
from hft_forecast.ml.base import N_CLASSES, SESSION_ID
from hft_forecast.ml.features import check_feature_lookahead, feature_columns
from hft_forecast.ml.features.h30s_flow import H30sFlow
from hft_forecast.ml.models.h30s_lgbm import Lgbm


def test_flow_features_are_causal(quote_bars):
    check_feature_lookahead(H30sFlow(), quote_bars.head(3000))


def test_flow_features_are_finite_and_restart_each_session(quote_bars):
    frame = H30sFlow().compute(quote_bars)
    columns = feature_columns(frame)
    assert {"f_ofi_6", "f_imb_x_one_tick", "f_spread_ticks", "f_twap_dev_bps"} <= set(columns)
    settled = frame.group_by([C.INSTRUMENT, SESSION_ID]).tail(100)
    assert np.isfinite(settled.select(columns).to_numpy()).all()
    first = frame.sort(C.TIMESTAMP).group_by([C.INSTRUMENT, SESSION_ID]).first()
    assert first["f_imb_change_1"].is_null().all()  # no previous bar in the session


def test_ofi_matches_a_hand_computed_case():
    t0 = datetime(2024, 1, 2, 15, 0, 5, tzinfo=UTC)
    rows = [  # bid, ask, bid_size, ask_size
        (10.00, 10.01, 100.0, 200.0),
        (10.00, 10.01, 150.0, 200.0),  # bid queue grows: +50
        (10.01, 10.02, 300.0, 100.0),  # bid up: +300; ask up: +200 (old ask queue)
    ]
    frame = (
        pl.DataFrame(
            {
                C.TIMESTAMP: [t0 + timedelta(seconds=5 * i) for i in range(3)],
                C.INSTRUMENT: "X",
                SESSION_ID: 1,
                C.BID: [r[0] for r in rows],
                C.ASK: [r[1] for r in rows],
                C.BID_SIZE: [r[2] for r in rows],
                C.ASK_SIZE: [r[3] for r in rows],
            }
        )
        .with_columns(
            (pl.col(C.BID) + pl.col(C.ASK)).truediv(2).alias(C.MID),
            (pl.col(C.ASK) - pl.col(C.BID)).alias(C.SPREAD),
        )
        .with_columns(
            pl.col(C.MID).alias(C.MID_TWAP),
            pl.col(C.MID).alias(C.MICROPRICE),
            pl.col(C.MID).alias(C.HIGH),
            pl.col(C.MID).alias(C.LOW),
            pl.lit(10.0).alias(C.N_QUOTES),
        )
    )
    out = H30sFlow().compute(frame)
    depth_1 = [(r[2] + r[3]) / 2 for r in rows]
    assert out["f_ofi_1"][1] == pytest.approx(50 / depth_1[1])
    assert out["f_ofi_1"][2] == pytest.approx((300 + 200) / depth_1[2])
    assert out["f_ofi_2"][2] == pytest.approx((50 + 500) / np.mean(depth_1[1:]))
    assert out["f_spread_ticks"].to_list() == pytest.approx([1.0, 1.0, 1.0])
    assert out["f_one_tick_share"][2] == pytest.approx(1.0)


@pytest.fixture
def sample():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3000, 3))
    y_ret = 1e-4 * (0.5 * x[:, 0] + rng.normal(scale=0.5, size=3000))
    y_dir = np.sign(y_ret).astype(int)
    return x, y_dir, y_ret


def _fit(sample, weight=None):
    model = Lgbm(Lgbm.Params(min_child_samples=20, n_estimators=50, n_jobs=1))
    model.set_feature_names(("a", "b", "c"))
    model.fit(*sample, weight, 7)
    return model


def test_lgbm_returns_a_valid_distribution_and_learns(sample):
    x, y_dir, _ = sample
    probabilities, y_hat = _fit(sample).predict(x)
    assert probabilities.shape == (len(x), N_CLASSES)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.isfinite(y_hat).all()
    truth = np.array([{-1: 0, 0: 1, 1: 2}[int(d)] for d in y_dir])
    assert (probabilities.argmax(axis=1) == truth).mean() > 0.6
    assert np.corrcoef(y_hat, x[:, 0])[0, 1] > 0.5


def test_lgbm_is_deterministic_and_uses_weights(sample):
    first, second = _fit(sample).predict(sample[0]), _fit(sample).predict(sample[0])
    assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])
    weighted = _fit(sample, np.linspace(0.1, 1.0, len(sample[0]))).predict(sample[0])
    assert not np.allclose(first[1], weighted[1])


def test_lgbm_holdout_blocks_spread_over_the_fold():
    model = Lgbm(Lgbm.Params(valid_block=100, valid_every=10))
    mask = model.holdout_mask(10_000)
    assert mask.mean() == pytest.approx(0.1)
    assert mask[900:1000].all() and not mask[:900].any()  # every 10th block of 100 rows
    small = model.holdout_mask(50)  # one block only: the last tenth is held out
    assert small.any() and not small.all()


def test_lgbm_refuses_to_predict_before_fitting():
    with pytest.raises(ConfigError, match="fitted"):
        Lgbm().predict(np.zeros((2, 3)))

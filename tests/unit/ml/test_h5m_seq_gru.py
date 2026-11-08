"""h5m_seq (causal lagged raw variables) and h5m_gru (reads them as a sequence)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from hft_forecast.core import columns as C
from hft_forecast.ml.base import SESSION_ID
from hft_forecast.ml.features import FEATURE_SETS
from hft_forecast.ml.features.h5m_seq import VARIABLES, H5mSeq, SeqParams


def _needs(frame: pl.DataFrame) -> None:
    missing = [c for c in H5mSeq.requires if c not in frame.columns]
    if missing:
        pytest.skip(f"fixture lacks {missing}")


def test_seq_is_registered():
    assert FEATURE_SETS.get("h5m_seq").create().params.length == 20


def test_seq_columns_are_lags_of_each_other_within_a_session(quote_bars):
    _needs(quote_bars)
    out = H5mSeq(SeqParams(length=5)).compute(quote_bars)
    names = [c for c in out.columns if c.startswith("f_seq_")]
    assert len(names) == len(VARIABLES) * 5
    one = out.filter(
        (pl.col(C.INSTRUMENT) == out[C.INSTRUMENT][0]) & (pl.col(SESSION_ID) == out[SESSION_ID][0])
    ).sort(C.TIMESTAMP)
    now = one["f_seq_imb_l00"].to_numpy()
    lag2 = one["f_seq_imb_l02"].to_numpy()
    assert np.allclose(lag2[2:], now[:-2], equal_nan=True)
    assert np.isnan(one["f_seq_ret_l00"].to_numpy()[0])  # no previous bar in the session
    assert one["f_seq_imb_l02"].head(2).is_null().all()


torch = pytest.importorskip("torch")

from hft_forecast.ml.models import MODELS  # noqa: E402
from hft_forecast.ml.models.h5m_gru import GruParams, H5mGru, sequence_layout  # noqa: E402

LAGS, N = 6, 3000
NAMES = tuple(
    [f"f_seq_{v}_l{lag:02d}" for v in ("a", "b") for lag in range(LAGS)] + ["f_static"]
)


def test_layout_orders_lags_oldest_first():
    grid, static = sequence_layout(NAMES)
    assert grid.shape == (LAGS, 2)
    assert NAMES[grid[0, 0]] == "f_seq_a_l05" and NAMES[grid[-1, 1]] == "f_seq_b_l00"
    assert [NAMES[i] for i in static] == ["f_static"]


@pytest.fixture
def seq_sample():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(N, len(NAMES)))
    signal = x[:, NAMES.index("f_seq_a_l03")]  # one lagged value carries the signal
    y_ret = 1e-4 * signal + 2e-5 * rng.normal(size=N)
    y_dir = np.where(y_ret > 3e-5, 1, np.where(y_ret < -3e-5, -1, 0))
    return x, y_dir, y_ret


def _fit(sample, seed=7):
    model = H5mGru(GruParams(valid_block=200, max_epochs=15, patience=4, threads=1, batch_size=128,
                             learning_rate=3e-3))
    model.set_feature_names(NAMES)
    model.fit(*sample, None, seed)
    return model


def test_gru_registered():
    assert MODELS.get("h5m_gru").label == "h5m_gru@1.0.0"


def test_gru_finds_a_signal_inside_the_sequence(seq_sample):
    x, _, y_ret = seq_sample
    probs, y_hat = _fit(seq_sample).predict(x)
    assert probs.shape == (N, 3) and np.allclose(probs.sum(axis=1), 1.0)
    assert np.corrcoef(y_hat, y_ret)[0, 1] > 0.3


def test_gru_same_seed_same_predictions(seq_sample):
    a = _fit(seq_sample).predict(seq_sample[0])
    b = _fit(seq_sample).predict(seq_sample[0])
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])

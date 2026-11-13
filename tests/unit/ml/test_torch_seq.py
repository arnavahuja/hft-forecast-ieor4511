"""The three sequence encoders: shapes, session-safe windows and a learnable signal."""

from __future__ import annotations

import numpy as np
import pytest

from hft_forecast.core.errors import ConfigError
from hft_forecast.ml.base import N_CLASSES
from hft_forecast.ml.models.torch_seq import TorchSeq, TorchSeqParams, session_windows

pytest.importorskip("torch")

SESSION_BARS = 200
CLOCK = "f_seconds_since_open"


def _fold(sessions: int = 12, rng_seed: int = 0):
    """Two features over whole sessions; the label is the first feature's own sign.

    The signal sits in the last bar of the window, so every encoder can reach it.
    """
    rng = np.random.default_rng(rng_seed)
    clock = np.tile(np.arange(1, SESSION_BARS + 1) * 5.0, sessions)
    signal = rng.normal(size=sessions * SESSION_BARS)
    noise = rng.normal(size=sessions * SESSION_BARS)
    x = np.column_stack([signal, noise, clock])
    y_ret = signal * 2.0
    y_dir = np.sign(y_ret).astype(np.int64)
    return x, y_dir, y_ret


def _model(encoder: str) -> TorchSeq:
    model = TorchSeq(
        TorchSeqParams(
            encoder=encoder,
            lookback=8,
            hidden=16,
            epochs=6,
            batch_size=64,
            train_stride=1,
            learning_rate=0.01,
        )
    )
    model.set_feature_names(("f_signal", "f_noise", CLOCK))
    return model


def test_windows_never_cross_a_session() -> None:
    """The mask is false for steps that belong to the previous session."""
    clock = np.array([5.0, 10.0, 15.0, 5.0, 10.0])  # a new session at position 3
    index, valid = session_windows(clock, 3)
    assert valid[2].tolist() == [True, True, True]  # 5, 10, 15 of session one
    assert valid[3].tolist() == [False, False, True]  # only its own bar
    assert index[0].tolist() == [0, 0, 0]  # clipped at the start of the array


@pytest.mark.parametrize("encoder", ["gru", "lstm", "tcn"])
def test_each_encoder_fits_and_predicts(encoder: str) -> None:
    """Shapes are right and the fitted model beats a zero prediction on the training signal."""
    x, y_dir, y_ret = _fold()
    model = _model(encoder)
    model.fit(x, y_dir, y_ret, None, seed=7)
    probabilities, predicted = model.predict(x)
    assert probabilities.shape == (len(x), N_CLASSES)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert predicted.shape == (len(x),)
    assert np.abs(y_ret - predicted).mean() < np.abs(y_ret).mean()
    assert np.corrcoef(y_ret, predicted)[0, 1] > 0.5


def test_unknown_encoder_is_rejected() -> None:
    x, y_dir, y_ret = _fold(sessions=4)
    model = _model("transformer")
    with pytest.raises(ConfigError, match="unknown encoder"):
        model.fit(x, y_dir, y_ret, None, seed=1)


def test_without_the_clock_feature_the_fold_is_one_session() -> None:
    """No clock means no boundaries to find, so it still fits (and says so in the log)."""
    x, y_dir, y_ret = _fold(sessions=4)
    model = _model("gru")
    model.set_feature_names(("f_signal", "f_noise", "f_other"))
    model.fit(x, y_dir, y_ret, None, seed=1)
    assert model.predict(x)[1].shape == (len(x),)


def test_predicting_before_fitting_is_rejected() -> None:
    with pytest.raises(ConfigError, match="must be fitted"):
        _model("gru").predict(np.zeros((4, 3)))

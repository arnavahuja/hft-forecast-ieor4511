"""h5m_mlp: valid outputs, learns a planted signal, deterministic, uses the weights."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from hft_forecast.ml.models import MODELS  # noqa: E402
from hft_forecast.ml.models.h5m_mlp import H5mMlp, MlpParams  # noqa: E402

N = 4000


@pytest.fixture
def sample():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(N, 4))
    y_ret = 1e-4 * x[:, 0] + 2e-5 * rng.normal(size=N)  # column 0 carries the signal
    y_dir = np.where(y_ret > 3e-5, 1, np.where(y_ret < -3e-5, -1, 0))
    return x, y_dir, y_ret


def _fit(sample, weight=None, seed=7, **params):
    settings = {"valid_block": 200, "max_epochs": 8, "threads": 1, "batch_size": 128,
                "learning_rate": 3e-3, **params}
    model = H5mMlp(MlpParams(**settings))
    model.fit(*sample, weight, seed)
    return model


def test_registered():
    assert MODELS.get("h5m_mlp").label == "h5m_mlp@1.0.0"


def test_outputs_are_valid_and_the_signal_is_learned(sample):
    x, y_dir, y_ret = sample
    probs, y_hat = _fit(sample).predict(x)
    assert probs.shape == (N, 3) and y_hat.shape == (N,)
    assert np.all(probs >= 0) and np.allclose(probs.sum(axis=1), 1.0)
    assert np.corrcoef(y_hat, y_ret)[0, 1] > 0.5
    assert np.corrcoef(probs[:, 2] - probs[:, 0], y_ret)[0, 1] > 0.5


def test_same_seed_same_predictions(sample):
    a = _fit(sample).predict(sample[0])
    b = _fit(sample).predict(sample[0])
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_weights_change_the_fit(sample):
    weight = np.where(np.arange(N) % 2 == 0, 1.0, 0.01)
    plain = _fit(sample).predict(sample[0])[1]
    weighted = _fit(sample, weight=weight).predict(sample[0])[1]
    assert not np.allclose(plain, weighted)


def test_two_classes_only_and_early_stopping_record():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(1000, 2))
    y_dir = np.where(x[:, 0] > 0, 1, -1)
    y_ret = 1e-4 * x[:, 0]
    model = _fit((x, y_dir, y_ret))
    probs, _ = model.predict(x)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert 0 <= model.best_epoch < 8 and len(model.history) >= 1
    assert set(model.state_dict()) == {"net", "mean", "std", "best_epoch", "history"}


def test_scaling_uses_fitting_rows_only(sample):
    model = _fit(sample)
    mask = model.holdout_mask(N)
    assert mask.any() and (~mask).any()
    assert np.allclose(model._mean, sample[0][~mask].mean(axis=0))

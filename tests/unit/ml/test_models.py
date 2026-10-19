"""The three baselines, the two stubs and the registry contract."""

from __future__ import annotations

import numpy as np
import pytest

from hft_forecast.core.errors import ConfigError, PluginNotFoundError
from hft_forecast.ml.base import N_CLASSES
from hft_forecast.ml.models import MODELS

FEATURES = ("f_logret_10", "f_spread_bps")
IMPLEMENTED = [s.name for s in MODELS.specs() if s.cls.implemented]
STUBS = [s.name for s in MODELS.specs() if not s.cls.implemented]


@pytest.fixture
def sample():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, len(FEATURES)))
    y_ret = 0.3 * x[:, 0] + rng.normal(scale=0.5, size=400)
    y_dir = np.sign(y_ret).astype(int)
    return x, y_dir, y_ret


def _fit(name, sample, weight=None):
    model = MODELS.get(name).create()
    model.set_feature_names(FEATURES)
    model.fit(*sample, weight, 7)
    return model


@pytest.mark.parametrize("name", IMPLEMENTED)
def test_every_baseline_returns_a_valid_distribution(name, sample):
    model = _fit(name, sample)
    probabilities, y_hat = model.predict(sample[0])
    assert probabilities.shape == (len(sample[0]), N_CLASSES)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert (probabilities >= 0).all()
    assert y_hat.shape == (len(sample[0]),)
    assert np.isfinite(y_hat).all()


@pytest.mark.parametrize("name", IMPLEMENTED)
def test_fitting_twice_gives_identical_predictions(name, sample):
    first = _fit(name, sample).predict(sample[0])
    second = _fit(name, sample).predict(sample[0])
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_zero_learns_the_class_priors(sample):
    x, y_dir, _ = sample
    probabilities, y_hat = _fit("zero", sample).predict(x)
    assert probabilities[0, 2] == pytest.approx((y_dir == 1).mean())
    assert not y_hat.any()


def test_last_sign_follows_the_named_feature(sample):
    x, _, _ = sample
    probabilities, y_hat = _fit("last_sign", sample).predict(x)
    assert np.array_equal(y_hat, x[:, 0])
    assert (probabilities[x[:, 0] > 0, 2] == 1.0).all()
    assert (probabilities[x[:, 0] < 0, 0] == 1.0).all()


def test_last_sign_needs_its_feature(sample):
    model = MODELS.get("last_sign").create()
    model.set_feature_names(("f_other",))
    with pytest.raises(ConfigError, match="f_logret_10"):
        model.fit(*sample, None, 7)


def test_logit_ridge_beats_zero_on_a_learnable_signal(sample):
    x, y_dir, _ = sample
    fitted = _fit("logit_ridge", sample).predict(x)[0]
    baseline = _fit("zero", sample).predict(x)[0]
    truth = np.array([{-1: 0, 0: 1, 1: 2}[int(d)] for d in y_dir])
    assert (fitted.argmax(axis=1) == truth).mean() > (baseline.argmax(axis=1) == truth).mean()


def test_logit_ridge_accepts_sample_weights(sample):
    weight = np.linspace(0.1, 1.0, len(sample[0]))
    plain = _fit("logit_ridge", sample).predict(sample[0])[1]
    weighted = _fit("logit_ridge", sample, weight).predict(sample[0])[1]
    assert not np.allclose(plain, weighted)


def test_logit_ridge_refuses_to_predict_before_fitting():
    with pytest.raises(ConfigError, match="fitted"):
        MODELS.get("logit_ridge").create().predict(np.zeros((2, 2)))


@pytest.mark.parametrize("name", STUBS)
def test_stubs_say_what_is_missing(name, sample):
    model = MODELS.get(name).create()
    with pytest.raises(NotImplementedError, match="stub"):
        model.fit(*sample, None, 1)


def test_the_registry_reports_what_it_has():
    assert {"zero", "last_sign", "logit_ridge", "gbm", "torch_seq"} == set(MODELS.names())
    assert MODELS.get("logit_ridge").label == "logit_ridge@1.0.0"
    with pytest.raises(PluginNotFoundError):
        MODELS.get("nope")

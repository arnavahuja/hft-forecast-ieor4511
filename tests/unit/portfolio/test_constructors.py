from __future__ import annotations

import numpy as np
import pytest

from hft_forecast.core.interfaces import ConstructionContext
from hft_forecast.core.portfolio_math import (
    hrp_weights,
    risk_parity_weights,
    trailing_cov,
)
from hft_forecast.core.registry import registry
from hft_forecast.core.types import TargetFrame, TargetKind
from tests.helpers import PPY, synthetic

DATA = synthetic(instruments=("A", "B", "C", "D", "E", "F", "G", "H", "I", "J"))
T = len(DATA.timestamps)


def _signals(values=None) -> TargetFrame:
    rng = np.random.default_rng(1)
    vals = rng.normal(size=(T, 10)) if values is None else values
    return TargetFrame(DATA.timestamps, DATA.instruments, vals, TargetKind.SIGNALS)


def _construct(name: str, **params):
    cons = registry("portfolio_constructor").get(name).create(params)
    return cons.construct(_signals(), ConstructionContext(DATA, PPY)).values


def test_equal_weight_gross_and_sides():
    w = _construct("equal_weight", gross=1.0)
    np.testing.assert_allclose(np.abs(w).sum(axis=1), 1.0)
    w_long = _construct("equal_weight", allow_short=False)
    assert (w_long >= 0).all()


def test_quantile_long_short_is_dollar_neutral_deciles():
    w = _construct("quantile_long_short", quantiles=10)
    np.testing.assert_allclose(w.sum(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(w).sum(axis=1), 1.0)
    assert ((w > 0).sum(axis=1) == 1).all()  # 10 names -> 1 per decile


def test_risk_parity_equalizes_risk_contributions():
    rets = np.random.default_rng(0).normal(0, [0.01, 0.02, 0.04], size=(500, 3))
    cov = trailing_cov(rets, 499, 500, np.arange(3), shrinkage=0.0)
    w = risk_parity_weights(cov)
    rc = w * (cov @ w)
    np.testing.assert_allclose(rc / rc.sum(), 1 / 3, atol=1e-6)
    assert w[0] > w[1] > w[2]


def test_hrp_is_long_only_and_sums_to_one():
    rets = np.random.default_rng(2).normal(0, 0.01, size=(300, 6))
    cov = trailing_cov(rets, 299, 300, np.arange(6))
    w = hrp_weights(cov)
    assert w.sum() == pytest.approx(1.0) and (w > 0).all()


def test_signal_proportional_scales_to_gross():
    w = _construct("signal_proportional", gross=1.0)
    np.testing.assert_allclose(np.abs(w).sum(axis=1), 1.0)

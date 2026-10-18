from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from hft_forecast.engine.simulation import last_true_before, simulate_segments


def test_last_true_before():
    mask = np.array([False, True, False, False, True, False])
    assert last_true_before(mask).tolist() == [-1, -1, 1, 1, 1, 4]


def test_single_asset_buy_and_hold():
    r = np.array([[0.0], [0.1], [-0.05], [0.02]])
    tgt = np.ones_like(r)
    rebal = np.array([True, False, False, False])
    path = simulate_segments(r, tgt, rebal)
    np.testing.assert_allclose(path.returns, [0.0, 0.1, -0.05, 0.02])
    np.testing.assert_allclose(path.post_trade[:, 0], 1.0)


def test_drift_vs_constant_mix():
    r = np.array([[0.0, 0.0], [0.10, -0.10], [0.10, -0.10]])
    tgt = np.full_like(r, 0.5)
    drift = simulate_segments(r, tgt, np.array([True, False, False]))
    mix = simulate_segments(r, tgt, np.array([True, True, True]))
    # constant mix: every bar returns the average
    np.testing.assert_allclose(mix.returns, [0.0, 0.0, 0.0], atol=1e-15)
    # drift: after bar 1 weights are 0.55/0.45, so bar 2 return is 0.055 - 0.045 = 0.01
    np.testing.assert_allclose(drift.returns, [0.0, 0.0, 0.01], atol=1e-15)
    np.testing.assert_allclose(drift.pre_trade[1], [0.55, 0.45])


def test_short_position_and_cash():
    r = np.array([[0.0], [0.10], [0.10]])
    tgt = -np.ones_like(r)
    path = simulate_segments(r, tgt, np.array([True, False, False]))
    # value: short -1 x growth + cash 2 -> 0.9, then 2 - 1.21 = 0.79
    np.testing.assert_allclose(path.returns, [0.0, -0.1, 0.79 / 0.9 - 1.0])


def test_cash_rate():
    r = np.zeros((3, 1))
    path = simulate_segments(r, np.zeros_like(r), np.zeros(3, bool), np.full(3, 0.01))
    np.testing.assert_allclose(path.returns, 0.01)


@settings(max_examples=60, deadline=None)
@given(
    rets=arrays(np.float64, (30, 3), elements=st.floats(-0.2, 0.2)),
    weights=arrays(np.float64, (3,), elements=st.floats(-1.0, 1.0)),
)
def test_step_return_equals_start_weights_times_returns(rets, weights):
    """Invariant: step return = sum(start-of-step weights x asset returns)."""
    tgt = np.tile(weights, (30, 1))
    rebal = np.zeros(30, bool)
    rebal[[0, 10, 20]] = True
    path = simulate_segments(rets, tgt, rebal)
    implied = (path.start_of_step * rets).sum(axis=1)
    np.testing.assert_allclose(path.returns, implied, atol=1e-9)

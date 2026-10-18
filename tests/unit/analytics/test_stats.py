"""Metric statistics against hand-computed values and empyrical as a reference library."""

from __future__ import annotations

import empyrical as ep
import numpy as np
import pandas as pd
import pytest

from hft_forecast.analytics import stats as S
from tests.helpers import PPY

RNG = np.random.default_rng(42)
R = RNG.normal(0.0004, 0.012, 1500)
B = 0.6 * R + RNG.normal(0.0002, 0.008, 1500)
IDX = pd.bdate_range("2015-01-01", periods=len(R), tz="UTC")
RS = pd.Series(R, index=IDX)
BS = pd.Series(B, index=IDX)


def test_hand_computed_basics():
    r = np.array([0.10, -0.05, 0.02])
    assert S.total_return(r) == pytest.approx(1.1 * 0.95 * 1.02 - 1)
    assert S.max_drawdown(np.array([0.1, -0.5, 0.2])) == pytest.approx(-0.5)
    assert S.cagr(np.full(252, 0.001), 252) == pytest.approx(1.001**252 - 1)
    periods = S.drawdown_periods(np.array([0.1, -0.1, -0.1, 0.3, -0.05]))
    assert periods[0].start == 0 and periods[0].trough == 2 and periods[0].end == 3


def test_against_empyrical():
    assert S.annualized_vol(R, PPY) == pytest.approx(ep.annual_volatility(RS, annualization=252))
    assert S.sharpe(R, PPY) == pytest.approx(ep.sharpe_ratio(RS, annualization=252))
    assert S.sortino(R, PPY) == pytest.approx(ep.sortino_ratio(RS, annualization=252))
    assert S.max_drawdown(R) == pytest.approx(ep.max_drawdown(RS))
    assert S.cagr(R, PPY) == pytest.approx(ep.annual_return(RS, annualization=252))
    assert S.omega(R) == pytest.approx(ep.omega_ratio(RS))
    assert S.tail_ratio(R) == pytest.approx(ep.tail_ratio(RS))
    assert S.downside_deviation(R, PPY) == pytest.approx(ep.downside_risk(RS, annualization=252))
    calmar = S.cagr(R, PPY) / abs(S.max_drawdown(R))
    assert calmar == pytest.approx(ep.calmar_ratio(RS, annualization=252))
    rel = S.relative_stats(R, B, PPY, 0.0)
    alpha, beta = ep.alpha_beta(RS, BS, annualization=252)
    assert rel.beta == pytest.approx(beta)
    # empyrical compounds the daily alpha; we report the arithmetic annualized alpha
    assert rel.alpha / 252 == pytest.approx((1 + alpha) ** (1 / 252) - 1)
    assert S.var_historical(R) == pytest.approx(-ep.value_at_risk(RS, cutoff=0.05))
    assert S.cvar_historical(R) == pytest.approx(-ep.conditional_value_at_risk(RS, 0.05))


def test_psr_and_dsr_properties():
    psr = S.probabilistic_sharpe(R)
    assert 0.5 < psr < 1.0
    trials = RNG.normal(0.0, 0.03, 50)
    dsr = S.deflated_sharpe(R, trials, 50)
    assert dsr < psr
    assert S.deflated_sharpe(R, trials, 1) == pytest.approx(psr)


def test_aggregate_returns_compounds_within_months():
    ts = np.array(["2024-01-30", "2024-01-31", "2024-02-01"], dtype="datetime64[us]")
    codes, monthly = S.aggregate_returns(np.array([0.1, 0.1, -0.5]), ts, "M")
    assert len(codes) == 2
    np.testing.assert_allclose(monthly, [0.21, -0.5])


def test_bootstrap_is_seeded_and_brackets_point_estimate():
    a = S.bootstrap_ci(R, PPY, seed=1, n_samples=500)
    b = S.bootstrap_ci(R, PPY, seed=1, n_samples=500)
    assert a == b
    lo, hi = a["sharpe"]
    assert lo < S.sharpe(R, PPY) < hi


def test_stationary_bootstrap_indices_in_range():
    idx = S.stationary_bootstrap_indices(100, 20, 5.0, np.random.default_rng(0))
    assert idx.shape == (20, 100)
    assert idx.min() >= 0 and idx.max() < 100
    # mostly consecutive (blocks)
    assert (np.diff(idx, axis=1) == 1).mean() > 0.7

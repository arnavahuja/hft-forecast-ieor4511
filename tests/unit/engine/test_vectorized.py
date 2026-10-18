from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hft_forecast.core import columns as C
from hft_forecast.core.registry import registry
from hft_forecast.core.run_config import RebalanceRule
from hft_forecast.core.types import MONEY_ATOL, MONEY_RTOL
from hft_forecast.engine.base import RunOptions
from hft_forecast.engine.pipeline import Pipeline
from hft_forecast.engine.vectorized import VectorizedEngine
from tests.helpers import PPY, config, make_market

OPTS = RunOptions(periods_per_year=PPY)


def _strategy(name: str = "buy_and_hold", **params):
    return registry("strategy").get(name).create(params)


def test_lagged_close_execution_single_asset():
    closes = [100.0, 110.0, 99.0, 108.9]
    data = make_market({"A": closes})
    res = VectorizedEngine().run(config(), data, _strategy(), Pipeline(), OPTS)
    # decision at close of bar 0, trade at close of bar 1 -> first exposed return is bar 2
    np.testing.assert_allclose(res.returns, [0.0, 0.0, -0.1, 0.1], atol=1e-12)
    assert res.equity[-1] == pytest.approx(1_000_000 * 108.9 / 110.0)
    assert res.weights[1, 0] == pytest.approx(1.0)
    assert res.trades.height == 1
    assert res.trades["is_open"][0]


def test_open_execution_uses_intraday_leg_on_trade_bar():
    closes = [100.0, 110.0, 121.0]
    opens = [100.0, 105.0, 115.0]
    data = make_market({"A": closes}, opens={"A": opens})
    res = VectorizedEngine().run(config(price="open"), data, _strategy(), Pipeline(), OPTS)
    np.testing.assert_allclose(res.returns, [0.0, 110 / 105 - 1, 0.1], atol=1e-12)


def test_bps_cost_charged_on_trade():
    data = make_market({"A": [100.0, 100.0, 100.0]})
    cost = registry("cost_model").get("bps_notional").create({"bps": 10})
    res = VectorizedEngine([cost]).run(config(), data, _strategy(), Pipeline(), OPTS)
    np.testing.assert_allclose(res.returns, [0.0, -0.001, 0.0], atol=1e-12)
    assert res.costs["commission"].sum() == pytest.approx(1000.0)


def test_rebalance_every_bar_vs_on_change():
    data = make_market({"A": [100, 110, 121, 133.1], "B": [100, 90, 81, 72.9]})
    hold = VectorizedEngine().run(config(), data, _strategy(), Pipeline(), OPTS)
    mix = VectorizedEngine().run(
        config(rebalance=RebalanceRule.EVERY_BAR), data, _strategy(), Pipeline(), OPTS
    )
    np.testing.assert_allclose(mix.returns[2:], 0.0, atol=1e-12)
    assert hold.returns[3] > 0  # drifted towards the winner
    assert int(hold.metadata["rebalances"]) == 1


def test_positions_and_cash_reconcile_to_equity():
    data = make_market({"A": [10.0, 11, 12, 11, 13], "B": [20.0, 19, 21, 22, 20]})
    res = VectorizedEngine().run(config(), data, _strategy(), Pipeline(), OPTS)
    held_value = (res.positions * res.prices).sum(axis=1)
    cash = res.equity * (1 - res.weights.sum(axis=1))
    np.testing.assert_allclose(held_value + cash, res.equity, rtol=MONEY_RTOL, atol=MONEY_ATOL)


@settings(max_examples=40, deadline=None)
@given(st.lists(st.floats(0.5, 2.0), min_size=5, max_size=40))
def test_zero_cost_buy_and_hold_equals_asset_return(moves):
    closes = list(np.cumprod(np.array(moves)) * 100.0)
    data = make_market({"A": closes})
    res = VectorizedEngine().run(config(), data, _strategy(), Pipeline(), OPTS)
    expected = closes[-1] / closes[1]
    assert res.equity[-1] / res.equity[0] == pytest.approx(expected, rel=1e-9)


def test_dividends_are_income():
    data = make_market({"A": [100.0, 100.0, 98.0]}, extra={C.DIVIDEND: {"A": [0.0, 0.0, 2.0]}})
    res = VectorizedEngine().run(config(), data, _strategy(), Pipeline(), OPTS)
    assert res.returns[2] == pytest.approx(0.0)
    assert res.metadata["return_basis"] == "close + dividends"


def test_trade_pnl_sums_to_gross_pnl():
    rng = np.random.default_rng(3)
    closes = list(100 * np.cumprod(1 + rng.normal(0, 0.02, 120)))
    data = make_market({"A": closes, "B": closes[::-1]})
    cfg = config(rebalance=RebalanceRule.EVERY_BAR)
    res = VectorizedEngine().run(cfg, data, _strategy(), Pipeline(), OPTS)
    gross_pnl = res.equity[-1] - res.initial_capital
    assert res.trades["pnl"].sum() == pytest.approx(gross_pnl, rel=1e-9)

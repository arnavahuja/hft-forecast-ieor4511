"""Event engine: parity with the vectorized engine, order types, lookahead protection."""

from __future__ import annotations

import numpy as np
import pytest

from hft_forecast.core import columns as C
from hft_forecast.core.errors import LookaheadError
from hft_forecast.core.interfaces import Context, Strategy
from hft_forecast.core.registry import registry
from hft_forecast.core.run_config import ExecutionPrice, RebalanceRule
from hft_forecast.core.types import MONEY_ATOL, OrderType
from hft_forecast.engine.base import RunOptions
from hft_forecast.engine.event.engine import EventEngine
from hft_forecast.engine.pipeline import NamedOverlay, Pipeline
from hft_forecast.engine.vectorized import VectorizedEngine
from tests.helpers import PPY, config, make_market, synthetic

PARITY_TOL = 1e-10
"""Documented parity tolerance on per-bar returns (zero costs)."""

DATA = synthetic(instruments=("AAA", "BBB", "CCC", "MKT"))
OPTS = RunOptions(
    periods_per_year=PPY, benchmark_id="MKT", strategy_instruments=("AAA", "BBB", "CCC")
)


def _strategy(name, **params):
    return registry("strategy").get(name).create(params)


CASES = [
    ("buy_and_hold", {}, {}),
    ("buy_and_hold", {}, {"rebalance": RebalanceRule.EVERY_BAR}),
    ("buy_and_hold", {}, {"rebalance": RebalanceRule.EVERY_BAR, "price": "open"}),
    ("buy_and_hold", {}, {"rebalance": RebalanceRule.MONTHLY, "lag_bars": 2}),
    ("buy_and_hold", {}, {"cash_rate": 0.03}),
]


@pytest.mark.parametrize(("name", "params", "execution"), CASES)
def test_parity_with_vectorized_engine(name, params, execution):
    cfg = config(name, params, **execution)
    vec = VectorizedEngine().run(cfg, DATA, _strategy(name, **params), Pipeline(), OPTS)
    evt = EventEngine().run(cfg, DATA, _strategy(name, **params), Pipeline(), OPTS)
    np.testing.assert_allclose(evt.returns, vec.returns, atol=PARITY_TOL)
    np.testing.assert_allclose(evt.equity, vec.equity, rtol=1e-9)
    np.testing.assert_allclose(evt.weights, vec.weights, atol=1e-9)
    assert evt.metadata["engine"] == "event"


def test_parity_with_overlays():
    overlays = (
        NamedOverlay(
            "vol_target", registry("overlay").get("vol_target").create({"target_vol": 0.1})
        ),
    )
    cfg = config("buy_and_hold", rebalance=RebalanceRule.EVERY_BAR)
    s = _strategy("buy_and_hold")
    vec = VectorizedEngine().run(cfg, DATA, s, Pipeline(None, overlays), OPTS)
    evt = EventEngine().run(cfg, DATA, s, Pipeline(None, overlays), OPTS)
    np.testing.assert_allclose(evt.returns, vec.returns, atol=PARITY_TOL)
    assert evt.overlay_reports[0].name == "vol_target"


def test_costs_reduce_returns_and_are_reported():
    cfg = config("buy_and_hold", rebalance=RebalanceRule.EVERY_BAR)
    cost = registry("cost_model").get("per_share").create()
    slip = registry("slippage_model").get("half_spread").create({"spread_bps": 10})
    s = _strategy("buy_and_hold")
    free = EventEngine().run(cfg, DATA, s, Pipeline(), OPTS)
    paid = EventEngine([cost], slip).run(cfg, DATA, s, Pipeline(), OPTS)
    assert paid.equity[-1] < free.equity[-1]
    assert paid.costs["commission"].sum() > 0 and paid.costs["slippage"].sum() > 0
    np.testing.assert_allclose(paid.gross_returns[1:] >= paid.returns[1:] - 1e-15, True)
    assert paid.fills.height > 0 and paid.orders.height > 0


def test_cash_plus_positions_reconcile_every_bar():
    cfg = config("buy_and_hold", rebalance=RebalanceRule.EVERY_BAR)
    res = EventEngine().run(cfg, DATA, _strategy("buy_and_hold"), Pipeline(), OPTS)
    value = (res.positions * np.nan_to_num(res.prices)).sum(axis=1)
    cash = res.equity - value
    np.testing.assert_allclose(
        res.weights.sum(axis=1) * res.equity + cash, res.equity, atol=MONEY_ATOL
    )


class StopTester(Strategy):
    """Buys once, places a stop 5% below."""

    capabilities = frozenset({"engine:event"})

    def on_bar(self, ctx: Context) -> None:
        if ctx.bar_index == 0:
            ctx.order("A", 100)
            ctx.order("A", -100, OrderType.STOP, stop_price=95.0)


def test_stop_order_fills_at_stop_or_gap():
    opens = [100, 100, 99, 97, 90, 90]
    closes = [100, 100, 98, 96, 91, 90]
    lows = [100, 99, 97, 94, 89, 89]
    data = make_market(
        {"A": closes},
        opens={"A": opens},
        extra={
            C.LOW: {"A": lows},
            C.HIGH: {"A": [max(o, c) + 1 for o, c in zip(opens, closes, strict=True)]},
        },
    )
    cfg = config()
    res = EventEngine().run(cfg, data, StopTester(), Pipeline(), RunOptions(periods_per_year=PPY))
    fills = res.fills.sort("timestamp")
    assert fills["quantity"].to_list() == [100.0, -100.0]
    assert fills["price"].to_list()[0] == 100.0  # entry at close of bar 1
    assert fills["price"].to_list()[1] == 95.0  # stop hit intrabar on bar 3 (low 94)
    assert res.positions[-1, 0] == 0


def test_limit_order_and_partial_fills():
    class LimitTester(Strategy):
        capabilities = frozenset({"engine:event"})

        def on_bar(self, ctx: Context) -> None:
            if ctx.bar_index == 0:
                ctx.order("A", 1000, OrderType.LIMIT, limit_price=95.0)

    closes = [100, 100, 96, 94, 94]
    data = make_market(
        {"A": closes},
        opens={"A": closes},
        extra={
            C.LOW: {"A": [c - 1 for c in closes]},
            C.HIGH: {"A": [c + 1 for c in closes]},
            C.VOLUME: {"A": [2000.0] * 5},
        },
    )
    cfg = config(max_participation=0.25)
    res = EventEngine().run(cfg, data, LimitTester(), Pipeline(), RunOptions(periods_per_year=PPY))
    fills = res.fills.sort("timestamp")
    assert fills["price"].to_list()[0] == 95.0  # low 95 touched on bar 2
    assert fills["quantity"].to_list() == [500.0, 500.0]  # 25% of 2000 per bar


def test_context_cannot_see_the_future():
    seen: list[int] = []

    class Peek(Strategy):
        capabilities = frozenset({"engine:event"})

        def on_bar(self, ctx: Context) -> None:
            seen.append(len(ctx.history(C.CLOSE)))
            with pytest.raises(LookaheadError):
                ctx.value_at(C.CLOSE, ctx.bar_index + 1)
            with pytest.raises(ValueError):
                ctx.history(C.CLOSE)[0, 0] = 1.0

    data = make_market({"A": [1.0, 2.0, 3.0]})
    EventEngine().run(config(), data, Peek(), Pipeline(), RunOptions(periods_per_year=PPY))
    assert seen == [1, 2, 3]


def test_open_execution_parity_uses_open_prices():
    cfg = config(price=ExecutionPrice.OPEN)
    vec = VectorizedEngine().run(cfg, DATA, _strategy("buy_and_hold"), Pipeline(), OPTS)
    evt = EventEngine().run(cfg, DATA, _strategy("buy_and_hold"), Pipeline(), OPTS)
    np.testing.assert_allclose(evt.returns, vec.returns, atol=PARITY_TOL)


def test_resampled_only_returns_completed_periods():
    from datetime import UTC, datetime, timedelta

    import polars as pl

    from hft_forecast.core.types import MarketData

    stamps = [datetime(2024, 1, 2, 15, tzinfo=UTC) + timedelta(hours=h) for h in range(3)]
    stamps += [datetime(2024, 1, 3, 15, tzinfo=UTC) + timedelta(hours=h) for h in range(3)]
    data = MarketData(
        pl.DataFrame(
            {
                "timestamp": stamps,
                "instrument_id": ["A"] * 6,
                "close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
    )
    seen = []

    class Daily(Strategy):
        capabilities = frozenset({"engine:event"})

        def on_bar(self, ctx: Context) -> None:
            seen.append(ctx.resampled(C.CLOSE, "1d")[:, 0].tolist())

    EventEngine().run(config(), data, Daily(), Pipeline(), RunOptions(periods_per_year=PPY))
    assert seen == [[], [], [], [3.0], [3.0], [3.0]]

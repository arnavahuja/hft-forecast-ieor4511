from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from hft_forecast.analytics import stats as S
from hft_forecast.core.interfaces import OverlayContext
from hft_forecast.core.registry import registry
from hft_forecast.core.types import TargetFrame
from hft_forecast.engine.base import RunOptions
from hft_forecast.engine.pipeline import NamedOverlay, Pipeline
from hft_forecast.engine.returns import compute_returns
from hft_forecast.engine.vectorized import VectorizedEngine
from tests.helpers import PPY, config, make_market, synthetic

DATA = synthetic(instruments=("AAA", "BBB", "CCC", "MKT"))
T = len(DATA.timestamps)
CFG = config()


def _ctx(data=DATA, targets_fn=None) -> OverlayContext:
    engine = VectorizedEngine()
    sim = engine.simulate_zero_cost(CFG, data, compute_returns(data), PPY, None)
    return OverlayContext(data, PPY, "MKT", 1, sim, [])


def _overlay(name, **params):
    return registry("overlay").get(name).create(params)


def _targets(values=None, data=DATA) -> TargetFrame:
    shape = (len(data.timestamps), len(data.instruments))
    vals = np.full(shape, 1.0 / shape[1]) if values is None else values
    return TargetFrame(data.timestamps, data.instruments, vals)


@settings(max_examples=40, deadline=None)
@given(
    w=arrays(np.float64, (30, 4), elements=st.floats(-3, 3)),
    cap=st.floats(0.05, 1.0),
    gross=st.floats(0.2, 3.0),
)
def test_limit_overlays_respect_caps(w, cap, gross):
    data = make_market({k: list(np.linspace(10, 11, 30)) for k in "ABCD"})
    tf = _targets(w, data)
    ctx = OverlayContext(data, PPY, None, 1, lambda t: np.zeros(30), [])
    capped = _overlay("max_position_weight", max_weight=cap).apply(tf, ctx).values
    assert np.abs(capped).max() <= cap + 1e-12
    lev = _overlay("leverage_cap", max_leverage=gross).apply(tf, ctx).values
    assert np.abs(lev).sum(axis=1).max() <= gross + 1e-9
    lim = (
        _overlay("exposure_limits", max_gross=gross, max_net=0.5, min_net=-0.5)
        .apply(tf, ctx)
        .values
    )
    assert np.abs(lim).sum(axis=1).max() <= gross + 1e-9
    net = lim.sum(axis=1)
    assert net.max() <= 0.5 + 1e-9 and net.min() >= -0.5 - 1e-9
    turn = _overlay("turnover_cap", max_turnover=0.1).apply(tf, ctx).values
    steps = np.abs(np.diff(np.vstack([np.zeros(4), turn]), axis=0)).sum(axis=1)
    assert steps.max() <= 0.2 + 1e-9


def test_vol_target_moves_realized_vol_towards_target():
    tf = _targets(np.column_stack([np.full(T, 2.0), np.zeros((T, 3))]))
    ctx = _ctx()
    out = _overlay("vol_target", target_vol=0.05, window=63).apply(tf, ctx)
    before = S.annualized_vol(ctx.simulate(tf)[200:], PPY)
    after = S.annualized_vol(ctx.simulate(out)[200:], PPY)
    assert abs(after - 0.05) < abs(before - 0.05)
    assert after == pytest.approx(0.05, rel=0.3)


def test_pipeline_records_overlay_reports_and_changes_results():
    strat = registry("strategy").get("buy_and_hold").create()
    engine = VectorizedEngine()
    opts = RunOptions(
        periods_per_year=PPY, benchmark_id="MKT", strategy_instruments=("AAA", "BBB", "CCC")
    )
    plain = engine.run(CFG, DATA, strat, Pipeline(), opts)
    overlays = (
        NamedOverlay("vol_target", _overlay("vol_target", target_vol=0.05)),
        NamedOverlay("max_position_weight", _overlay("max_position_weight", max_weight=0.2)),
    )
    with_ov = engine.run(CFG, DATA, strat, Pipeline(None, overlays), opts)
    assert [r.name for r in with_ov.overlay_reports] == ["vol_target", "max_position_weight"]
    assert with_ov.overlay_reports[1].cells_changed > 0
    assert not np.allclose(plain.returns, with_ov.returns)
    assert np.abs(with_ov.target_weights).max() <= 0.2 + 1e-12

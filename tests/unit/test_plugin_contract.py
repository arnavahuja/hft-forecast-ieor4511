"""Generic contract tests applied automatically to every registered plugin."""

from __future__ import annotations

import json

import numpy as np
import pytest

from hft_forecast.core.errors import ConfigError, DataError
from hft_forecast.core.interfaces import (
    ChartBuilder,
    ChartInput,
    ConstructionContext,
    CostContext,
    CostModel,
    Metric,
    MetricContext,
    Overlay,
    OverlayContext,
    PortfolioConstructor,
    SlippageModel,
    Strategy,
    UniverseProvider,
)
from hft_forecast.core.registry import REGISTRIES, PluginKind, PluginSpec
from hft_forecast.core.types import TargetFrame, TargetKind
from hft_forecast.engine.base import RunOptions
from hft_forecast.engine.pipeline import Pipeline
from hft_forecast.engine.vectorized import VectorizedEngine
from tests.helpers import PPY, config, synthetic, with_required_fields

ALL_SPECS = [spec for reg in REGISTRIES.values() for spec in reg.specs()]
DATA = synthetic(end=__import__("datetime").date(2019, 12, 31))


def _ids(spec) -> str:
    # pytest passes a sentinel when a kind has no plugins yet
    return f"{spec.kind}:{spec.name}" if isinstance(spec, PluginSpec) else "none"


@pytest.mark.parametrize("spec", ALL_SPECS, ids=_ids)
def test_metadata_and_params_schema(spec):
    assert spec.name and spec.version and spec.description
    schema = spec.params_schema()
    json.dumps(schema)
    assert schema.get("type") == "object"
    assert isinstance(spec.capabilities, frozenset)
    spec.create()  # instantiates with defaults


def _result():
    strat = REGISTRIES[PluginKind.STRATEGY].get("buy_and_hold").create()
    return VectorizedEngine().run(
        config(), DATA, strat, Pipeline(), RunOptions(periods_per_year=PPY)
    )


RESULT = _result()


def _kind_specs(kind: PluginKind):
    return [s for s in ALL_SPECS if s.kind is kind]


@pytest.mark.parametrize("spec", _kind_specs(PluginKind.STRATEGY), ids=_ids)
def test_strategies_run_on_fixture_data(spec):
    strat: Strategy = spec.create()
    assert spec.capabilities, "strategies must declare capabilities"
    if not strat.implements_vectorized():
        pytest.skip("event-only strategy (covered by event engine tests)")
    tf = strat.generate_targets(with_required_fields(strat, DATA))
    assert tf.values.shape[0] == len(DATA.timestamps)


@pytest.mark.parametrize("spec", _kind_specs(PluginKind.PORTFOLIO_CONSTRUCTOR), ids=_ids)
def test_constructors_run(spec):
    cons: PortfolioConstructor = spec.create()
    sig = TargetFrame(
        DATA.timestamps,
        DATA.instruments,
        np.random.default_rng(0).normal(size=(len(DATA.timestamps), 4)),
        TargetKind.SIGNALS,
    )
    out = cons.construct(sig, ConstructionContext(DATA, PPY))
    assert out.shape == sig.shape


@pytest.mark.parametrize("spec", _kind_specs(PluginKind.OVERLAY), ids=_ids)
def test_overlays_run(spec):
    overlay: Overlay = spec.create()
    if not overlay.implements_vectorized():
        pytest.skip("event-only overlay")
    tf = TargetFrame(DATA.timestamps, DATA.instruments, np.full((len(DATA.timestamps), 4), 0.25))
    ctx = OverlayContext(DATA, PPY, "AAA", 1, lambda t: np.zeros(len(DATA.timestamps)))
    try:
        out = overlay.apply(tf, ctx)
    except ConfigError as exc:
        pytest.skip(f"needs data not in the fixture: {exc.message}")
    assert out.values.shape[0] == len(DATA.timestamps)


def _cost_ctx() -> CostContext:
    n_t, n_i = RESULT.weights.shape
    return CostContext(
        timestamps=RESULT.timestamps,
        instruments=RESULT.instruments,
        trades=np.diff(RESULT.weights, axis=0, prepend=0.0),
        holdings=RESULT.weights,
        prices=RESULT.prices,
        volumes=np.full((n_t, n_i), 1e6),
        equity=RESULT.equity,
        returns=np.zeros((n_t, n_i)),
        multipliers=np.ones(n_i),
        asset_classes=("equity",) * n_i,
        periods_per_year=PPY,
    )


@pytest.mark.parametrize(
    "spec", _kind_specs(PluginKind.COST_MODEL) + _kind_specs(PluginKind.SLIPPAGE_MODEL), ids=_ids
)
def test_cost_models_run(spec):
    model: CostModel | SlippageModel = spec.create()
    out = model.vectorized(_cost_ctx())
    assert out.shape == RESULT.weights.shape
    assert np.all(np.nan_to_num(out) >= 0)


@pytest.mark.parametrize("spec", _kind_specs(PluginKind.METRIC), ids=_ids)
def test_metrics_cover_descriptors(spec):
    metric: Metric = spec.create()
    assert metric.descriptors
    ctx = MetricContext(RESULT, RESULT.returns, RESULT.benchmark_returns, PPY)
    out = metric.compute(ctx)
    assert set(out) == {d.key for d in metric.descriptors}


@pytest.mark.parametrize("spec", _kind_specs(PluginKind.CHART), ids=_ids)
def test_charts_build(spec):
    builder: ChartBuilder = spec.create()
    inputs = [ChartInput("r1", "run 1", RESULT), ChartInput("r2", "run 2", RESULT)]
    if "single" in builder.scopes:
        inputs = inputs[:1]
    if builder.applicable(inputs):
        chart = builder.build(inputs)
        json.dumps(chart.model_dump(mode="json"))


@pytest.mark.parametrize("spec", _kind_specs(PluginKind.UNIVERSE), ids=_ids)
def test_universes_return_mask(spec):
    uni: UniverseProvider = spec.create()
    try:
        mask = uni.membership(DATA)
    except DataError as exc:
        pytest.skip(f"needs data not in the fixture: {exc.message}")
    assert mask.shape == (len(DATA.timestamps), len(DATA.instruments))


def test_ml_scaffolds_produce_registered_plugins(tmp_path):
    """`hftf new model` / `new feature_set` must land in the ml registries."""
    from hft_forecast.ml.features.registry import FEATURE_SETS
    from hft_forecast.ml.models.registry import MODELS
    from hft_forecast.scaffold import scaffold
    from hft_forecast.services.plugins import PluginService

    for kind in ("model", "feature_set"):
        scaffold(kind, f"t_scaffold_{kind}", tmp_path)
    PluginService(user_dir=tmp_path).load()
    assert "t_scaffold_model" in MODELS
    assert "t_scaffold_feature_set" in FEATURE_SETS
    assert MODELS.get("t_scaffold_model").create() is not None

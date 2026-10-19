"""End-to-end runs: determinism, storage round trip, compatibility errors, CLI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from hft_forecast.core.config import Settings
from hft_forecast.core.errors import CompatibilityError, NotFoundError
from hft_forecast.core.run_config import PluginRef, RebalanceRule
from hft_forecast.services.container import Services
from hft_forecast.services.run_store import RunStatus
from tests.helpers import config


@pytest.fixture
def services(tmp_path: Path) -> Services:
    return Services.create(Settings(data_dir=tmp_path / "data", user_plugins_dir=tmp_path / "up"))


def _cfg():
    cfg = config("buy_and_hold", instruments=("AAA", "BBB"), rebalance=RebalanceRule.EVERY_BAR)
    return cfg.model_copy(
        update={
            "costs": [PluginRef(name="bps_notional", params={"bps": 5})],
            "slippage": PluginRef(name="half_spread"),
            "benchmark": "MKT",
            "experiment": "exp1",
        }
    )


def test_same_config_gives_identical_results(services):
    a = services.execute_run(_cfg())
    b = services.execute_run(_cfg())
    ra, rb = services.store.result(a.id), services.store.result(b.id)
    np.testing.assert_array_equal(ra.equity, rb.equity)
    np.testing.assert_array_equal(ra.weights, rb.weights)
    assert a.config_hash == b.config_hash
    ma, mb = services.store.metrics(a.id), services.store.metrics(b.id)
    assert ma["sharpe"].value == mb["sharpe"].value
    # the second run of the experiment counts two trials
    assert services.store.trials("exp1")[0] == 2


def test_result_round_trip_and_provenance(services):
    record = services.execute_run(_cfg())
    res = services.store.result(record.id)
    assert res.benchmark_id == "MKT"
    assert res.benchmark_returns is not None
    assert "strategy:buy_and_hold" in record.plugin_versions
    assert record.data_refs["main"]["cache_key"]
    assert record.headline["sharpe"] is not None
    assert "completed" in services.store.logs(record.id)
    assert services.store.metrics(record.id)["top_drawdowns"].table


def test_incompatible_config_is_rejected(services):
    bad = _cfg().model_copy(update={"strategy": PluginRef(name="no_such_strategy")})
    with pytest.raises(CompatibilityError) as info:
        services.execute_run(bad)
    assert info.value.details["issues"]


def test_cli_run(tmp_path: Path, monkeypatch):
    from hft_forecast.cli import app

    monkeypatch.setenv("HFTF_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HFTF_USER_PLUGINS_DIR", str(tmp_path / "none"))
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(_cfg().model_dump(mode="json")))
    result = CliRunner().invoke(app, ["run", str(path)])
    assert result.exit_code == 0, result.output
    assert "completed" in result.output


def test_ml_runs_are_stored_without_a_result_and_count_as_trials(services, tmp_path: Path):
    store = services.store
    before = store.count_trials("h30s")
    record = store.create_ml(
        name="h30s logit_ridge",
        config={"experiment": "h30s"},
        config_hash="abc123",
        experiment="h30s",
        model="logit_ridge",
        tags=["ml", "h30s"],
    )
    assert record.kind == "ml"
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    done = store.complete_ml(record.id, {"accuracy": 0.51}, manifest)
    assert done.status is RunStatus.COMPLETED
    assert done.headline["accuracy"] == 0.51
    assert store.ml_metrics(record.id)["accuracy"] == 0.51
    assert store.count_trials("h30s") == before + 1
    with pytest.raises(NotFoundError):
        store.result(record.id)  # no fabricated BacktestResult

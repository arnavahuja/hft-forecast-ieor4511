"""A feature study from the CLI, through the store, out of the API.

Small on purpose (four sessions, two names): it proves the study is registered, stored and
served, not that any feature works.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from hft_forecast.api.app import create_app
from hft_forecast.cli import app
from hft_forecast.core.config import Settings
from hft_forecast.services.container import Services

EXPERIMENT = {
    "experiment": "h1m",
    "name": "eda smoke",
    "horizon": {"name": "h1m", "seconds": 60, "grid": "5s"},
    "data": {
        "source": "synthetic",
        "dataset": "quotes",
        "instruments": ["AAA", "BBB"],
        "start": "2024-07-01",
        "end": "2024-07-04",
        "frequency": "5s",
    },
    "features": [
        {"name": "basic", "params": {"vol_windows": [12]}},
        {"name": "ofi", "params": {"windows": [1, 12]}},
        {"name": "quote_dynamics"},
    ],
    "labels": {"dead_zone_bps": 1.0},
    "cv": {"train_sessions": 2, "test_sessions": 1, "min_train_rows": 100},
    "seed": 7,
}


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HFTF_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HFTF_USER_PLUGINS_DIR", str(tmp_path / "none"))
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(EXPERIMENT))
    return path, tmp_path


@pytest.fixture
def client(tmp_path: Path):
    settings = Settings(data_dir=tmp_path / "data", user_plugins_dir=tmp_path / "none")
    services = Services.create(settings)
    with TestClient(create_app(settings, services)) as test_client:
        yield test_client


def test_study_is_registered_stored_and_served(workspace, client):
    config, tmp_path = workspace
    result = CliRunner().invoke(app, ["ml", "eda", str(config)])
    assert result.exit_code == 0, result.output
    assert "features ranked" in result.output

    listed = client.get("/api/v1/eda").json()
    assert len(listed) == 1
    record = listed[0]
    assert record["kind"] == "eda"
    assert record["status"] == "completed"
    assert record["experiment"] == "h1m"
    assert "eda" in record["tags"] and "h1m" in record["tags"]

    study = client.get(f"/api/v1/eda/{record['id']}").json()
    assert study["horizon"] == "h1m"
    assert study["horizon_seconds"] == 60
    assert study["bar_seconds"] == 5
    # the study only reads the CV training window, so it must not see all four sessions
    assert study["summary"]["sessions"] == 2
    assert study["params"]["train_sessions"] == 2
    assert study["summary"]["features"] >= 10
    assert {c["id"] for c in study["charts"]} >= {"eda_ic_ranking", "eda_coverage"}
    assert {t["feature"] for t in study["tables"]["ranking"]}
    assert len(study["tables"]["coverage"]) == 2

    exported = tmp_path / "data" / "runs" / record["id"] / "eda"
    assert (exported / "ranking.csv").exists()
    assert (exported / "blocks.csv").exists()


def test_studies_do_not_appear_in_the_backtest_run_list(workspace, client):
    config, _ = workspace
    assert CliRunner().invoke(app, ["ml", "eda", str(config)]).exit_code == 0
    assert client.get("/api/v1/runs").json() == [], "the Runs page stays strategy books"
    assert len(client.get("/api/v1/runs", params={"kind": "eda"}).json()) == 1


def test_a_study_is_not_counted_as_a_trial(workspace, client):
    """Deflation counts strategies tried, and looking at a feature is not one."""
    config, tmp_path = workspace
    assert CliRunner().invoke(app, ["ml", "eda", str(config)]).exit_code == 0
    settings = Settings(data_dir=tmp_path / "data", user_plugins_dir=tmp_path / "none")
    assert Services.create(settings).store.count_trials("h1m") == 0


def test_all_sessions_overrides_the_training_window(workspace, client):
    config, _ = workspace
    result = CliRunner().invoke(app, ["ml", "eda", str(config), "--all-sessions"])
    assert result.exit_code == 0, result.output
    record = client.get("/api/v1/eda").json()[0]
    study = client.get(f"/api/v1/eda/{record['id']}").json()
    assert study["summary"]["sessions"] == 3, "1-3 July; the 4th is a holiday"
    assert study["params"]["train_sessions"] is None


def test_the_stored_json_is_strictly_valid(workspace):
    """``NaN`` would parse in Python and fail in a browser."""
    config, tmp_path = workspace
    assert CliRunner().invoke(app, ["ml", "eda", str(config)]).exit_code == 0
    stored = next((tmp_path / "data" / "runs").glob("*/eda.json"))
    json.loads(stored.read_text(), parse_constant=_reject)


def _reject(value: str) -> None:
    raise AssertionError(f"non-JSON constant in the stored study: {value}")

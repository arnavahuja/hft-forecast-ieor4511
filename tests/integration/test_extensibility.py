"""Section 17 acceptance: a strategy added under user_plugins/ is runnable from the API (and
thus the UI) and comparable against existing runs with zero edits elsewhere."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hft_forecast.api.app import create_app
from hft_forecast.core.config import Settings
from hft_forecast.core.registry import registry
from hft_forecast.scaffold import scaffold
from hft_forecast.services.container import Services
from hft_forecast.services.jobs import JobManager
from hft_forecast.services.plugins import PluginService

API = "/api/v1"


@pytest.fixture
def user_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "user_plugins"
    scaffold("strategy", "my_ext_strategy", directory)
    scaffold("metric", "my_ext_metric", directory)
    scaffold("chart", "my_ext_chart", directory)
    yield directory
    for kind in ("strategy", "metric", "chart"):
        for name in list(registry(kind).names()):
            if name.startswith("my_ext_"):
                registry(kind).remove_module(registry(kind).get(name).module)


def _config(strategy: str, params: dict) -> dict:
    return {
        "data": {
            "source": "synthetic",
            "dataset": "gbm",
            "instruments": ["AAA", "BBB"],
            "start": "2018-01-01",
            "end": "2020-12-31",
        },
        "strategy": {"name": strategy, "params": params},
        "benchmark": "MKT",
    }


def test_user_plugins_are_runnable_and_comparable(tmp_path: Path, user_dir: Path):
    settings = Settings(data_dir=tmp_path / "data", user_plugins_dir=user_dir)
    jobs = JobManager(settings, max_workers=1, use_processes=False)
    with TestClient(create_app(services=Services.create(settings), jobs=jobs)) as client:
        plugin = client.get(f"{API}/plugins/strategy/my_ext_strategy").json()
        assert plugin["origin"] == "user"
        assert plugin["params_schema"]["properties"]["window"]["default"] == 50
        ids = []
        for name, params in (("my_ext_strategy", {"window": 20}), ("buy_and_hold", {})):
            launched = client.post(f"{API}/runs", json={"config": _config(name, params)}).json()
            job = jobs.wait(launched["job"]["id"], timeout=60)
            assert job.status.value == "completed", job.error
            ids.append(launched["run"]["id"])
        metrics = client.get(f"{API}/runs/{ids[0]}/metrics").json()
        assert metrics["values"]["my_ext_metric_big_up_days"]["value"] is not None
        charts = {c["name"] for c in client.get(f"{API}/runs/{ids[0]}/charts").json()}
        assert "my_ext_chart" in charts
        table = client.post(f"{API}/compare/metrics", json={"run_ids": ids}).json()
        assert any(r["key"] == "my_ext_metric_big_up_days" for r in table["rows"])
    jobs.shutdown()


def test_hot_reload_and_broken_plugin_health(tmp_path: Path):
    directory = tmp_path / "user_plugins"
    path = scaffold("strategy", "my_hot_strategy", directory)[0]
    service = PluginService(directory)
    service.load()
    assert registry("strategy").get("my_hot_strategy").version == "0.1.0"
    path.write_text(path.read_text().replace('version="0.1.0"', 'version="0.2.0"'))
    service.reload_user_file(path)
    assert registry("strategy").get("my_hot_strategy").version == "0.2.0"
    path.write_text("raise RuntimeError('broken on purpose')\n")
    service.reload_user_file(path)
    assert "my_hot_strategy" not in registry("strategy")
    assert any("broken on purpose" in f["error"] for f in service.health()["failures"])
    path.unlink()
    service.reload_user_file(path)
    assert service.health()["failures"] == []

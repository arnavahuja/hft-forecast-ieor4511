"""The whole path on synthetic quotes: dataset, folds, predictions, economic test, table.

Deliberately small (three sessions, two names) so it stays under a few seconds; it exists to
prove the pieces fit, not to measure anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
import yaml
from typer.testing import CliRunner

from hft_forecast.cli import app
from hft_forecast.ml.base import PREDICTION_COLUMNS

EXPERIMENT = {
    "experiment": "h30s",
    "name": "pipeline smoke",
    "horizon": {"name": "h30s", "seconds": 30, "grid": "5s"},
    "data": {
        "source": "synthetic",
        "dataset": "quotes",
        "instruments": ["AAA", "BBB"],
        "start": "2024-07-01",
        "end": "2024-07-03",
        "frequency": "5s",
    },
    "features": [{"name": "basic"}],
    "labels": {"dead_zone_bps": 1.0},
    "cv": {"train_sessions": 1, "test_sessions": 1, "min_train_rows": 100},
    "model": {"name": "logit_ridge"},
    "seed": 7,
}


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HFTF_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HFTF_USER_PLUGINS_DIR", str(tmp_path / "none"))
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(EXPERIMENT))
    return path, tmp_path


def _invoke(*args: str):
    result = CliRunner().invoke(app, list(args))
    assert result.exit_code == 0, result.output
    return result


def test_ml_pipeline_end_to_end(workspace):
    config, tmp_path = workspace
    runner = CliRunner()

    built = _invoke("ml", "build-dataset", str(config))
    assert "AAA" in built.output and "BBB" in built.output

    trained = _invoke("ml", "run", str(config))
    assert "logit_ridge@1.0.0" in trained.output
    run_id = trained.output.split("run ")[1].split()[0]

    run_dir = tmp_path / "data" / "runs" / run_id
    predictions = pl.read_parquet(run_dir / "predictions.parquet")
    assert predictions.columns == list(PREDICTION_COLUMNS)
    assert predictions["model"].n_unique() == 3  # the model and both baselines

    metrics = json.loads((run_dir / "ml_metrics.json").read_text())
    pooled = metrics["models"]["logit_ridge@1.0.0"]["pooled"]
    assert 0.0 <= pooled["accuracy"] <= 1.0
    assert metrics["n_eff"] < metrics["n_rows"]

    economic = _invoke("ml", "econ", run_id)
    assert "AAA" in economic.output and "pooled per-bar mean" in economic.output

    report = _invoke("ml", "compare", "--experiment", "h30s", "--out", str(tmp_path / "report"))
    assert "comparison.md" in report.output
    table = (tmp_path / "report" / "comparison.md").read_text()
    assert table.startswith("| metric |")
    assert "h30s / logit_ridge@1.0.0" in table
    assert "h30s / zero@1.0.0" in table  # the floor is always in the table

    # the economic runs are backtests under the same experiment, so they count as trials
    assert runner.invoke(app, ["data", "catalog"]).exit_code == 0


def test_an_imported_file_feeds_an_experiment(data_service, tmp_path):
    """The other way data arrives: a file on disk, mapped, then used as the experiment's source.

    This is the path the horizon groups take when quote or book bars show up from a venue
    rather than from the TAQ pull, so it has to reach ``build_dataset`` unchanged.
    """
    from datetime import date

    from hft_forecast.core.run_config import DataSpec
    from hft_forecast.core.types import DataRequest, Frequency
    from hft_forecast.data.importer import guess_mapping
    from hft_forecast.data.sources.synthetic import SyntheticSource
    from hft_forecast.ml.base import H5M
    from hft_forecast.ml.config import ExperimentConfig
    from hft_forecast.ml.datasets import build_dataset, load_dataset, usable_rows

    start, end = date(2024, 7, 1), date(2024, 7, 3)
    bars = SyntheticSource().fetch(
        DataRequest(
            source="synthetic",
            dataset="quotes",
            instruments=("AAA", "BBB"),
            start=start,
            end=end,
            frequency=Frequency.MIN1,
        )
    )
    path = tmp_path / "venue_quotes.parquet"
    bars.frame.write_parquet(path)
    mapping = guess_mapping(bars.frame, "venue_quotes").model_copy(
        update={"frequency": Frequency.MIN1}
    )
    record = data_service.import_file(path, mapping)

    config = ExperimentConfig(
        experiment="h5m",
        horizon=H5M,
        data=DataSpec(
            source="local_file",
            dataset=record.id,
            instruments=["AAA", "BBB"],
            start=start,
            end=end,
            frequency=Frequency.MIN1,
        ),
    )
    frame = load_dataset(build_dataset(config, data_service))
    assert usable_rows(frame).height > 0

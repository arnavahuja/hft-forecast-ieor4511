"""The fold loop: out-of-sample only, baselines alongside, and repeatable."""

from __future__ import annotations

import json

import polars as pl
import pytest

from hft_forecast.core import columns as C
from hft_forecast.core.errors import ConfigError
from hft_forecast.ml.base import PREDICTION_COLUMNS
from hft_forecast.ml.config import CVSpec
from hft_forecast.ml.train import MANIFEST_FILE, run_experiment

CV = CVSpec(train_sessions=1, test_sessions=1, min_train_rows=10)


@pytest.fixture
def config(experiment):
    return experiment.model_copy(update={"cv": CV})


@pytest.fixture
def run(config, data_service, tmp_path):
    return run_experiment(config, data_service, tmp_path / "run"), tmp_path / "run"


def test_predictions_have_the_frozen_schema_and_the_baselines(run):
    manifest, directory = run
    predictions = pl.read_parquet(directory / "predictions.parquet")
    assert predictions.columns == list(PREDICTION_COLUMNS)
    assert set(predictions["model"].unique()) == {
        "logit_ridge@1.0.0",
        "zero@1.0.0",
        "last_sign@1.0.0",
    }
    assert manifest.folds and manifest.n_rows > 0
    assert manifest.n_eff < manifest.n_rows  # overlapping labels


def test_every_prediction_is_out_of_sample(run):
    manifest, directory = run
    predictions = pl.read_parquet(directory / "predictions.parquet")
    for fold in manifest.folds:
        rows = predictions.filter(pl.col("fold") == fold.index)
        assert str(rows[C.TIMESTAMP].min()) == fold.first_test
        assert str(rows[C.TIMESTAMP].max()) == fold.last_test
    per_model = predictions.group_by("model").len()
    assert per_model["len"].n_unique() == 1  # identical rows for every model


def test_the_manifest_records_the_environment_and_hashes(run, config):
    manifest, directory = run
    saved = json.loads((directory / MANIFEST_FILE).read_text())
    assert saved["config_hash"] == config.config_hash()
    assert saved["dataset_hash"] == config.dataset_hash()
    assert saved["environment"]["packages"]["polars"]
    assert json.loads((directory / "environment.json").read_text())["seed"] == config.seed
    assert manifest.labels["n_valid"] > 0


def test_a_second_run_predicts_exactly_the_same(config, data_service, tmp_path):
    first = pl.read_parquet(run_experiment(config, data_service, tmp_path / "a").predictions)
    second = pl.read_parquet(run_experiment(config, data_service, tmp_path / "b").predictions)
    assert first.equals(second)


def test_an_impossible_cv_is_a_config_error(config, data_service, tmp_path):
    greedy = config.model_copy(update={"cv": CV.model_copy(update={"min_train_rows": 10**9})})
    with pytest.raises(ConfigError, match="min_train_rows"):
        run_experiment(greedy, data_service, tmp_path / "c")

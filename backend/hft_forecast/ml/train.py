"""The fold loop: fit, predict, write predictions and a manifest (§6.3).

Every fold is fitted from scratch, so folds are independent and a warm start can never leak
the future into the past. The run is registered in the ``RunStore`` by the caller: ``ml``
sits below ``services`` in the layer order and must not reach up into it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl
import structlog
from pydantic import BaseModel, ConfigDict, Field

from hft_forecast.core import columns as C
from hft_forecast.core.errors import ConfigError
from hft_forecast.core.types import FloatArray
from hft_forecast.data.service import DataService
from hft_forecast.ml.base import (
    DIRECTION,
    FWD_LOGRET,
    LABEL_END,
    MAGNITUDE_BPS,
    PREDICTION_COLUMNS,
    SESSION_ID,
)
from hft_forecast.ml.config import ExperimentConfig
from hft_forecast.ml.cv import Fold, effective_sample_size, split, uniqueness_weights
from hft_forecast.ml.datasets import build_dataset, design_matrix, load_dataset, usable_rows
from hft_forecast.ml.determinism import environment, seed_everything
from hft_forecast.ml.labels import label_report
from hft_forecast.ml.models.registry import MODELS

log = structlog.get_logger(__name__)

PREDICTIONS_FILE: Final = "predictions.parquet"
MANIFEST_FILE: Final = "manifest.json"
ENVIRONMENT_FILE: Final = "environment.json"
BASELINES: Final = ("zero", "last_sign")
"""Always run beside the configured model, on identical rows (§7.3)."""


class FoldSummary(BaseModel):
    """What one fold trained and tested on."""

    model_config = ConfigDict(frozen=True)

    index: int
    n_train: int
    n_test: int
    train_sessions: int
    test_sessions: int
    first_test: str
    last_test: str


class RunManifest(BaseModel):
    """The record of one ``run_experiment`` call."""

    model_config = ConfigDict(frozen=True)

    experiment: str
    model: str
    config_hash: str
    dataset_hash: str
    directory: str
    predictions: str
    n_rows: int
    n_eff: float
    folds: list[FoldSummary] = Field(default_factory=list)
    labels: dict[str, float] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    git_commit: str | None = None

    def save(self, directory: Path) -> Path:
        """Write ``manifest.json``."""
        path = directory / MANIFEST_FILE
        path.write_text(json.dumps(self.model_dump(mode="json"), indent=1, default=str))
        return path


def run_experiment(
    config: ExperimentConfig,
    data: DataService,
    out_dir: Path,
    git_commit: str | None = None,
) -> RunManifest:
    """Build the dataset, walk the folds and write the out-of-sample predictions.

    The two baselines are fitted on exactly the same rows as the configured model, so the
    comparison in the report needs no alignment.

    Raises:
        ConfigError: If no fold has enough rows to train on.
    """
    started = time.monotonic()
    seed_everything(config.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ENVIRONMENT_FILE).write_text(json.dumps(environment(config.seed), indent=1))

    directory = build_dataset(config, data)
    frame = usable_rows(load_dataset(directory, config.data.instruments))
    if frame.is_empty():
        raise ConfigError("No usable rows: every label was dropped")
    built = time.monotonic()

    matrix, columns = design_matrix(frame)
    y_dir = frame.get_column(DIRECTION).to_numpy().astype(np.int64)
    y_ret = frame.get_column(FWD_LOGRET).to_numpy()
    weights = uniqueness_weights(frame)
    names = [config.model.name, *(b for b in BASELINES if b != config.model.name)]
    models = {name: MODELS.get(name) for name in names}

    parts: list[pl.DataFrame] = []
    summaries: list[FoldSummary] = []
    for fold in split(frame, config.cv, config.horizon, config.embargo_bars):
        test_frame = frame.filter(pl.Series(fold.test))
        for name, spec in models.items():
            params = config.model.params if name == config.model.name else {}
            model = spec.create(params)
            model.set_feature_names(tuple(columns))
            model.fit(
                matrix[fold.train],
                y_dir[fold.train],
                y_ret[fold.train],
                weights[fold.train] if model.supports_sample_weight else None,
                config.seed + fold.index,
            )
            probabilities, y_hat = model.predict(matrix[fold.test])
            parts.append(_predictions(test_frame, fold, spec.label, probabilities, y_hat))
        summaries.append(_summary(fold, test_frame))
        log.info("fold_done", fold=fold.index, train=fold.n_train, test=fold.n_test)
    if not summaries:
        raise ConfigError(
            "No fold had enough training rows; lower cv.min_train_rows or widen the range"
        )

    predictions = pl.concat(parts).select(PREDICTION_COLUMNS)
    predictions.write_parquet(out_dir / PREDICTIONS_FILE)
    manifest = RunManifest(
        experiment=config.experiment,
        model=models[config.model.name].label,
        config_hash=config.config_hash(),
        dataset_hash=config.dataset_hash(),
        directory=str(out_dir),
        predictions=str(out_dir / PREDICTIONS_FILE),
        n_rows=frame.height,
        n_eff=effective_sample_size(weights),
        folds=summaries,
        labels=label_report(frame),
        timings={
            "dataset_seconds": round(built - started, 3),
            "total_seconds": round(time.monotonic() - started, 3),
        },
        environment=environment(config.seed),
        git_commit=git_commit,
    )
    manifest.save(out_dir)
    return manifest


def _predictions(
    test_frame: pl.DataFrame,
    fold: Fold,
    label: str,
    probabilities: FloatArray,
    y_hat: FloatArray,
) -> pl.DataFrame:
    return test_frame.select(C.TIMESTAMP, C.INSTRUMENT, SESSION_ID).with_columns(
        pl.lit(fold.index, dtype=pl.Int32).alias("fold"),
        pl.lit(label).alias("model"),
        pl.Series("p_down", probabilities[:, 0]),
        pl.Series("p_flat", probabilities[:, 1]),
        pl.Series("p_up", probabilities[:, 2]),
        pl.Series("y_hat_logret", y_hat),
        test_frame.get_column(FWD_LOGRET),
        test_frame.get_column(DIRECTION),
        test_frame.get_column(MAGNITUDE_BPS),
        test_frame.get_column(LABEL_END),
    )


def _summary(fold: Fold, test_frame: pl.DataFrame) -> FoldSummary:
    stamps = test_frame.get_column(C.TIMESTAMP)
    return FoldSummary(
        index=fold.index,
        n_train=fold.n_train,
        n_test=fold.n_test,
        train_sessions=len(fold.train_sessions),
        test_sessions=len(fold.test_sessions),
        first_test=str(stamps.min()),
        last_test=str(stamps.max()),
    )


def load_predictions(directory: Path) -> pl.DataFrame:
    """Read a run's predictions back."""
    return pl.read_parquet(directory / PREDICTIONS_FILE)

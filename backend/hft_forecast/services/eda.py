"""Running a feature-research study and storing it beside the runs it informs.

An EDA study is registered in the same run store as backtests and ML runs, with
``kind='eda'``. That is what makes it comparable: a study carries the config hash and the
experiment of the horizon it was computed for, so a group can point at the study its feature
choices came from, and three groups' studies sit in one list.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Final

import polars as pl
import structlog

from hft_forecast.core.errors import NotFoundError
from hft_forecast.ml.config import ExperimentConfig
from hft_forecast.ml.datasets import build_dataset, load_dataset
from hft_forecast.ml.eda import EdaParams, EdaReport, run_eda
from hft_forecast.ml.eda_charts import eda_charts
from hft_forecast.services.container import Services
from hft_forecast.services.run_store import RunRecord
from hft_forecast.services.runner import git_commit

log = structlog.get_logger(__name__)

EDA_FILE: Final = "eda.json"
EDA_TABLES_DIR: Final = "eda"
EDA_KIND: Final = "eda"


def study(config: ExperimentConfig, services: Services, params: EdaParams) -> RunRecord:
    """Build the dataset if needed, run the study, store it and register the run.

    Args:
        config: The horizon group's experiment config; its features, horizon and labels are
            what the study describes.
        services: Container holding the data service and the run store.
        params: Study knobs, taken as given. ``train_sessions=None`` means every session,
            which is a deliberate choice a caller has to make: see :func:`default_params`.

    Returns:
        The completed (or failed) run record.
    """
    resolved = params
    record = services.store.create_ml(
        name=config.name or f"{config.experiment} feature study",
        config={**config.model_dump(mode="json"), "eda": resolved.model_dump(mode="json")},
        config_hash=config.config_hash(),
        experiment=config.experiment,
        model="eda",
        tags=[EDA_KIND, config.horizon.name, *config.tags],
        notes=config.notes,
        kind=EDA_KIND,
    )
    services.store.mark_running(record.id)
    try:
        directory = build_dataset(config, services.data)
        frame = load_dataset(directory, config.data.instruments)
        report = run_eda(frame, config.horizon, resolved)
        path = save_study(services.store.run_dir(record.id), report, config, resolved)
    except Exception as exc:
        services.store.mark_failed(record.id, str(exc))
        raise
    log.info("eda_complete", run=record.id, path=str(path), **report.summary)
    return services.store.complete_ml(
        record.id,
        json_safe({"summary": report.summary, "horizon": report.horizon}),
        path,
        headline=json_safe(
            {
                "features": float(report.summary["features"]),
                "labelled_bars": float(report.summary["labelled_bars"]),
                "top_mean_ic": report.summary["top_mean_ic"],
            }
        ),
        git_commit=git_commit(),
    )


def default_params(config: ExperimentConfig) -> EdaParams:
    """The safe default: study only the sessions the CV uses for training.

    Ranking features on sessions that later serve as test folds leaks, and the leak does not
    show up in the model's own cross-validation, so the default has to be the careful one.
    """
    return EdaParams(train_sessions=config.cv.train_sessions)


def save_study(
    directory: Path, report: EdaReport, config: ExperimentConfig, params: EdaParams
) -> Path:
    """Write ``eda.json`` plus one CSV per table, and return the JSON path.

    The JSON is what the API serves; the CSVs are what goes into a write-up, and they carry
    every feature rather than the ones the charts have room for.
    """
    directory.mkdir(parents=True, exist_ok=True)
    tables = report.tables()
    payload = {
        "horizon": report.horizon,
        "horizon_seconds": report.horizon_seconds,
        "bar_seconds": report.bar_seconds,
        "experiment": config.experiment,
        "features": [f.model_dump(mode="json") for f in config.features],
        "params": params.model_dump(mode="json"),
        "summary": report.summary,
        "tables": tables,
        "charts": [c.model_dump(mode="json") for c in eda_charts(report)],
    }
    path = directory / EDA_FILE
    # allow_nan=False is the check, not the style: NaN and Infinity are not JSON, and a
    # browser's JSON.parse rejects the whole document over one of them
    path.write_text(json.dumps(json_safe(payload), indent=1, default=str, allow_nan=False))
    csv_dir = directory / EDA_TABLES_DIR
    csv_dir.mkdir(exist_ok=True)
    for name, rows in {**tables, "blocks": report.blocks.to_dicts()}.items():
        pl.DataFrame(rows).write_csv(csv_dir / f"{name}.csv")
    return path


def json_safe(value: Any) -> Any:
    """Replace every non-finite float with ``None``, recursively.

    An undefined statistic is normal here — a correlation needs two distinct values, a
    t-statistic needs a non-zero spread — and ``null`` is how the UI already renders one.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def load_study(directory: Path) -> dict[str, Any]:
    """Read a stored study back.

    Raises:
        NotFoundError: If the run has no stored study (it failed, or is not an EDA run).
    """
    path = directory / EDA_FILE
    if not path.exists():
        raise NotFoundError(f"No feature study stored for run at {directory.name}")
    loaded: dict[str, Any] = json.loads(path.read_text())
    return loaded

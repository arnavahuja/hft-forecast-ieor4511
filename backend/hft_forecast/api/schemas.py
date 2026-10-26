"""Request and response models of the HTTP API (the OpenAPI contract)."""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from hft_forecast.core.run_config import BacktestConfig
from hft_forecast.core.specs import MetricDescriptor, MetricValue
from hft_forecast.core.types import Adjustment, Frequency
from hft_forecast.data.importer import ImportMapping
from hft_forecast.services.run_store import RunRecord


class OutputModel(BaseModel):
    """Base of response models: defaulted fields are required in the output schema."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)


class PluginOut(OutputModel):
    """A plugin as described to the front end."""

    kind: str
    name: str
    version: str
    description: str
    tags: list[str]
    capabilities: list[str]
    origin: str
    module: str
    implements: list[str]
    params_schema: dict[str, Any]
    extra: dict[str, Any]


class PluginFailureOut(OutputModel):
    """A plugin module that failed to import."""

    module: str
    origin: str
    error: str
    traceback: str


class PluginHealthOut(OutputModel):
    """Plugin health panel."""

    counts: dict[str, int]
    loaded_modules: int
    failures: list[PluginFailureOut]


class DataSourceOut(OutputModel):
    """A data source with its datasets."""

    name: str
    description: str
    available: bool
    reason: str
    datasets: list[dict[str, Any]]


class PullRequest(BaseModel):
    """Pull data into the cache."""

    source: str = "yahoo"
    dataset: str = "daily"
    instruments: list[str]
    start: date
    end: date
    frequency: Frequency = Frequency.D1
    adjustment: Adjustment = Adjustment.SPLIT
    refresh: bool = False


class ImportPreviewOut(OutputModel):
    """Preview of a file to import."""

    token: str
    columns: list[str]
    dtypes: dict[str, str]
    rows: list[dict[str, Any]]
    mapping: ImportMapping


class ImportRequest(BaseModel):
    """Import a staged upload (or a local path) with a mapping."""

    token: str | None = None
    path: str | None = None
    mapping: ImportMapping
    dataset_id: str | None = None


class JobOut(OutputModel):
    """A background job."""

    id: str
    kind: str
    status: str
    progress: float
    message: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    run_id: str | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    summary: dict[str, Any] = Field(default_factory=dict)


class JobDetailOut(JobOut):
    """A job with its log lines."""

    logs: list[str] = Field(default_factory=list)


class RunCreate(BaseModel):
    """Launch a run."""

    config: BacktestConfig


class RunLaunchOut(OutputModel):
    """A queued run and its job."""

    run: RunRecord
    job: JobOut


class LabelsUpdate(BaseModel):
    """Editable run labels."""

    name: str | None = None
    tags: list[str] | None = None
    notes: str | None = None
    experiment: str | None = None
    archived: bool | None = None


class CloneRequest(BaseModel):
    """Clone a run with edits deep-merged into its config."""

    changes: dict[str, Any] = Field(default_factory=dict)
    run: bool = True


class MetricsOut(OutputModel):
    """Metric descriptors and values for one run."""

    descriptors: list[MetricDescriptor]
    values: dict[str, MetricValue]
    window: dict[str, str | None]


class ChartInfoOut(OutputModel):
    """A chart available for a scope."""

    name: str
    title: str
    group: str
    scopes: list[str]
    description: str
    applicable: bool = True


class TradesOut(OutputModel):
    """Page of trades."""

    total: int
    rows: list[dict[str, Any]]


class PositionsOut(OutputModel):
    """Positions at a bar."""

    timestamp: str
    equity: float
    rows: list[dict[str, Any]]


class CompareRequest(BaseModel):
    """Compare runs."""

    run_ids: list[str] = Field(min_length=1)
    align: bool = False


class CompareChartRequest(CompareRequest):
    """Comparison chart request."""

    params: dict[str, Any] = Field(default_factory=dict)
    max_points: int | None = None


class CombineRequest(BaseModel):
    """Combine runs into a portfolio."""

    run_ids: list[str] = Field(min_length=1)
    weights: list[float] | None = None


class SettingsOut(OutputModel):
    """Non-secret settings."""

    data_dir: str
    user_plugins_dir: str
    host: str
    port: int
    dev_mode: bool
    default_calendar: str
    default_benchmark: str
    wrds_username: str | None
    max_workers: int


class CompareRunOut(OutputModel):
    """A run in a comparison."""

    id: str
    name: str
    strategy: str


class CompareRowOut(OutputModel):
    """One metric row of a comparison table."""

    key: str
    label: str
    group: str
    format: str
    higher_is_better: bool | None
    values: dict[str, float | None]
    best: str | None
    worst: str | None


class CompareMetricsOut(OutputModel):
    """Comparison table."""

    runs: list[CompareRunOut]
    rows: list[CompareRowOut]
    warnings: list[str]
    window: dict[str, str | None]


class CombineOut(OutputModel):
    """Combined portfolio of runs."""

    metrics: dict[str, MetricValue]
    weights: list[float]
    diversification_ratio: float | None
    vol_reduction: float | None
    correlation: list[list[float]]
    equity: list[list[Any]]


class OverlayStatsOut(OutputModel):
    """What one overlay changed."""

    name: str
    position: int
    before: dict[str, float | None]
    after: dict[str, float | None]
    delta_cagr: float | None
    cells_changed: int
    mean_abs_change: float
    notes: list[str]


class DatasetPreviewOut(OutputModel):
    """Rows and per-instrument statistics of a dataset."""

    rows: list[dict[str, Any]]
    stats: dict[str, Any]


class CloneOut(OutputModel):
    """Result of cloning a run."""

    config: BacktestConfig
    run: RunRecord | None
    job: JobOut | None


class ExperimentOut(OutputModel):
    """An experiment with its run count."""

    name: str
    created_at: str
    notes: str
    runs: int


class InstrumentOut(OutputModel):
    """Instrument search result."""

    id: str
    symbol: str
    asset_class: str
    name: str | None = None
    exchange: str | None = None


class WrdsTestOut(OutputModel):
    """WRDS connection test result."""

    ok: bool
    message: str

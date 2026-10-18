"""Run and result endpoints."""

from __future__ import annotations

import json
from typing import Annotated, Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import PlainTextResponse, Response

from hft_forecast.api.deps import ServicesDep
from hft_forecast.api.schemas import (
    ChartInfoOut,
    CloneOut,
    CloneRequest,
    ExperimentOut,
    JobOut,
    LabelsUpdate,
    LookaheadOut,
    MetricsOut,
    OverlayStatsOut,
    PositionsOut,
    RunCreate,
    RunLaunchOut,
    TradesOut,
    ValidateOut,
)
from hft_forecast.core.errors import ConfigError
from hft_forecast.core.specs import ChartSpec
from hft_forecast.services.run_service import RunService
from hft_forecast.services.run_store import RunRecord

router = APIRouter(prefix="/runs", tags=["runs"])


def _svc(request: Request) -> RunService:
    svc: RunService = request.app.state.run_service
    return svc


def _params(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError("params must be a JSON object") from exc
    if not isinstance(parsed, dict):
        raise ConfigError("params must be a JSON object")
    return parsed


@router.post("/validate", response_model=ValidateOut)
def validate(body: RunCreate, request: Request) -> ValidateOut:
    """Check a configuration for compatibility problems."""
    report = _svc(request).validate(body.config)
    return ValidateOut(ok=report.ok, issues=report.issues)


@router.post("/lookahead-check", response_model=LookaheadOut)
def lookahead_check(body: RunCreate, request: Request) -> LookaheadOut:
    """Run the truncation lookahead check on the configured strategy and data."""
    report = _svc(request).lookahead(body.config)
    return LookaheadOut(
        passed=report.passed,
        message=report.message,
        cut_points=report.cut_points,
        mismatches=report.mismatches,
    )


@router.post("", response_model=RunLaunchOut, status_code=202)
def create_run(body: RunCreate, request: Request) -> RunLaunchOut:
    """Queue a run as a background job and return its id immediately."""
    record, job = _svc(request).launch(body.config)
    return RunLaunchOut(run=record, job=JobOut.model_validate(job.snapshot()))


@router.get("", response_model=list[RunRecord])
def list_runs(
    services: ServicesDep,
    search: str | None = None,
    experiment: str | None = None,
    status: str | None = None,
    tag: str | None = None,
    include_archived: bool = False,
    limit: Annotated[int, Query(le=5000)] = 500,
) -> list[RunRecord]:
    """Searchable, filterable run history (newest first)."""
    return services.store.list_runs(
        search=search,
        experiment=experiment,
        status=status,
        tag=tag,
        include_archived=include_archived,
        limit=limit,
    )


@router.get("/experiments", response_model=list[ExperimentOut])
def experiments(services: ServicesDep) -> list[dict[str, Any]]:
    """Experiments with run counts."""
    return services.store.experiments()


@router.get("/{run_id}", response_model=RunRecord)
def get_run(run_id: str, services: ServicesDep) -> RunRecord:
    """One run."""
    return services.store.get(run_id)


@router.patch("/{run_id}", response_model=RunRecord)
def update_run(run_id: str, body: LabelsUpdate, services: ServicesDep) -> RunRecord:
    """Rename, tag, annotate, move to an experiment or archive."""
    return services.store.update_labels(run_id, **body.model_dump())


@router.delete("/{run_id}", status_code=204, response_model=None)
def delete_run(run_id: str, request: Request) -> None:
    """Delete a run and its results."""
    _svc(request).delete(run_id)


@router.post("/{run_id}/clone", response_model=CloneOut)
def clone_run(run_id: str, body: CloneRequest, request: Request) -> dict[str, Any]:
    """Clone a run with config edits; launches it unless ``run`` is false."""
    config, record, job = _svc(request).clone(run_id, body.changes, body.run)
    return {
        "config": config.model_dump(mode="json"),
        "run": record.model_dump(mode="json") if record else None,
        "job": job.snapshot() if job else None,
    }


@router.post("/{run_id}/cancel", response_model=JobOut | None)
def cancel_run(run_id: str, request: Request) -> JobOut | None:
    """Cancel a queued or running run."""
    job = _svc(request).cancel(run_id)
    return JobOut.model_validate(job.snapshot()) if job else None


@router.get("/{run_id}/logs", response_class=PlainTextResponse)
def run_logs(run_id: str, services: ServicesDep) -> str:
    """Run log text."""
    services.store.get(run_id)
    return services.store.logs(run_id)


# ------------------------------------------------------------------ results


@router.get("/{run_id}/metrics", response_model=MetricsOut)
def run_metrics(
    run_id: str, services: ServicesDep, start: str | None = None, end: str | None = None
) -> MetricsOut:
    """Metrics (recomputed for ``[start, end]`` when a window is given)."""
    values = services.results.metrics(run_id, start, end)
    return MetricsOut(
        descriptors=services.results.descriptors(),
        values=values,
        window={"start": start, "end": end},
    )


@router.get("/{run_id}/charts", response_model=list[ChartInfoOut])
def run_charts(run_id: str, services: ServicesDep) -> list[ChartInfoOut]:
    """Charts available for a single run."""
    return [
        ChartInfoOut.model_validate(c) for c in services.results.available_charts(run_id, "single")
    ]


@router.get("/{run_id}/charts/{chart}", response_model=ChartSpec)
def run_chart(
    run_id: str,
    chart: str,
    services: ServicesDep,
    start: str | None = None,
    end: str | None = None,
    max_points: Annotated[int | None, Query(ge=10, le=100_000)] = None,
    params: str | None = None,
) -> ChartSpec:
    """A chart spec (time series downsampled with LTTB unless ``max_points`` is large)."""
    return services.results.chart([run_id], chart, _params(params), start, end, max_points)


@router.get("/{run_id}/trades", response_model=TradesOut)
def run_trades(
    run_id: str,
    services: ServicesDep,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=10_000)] = 500,
) -> TradesOut:
    """Trades table (paged)."""
    return TradesOut.model_validate(services.results.trades(run_id, offset, limit))


@router.get("/{run_id}/positions", response_model=PositionsOut)
def run_positions(run_id: str, services: ServicesDep, at: str | None = None) -> PositionsOut:
    """Positions at a bar (default: last)."""
    return PositionsOut.model_validate(services.results.positions(run_id, at))


@router.get("/{run_id}/overlays", response_model=list[OverlayStatsOut])
def run_overlays(run_id: str, services: ServicesDep) -> list[dict[str, Any]]:
    """Before/after statistics per overlay."""
    return services.results.overlays(run_id)


@router.get("/{run_id}/export")
def run_export(
    run_id: str,
    services: ServicesDep,
    format: str = "csv",  # noqa: A002 - query name
) -> Response:
    """Export as a CSV/Parquet zip or an HTML tearsheet."""
    content, media_type, filename = services.results.export(run_id, format)
    disposition = "inline" if media_type == "text/html" else "attachment"
    return Response(
        content,
        media_type=media_type,
        headers={"Content-Disposition": f'{disposition}; filename="{filename}"'},
    )

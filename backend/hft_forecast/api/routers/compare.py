"""Comparison endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import Response

from hft_forecast.api.deps import ServicesDep
from hft_forecast.api.schemas import (
    ChartInfoOut,
    CombineOut,
    CombineRequest,
    CompareChartRequest,
    CompareMetricsOut,
    CompareRequest,
)
from hft_forecast.core.errors import ConfigError
from hft_forecast.core.specs import ChartSpec
from hft_forecast.services.report import ReportBuilder, ReportRequest

router = APIRouter(prefix="/compare", tags=["compare"])


@router.get("/charts", response_model=list[ChartInfoOut])
def compare_charts(services: ServicesDep) -> list[ChartInfoOut]:
    """Charts available on the Compare page."""
    return [
        ChartInfoOut.model_validate(c) for c in services.results.available_charts(None, "compare")
    ]


@router.post("/metrics", response_model=CompareMetricsOut)
def compare_metrics(body: CompareRequest, services: ServicesDep) -> dict[str, Any]:
    """Metrics table with best and worst per row, plus config-difference warnings."""
    return services.results.compare_metrics(body.run_ids, body.align)


@router.post("/charts/{chart}", response_model=ChartSpec)
def compare_chart(chart: str, body: CompareChartRequest, services: ServicesDep) -> ChartSpec:
    """A comparison chart across runs."""
    start = end = None
    if body.align:
        window = services.results.common_window(body.run_ids)
        if window is None:
            raise ConfigError("Runs do not overlap in time")
        start, end = window
    return services.results.chart(body.run_ids, chart, body.params, start, end, body.max_points)


@router.post("/combine", response_model=CombineOut)
def combine(body: CombineRequest, services: ServicesDep) -> dict[str, Any]:
    """Combine runs into a fixed-weight portfolio and report the correlation benefit."""
    return services.results.combine(body.run_ids, body.weights)


@router.post(
    "/report",
    response_class=Response,
    responses={200: {"content": {"application/zip": {}}, "description": "Report zip"}},
)
def compare_report(body: ReportRequest, services: ServicesDep) -> Response:
    """CSV tables and PNG charts comparing the selected runs (zip)."""
    builder = ReportBuilder(services.store, services.results, services.runner)
    content = builder.build(body)
    return Response(
        content=content,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="comparison-report.zip"'},
    )

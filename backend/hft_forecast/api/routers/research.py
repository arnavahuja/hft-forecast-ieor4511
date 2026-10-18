"""Research endpoints: sweeps, walk-forward, Monte Carlo, sensitivity, regimes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from hft_forecast.api.deps import JobsDep, ServicesDep
from hft_forecast.api.schemas import (
    JobOut,
    MonteCarloRequest,
    ResearchResultOut,
    SensitivityRequest,
    SweepRequest,
    WalkForwardRequest,
)
from hft_forecast.services import research
from hft_forecast.services.run_store import RunRecord

router = APIRouter(prefix="/research", tags=["research"])


def _submit(jobs: JobsDep, kind: str, body: BaseModel, summary: dict[str, Any]) -> JobOut:
    job = jobs.submit(kind, body.model_dump(mode="json", exclude_none=True), summary=summary)
    return JobOut.model_validate(job.snapshot())


@router.post("/sweep", response_model=JobOut, status_code=202)
def sweep(body: SweepRequest, jobs: JobsDep) -> JobOut:
    """Grid or random parameter sweep (background job)."""
    return _submit(jobs, "sweep", body, {"strategy": body.config.strategy.name})


@router.post("/walk-forward", response_model=JobOut, status_code=202)
def walk_forward(body: WalkForwardRequest, jobs: JobsDep) -> JobOut:
    """Walk-forward optimization (background job)."""
    return _submit(jobs, "walk_forward", body, {"strategy": body.config.strategy.name})


@router.post("/monte-carlo", response_model=JobOut, status_code=202)
def monte_carlo(body: MonteCarloRequest, jobs: JobsDep) -> JobOut:
    """Monte Carlo of a stored run (background job)."""
    return _submit(jobs, "monte_carlo", body, {"run_id": body.run_id})


@router.post("/sensitivity", response_model=JobOut, status_code=202)
def sensitivity(body: SensitivityRequest, jobs: JobsDep) -> JobOut:
    """Cost, delay or capacity sensitivity (background job)."""
    return _submit(jobs, "sensitivity", body, {"kind": body.kind})


@router.get("", response_model=list[RunRecord])
def list_research(services: ServicesDep) -> list[RunRecord]:
    """Stored research results, newest first."""
    return services.store.list_runs(kind="research", include_archived=True)


@router.get("/regimes/{run_id}", response_model=ResearchResultOut)
def regimes(run_id: str, services: ServicesDep) -> dict[str, Any]:
    """Metrics by year, market trend and volatility regime for a run."""
    return research.regime_analysis(services, run_id)


@router.post("/unlock-test/{run_id}")
def unlock_test(run_id: str, services: ServicesDep) -> dict[str, Any]:
    """Return the run's config with the test period unlocked (the unlock is recorded)."""
    return research.unlock_test_period(services, run_id)


@router.get("/{run_id}", response_model=ResearchResultOut)
def get_research(run_id: str, services: ServicesDep) -> dict[str, Any]:
    """A stored research result."""
    return services.store.research(run_id)

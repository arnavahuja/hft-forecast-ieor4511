"""FastAPI dependencies giving routers access to services."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from hft_forecast.services.container import Services
from hft_forecast.services.jobs import JobManager


def get_services(request: Request) -> Services:
    """Application services stored on the app state."""
    services: Services = request.app.state.services
    return services


def get_jobs(request: Request) -> JobManager:
    """Job manager stored on the app state."""
    jobs: JobManager = request.app.state.jobs
    return jobs


ServicesDep = Annotated[Services, Depends(get_services)]
JobsDep = Annotated[JobManager, Depends(get_jobs)]

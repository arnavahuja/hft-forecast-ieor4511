"""Feature-research study endpoints (read-only).

Studies are produced by ``hftf ml eda`` and stored with the run they are registered as, so
the API only lists them and serves what was stored. Nothing here recomputes a study: the
numbers the UI shows are the ones a group can cite.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from hft_forecast.api.deps import ServicesDep
from hft_forecast.api.schemas import EdaOut
from hft_forecast.services.eda import EDA_KIND, load_study
from hft_forecast.services.run_store import RunRecord

router = APIRouter(prefix="/eda", tags=["eda"])


@router.get("", response_model=list[RunRecord])
def list_studies(
    services: ServicesDep,
    experiment: str | None = None,
    include_archived: bool = False,
    limit: Annotated[int, Query(le=5000)] = 500,
) -> list[RunRecord]:
    """Feature studies, newest first."""
    return services.store.list_runs(
        experiment=experiment,
        include_archived=include_archived,
        kind=EDA_KIND,
        limit=limit,
    )


@router.get("/{run_id}", response_model=EdaOut)
def get_study(run_id: str, services: ServicesDep) -> EdaOut:
    """One stored study: summary, tables and chart specs."""
    services.store.get(run_id)
    payload = load_study(services.store.run_dir(run_id))
    return EdaOut(run_id=run_id, **payload)

"""Data endpoints: sources, catalog, pulls, imports, previews, quality."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, Query, UploadFile

from hft_forecast.api.deps import JobsDep, ServicesDep
from hft_forecast.api.schemas import (
    DatasetPreviewOut,
    DataSourceOut,
    ImportPreviewOut,
    ImportRequest,
    InstrumentOut,
    JobOut,
    PullRequest,
)
from hft_forecast.core.errors import DataError
from hft_forecast.core.types import DataRequest
from hft_forecast.data.catalog import DatasetRecord
from hft_forecast.data.importer import ImportMapping, preview
from hft_forecast.data.validation import QualityReport

router = APIRouter(prefix="/data", tags=["data"])


@router.get("/sources", response_model=list[DataSourceOut])
def sources(services: ServicesDep) -> list[DataSourceOut]:
    """Data sources and their datasets."""
    return [DataSourceOut.model_validate(s) for s in services.data.sources()]


@router.get("/catalog", response_model=list[DatasetRecord])
def catalog(
    services: ServicesDep, source: str | None = None, kind: str | None = None
) -> list[DatasetRecord]:
    """Datasets in the catalog."""
    return services.data.catalog.records(source=source, kind=kind)


@router.post("/pull", response_model=JobOut)
def pull(body: PullRequest, jobs: JobsDep) -> JobOut:
    """Pull data into the cache as a background job."""
    request = DataRequest(
        source=body.source,
        dataset=body.dataset,
        instruments=tuple(body.instruments),
        start=body.start,
        end=body.end,
        frequency=body.frequency,
        adjustment=body.adjustment,
    )
    job = jobs.submit(
        "data_pull",
        {"request": request.model_dump(mode="json"), "refresh": body.refresh},
        summary={"source": body.source, "instruments": body.instruments[:10]},
    )
    return JobOut.model_validate(job.snapshot())


@router.post("/import/preview", response_model=ImportPreviewOut)
async def import_preview(
    services: ServicesDep,
    file: Annotated[UploadFile | None, File()] = None,
    path: Annotated[str | None, Form()] = None,
) -> ImportPreviewOut:
    """Upload a file (or point to a local path) and get a preview plus a guessed mapping."""
    if file is not None:
        staged = services.data.stage_upload(file.filename or "upload.csv", await file.read())
    elif path:
        staged = Path(path).expanduser()
        if not staged.exists():
            raise DataError(f"File not found: {path}")
    else:
        raise DataError("Provide a file or a path")
    info = preview(staged)
    token = staged.name if file is not None else str(staged)
    return ImportPreviewOut(token=token, **info)


@router.post("/import", response_model=DatasetRecord)
def import_file(body: ImportRequest, services: ServicesDep) -> DatasetRecord:
    """Import a previewed file with a (possibly corrected) mapping."""
    if body.token and not body.path and "/" not in body.token:
        path = services.data.resolve_upload(body.token)
    else:
        path = Path(body.path or body.token or "").expanduser()
        if not path.exists():
            raise DataError(f"File not found: {path}")
    return services.data.import_file(path, body.mapping, body.dataset_id)


@router.get("/datasets/{dataset_id}", response_model=DatasetRecord)
def dataset(dataset_id: str, services: ServicesDep) -> DatasetRecord:
    """One catalog record."""
    return services.data.catalog.get(dataset_id)


@router.delete("/datasets/{dataset_id}", status_code=204, response_model=None)
def delete_dataset(dataset_id: str, services: ServicesDep) -> None:
    """Delete a dataset and its files."""
    services.data.delete_dataset(dataset_id)


@router.post("/datasets/{dataset_id}/refresh", response_model=DatasetRecord)
def refresh(dataset_id: str, services: ServicesDep) -> DatasetRecord:
    """Force a re-pull of a cached dataset."""
    return services.data.refresh(dataset_id)


@router.get("/datasets/{dataset_id}/preview", response_model=DatasetPreviewOut)
def dataset_preview(
    dataset_id: str, services: ServicesDep, limit: Annotated[int, Query(le=1000)] = 100
) -> dict[str, Any]:
    """First rows and per-instrument statistics."""
    rows = services.data.catalog.preview(dataset_id, limit)
    for row in rows:
        for k, v in row.items():
            if hasattr(v, "isoformat"):
                row[k] = v.isoformat()
    stats = services.data.catalog.stats(dataset_id)
    for item in stats["instruments"]:
        for k in ("start", "end"):
            if hasattr(item[k], "isoformat"):
                item[k] = item[k].isoformat()
    return {"rows": rows, "stats": stats}


@router.get("/datasets/{dataset_id}/quality", response_model=QualityReport)
def dataset_quality(dataset_id: str, services: ServicesDep) -> QualityReport:
    """Data quality report."""
    return services.data.catalog.quality(dataset_id)


@router.get("/datasets/{dataset_id}/mapping", response_model=ImportMapping)
def dataset_mapping(dataset_id: str, services: ServicesDep) -> ImportMapping:
    """Saved import mapping (for one-click re-import)."""
    return services.data.saved_mapping(dataset_id)


@router.get("/instruments/search", response_model=list[InstrumentOut])
def search_instruments(
    services: ServicesDep, q: str, source: str = "yahoo"
) -> list[dict[str, Any]]:
    """Search instruments in a source."""
    return services.data.search_instruments(source, q)

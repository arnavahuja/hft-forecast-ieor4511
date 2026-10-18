"""Job endpoints and the progress WebSocket."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Final

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from hft_forecast.api.deps import JobsDep
from hft_forecast.api.schemas import JobDetailOut, JobOut
from hft_forecast.core.errors import NotFoundError
from hft_forecast.services.jobs import TERMINAL, JobManager

router = APIRouter(tags=["jobs"])
HEARTBEAT_SECONDS: Final = 15.0


@router.get("/jobs", response_model=list[JobOut])
def list_jobs(jobs: JobsDep) -> list[JobOut]:
    """All jobs of this server session, newest first."""
    return [JobOut.model_validate(j.snapshot()) for j in jobs.list()]


@router.get("/jobs/{job_id}", response_model=JobDetailOut)
def get_job(job_id: str, jobs: JobsDep) -> JobDetailOut:
    """One job with its log lines."""
    job = jobs.get(job_id)
    return JobDetailOut(**job.snapshot(), logs=list(job.logs))


@router.post("/jobs/{job_id}/cancel", response_model=JobOut)
def cancel_job(job_id: str, jobs: JobsDep) -> JobOut:
    """Cancel a job."""
    return JobOut.model_validate(jobs.cancel(job_id).snapshot())


@router.websocket("/ws/jobs/{job_id}")
async def job_events(websocket: WebSocket, job_id: str) -> None:
    """Stream job progress and log events until the job finishes."""
    jobs: JobManager = websocket.app.state.jobs
    await websocket.accept()
    try:
        job = jobs.get(job_id)
    except NotFoundError as exc:
        await websocket.send_json({"type": "error", "code": exc.code, "message": exc.message})
        await websocket.close()
        return
    queue = jobs.subscribe(job_id)
    try:
        await websocket.send_json(
            {"type": "snapshot", "job": job.snapshot(), "logs": list(job.logs)}
        )
        if job.status in TERMINAL:
            return
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
            except TimeoutError:
                await websocket.send_json({"type": "heartbeat"})
                continue
            await websocket.send_json(event)
            if event["job"]["status"] in {s.value for s in TERMINAL}:
                break
    except WebSocketDisconnect:
        pass
    finally:
        jobs.unsubscribe(job_id, queue)
        with contextlib.suppress(RuntimeError):
            await websocket.close()

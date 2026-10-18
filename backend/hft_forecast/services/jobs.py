"""In-process background job manager.

Long work (backtests, sweeps, data pulls) runs in a ``ProcessPoolExecutor``. Workers report
progress and log lines through a queue; a pump thread in the API process updates job state
and fans events out to WebSocket subscribers. Cancellation is cooperative: workers check a
shared flag at every progress report. A thread pool can be used instead (tests, or when
``max_workers`` is 0).
"""

from __future__ import annotations

import asyncio
import functools
import multiprocessing as mp
import queue
import threading
import traceback
import uuid
from collections import deque
from collections.abc import Callable
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Final

import structlog

from hft_forecast.core.config import Settings
from hft_forecast.core.errors import BackboneError, JobCancelledError, NotFoundError

log = structlog.get_logger(__name__)

MAX_LOG_LINES: Final = 500
PUMP_TIMEOUT_SECONDS: Final = 0.5


class JobStatus(StrEnum):
    """Job lifecycle."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL: Final = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class Job:
    """A background job."""

    id: str
    kind: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    message: str = ""
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    run_id: str | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_LOG_LINES))
    payload_summary: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "run_id": self.run_id,
            "result": self.result,
            "error": self.error,
            "summary": self.payload_summary,
        }


# ------------------------------------------------------------------ worker side

Handler = Callable[
    [Any, dict[str, Any], Callable[[float, str], None], Callable[[], bool]], dict[str, Any]
]


@functools.lru_cache(maxsize=2)
def _worker_services(settings_json: str) -> Any:
    """Services for a worker process, built once per process (memoized, not global state)."""
    from hft_forecast.core.logging import configure_logging
    from hft_forecast.services.container import Services

    settings = Settings.model_validate_json(settings_json)
    configure_logging(settings.log_level, json=settings.log_json)
    return Services.create(settings)


def run_job(
    kind: str,
    payload: dict[str, Any],
    settings_json: str,
    job_id: str,
    events: Any,
    cancel_flags: Any,
) -> dict[str, Any]:
    """Worker entry point (top-level so it can be pickled)."""
    from hft_forecast.services.job_handlers import HANDLERS

    def emit(event: dict[str, Any]) -> None:
        events.put({"job_id": job_id, **event})

    def cancelled() -> bool:
        return bool(cancel_flags.get(job_id, False))

    def progress(fraction: float, message: str) -> None:
        emit({"type": "progress", "progress": float(fraction), "message": message})
        if cancelled():
            raise JobCancelledError("Job cancelled")

    emit({"type": "started"})
    try:
        services = _worker_services(settings_json)
        handler = HANDLERS[kind]
        result = handler(services, payload, progress, cancelled)
    except JobCancelledError:
        emit({"type": "cancelled"})
        return {"status": "cancelled"}
    except BackboneError as exc:
        emit(
            {
                "type": "failed",
                "error": {"code": exc.code, "message": exc.message, "details": exc.details},
            }
        )
        return {"status": "failed"}
    except Exception as exc:
        emit(
            {
                "type": "failed",
                "error": {
                    "code": "internal_error",
                    "message": f"{type(exc).__name__}: {exc}",
                    "details": {},
                },
            }
        )
        emit({"type": "log", "line": traceback.format_exc()})
        return {"status": "failed"}
    emit({"type": "completed", "result": result})
    return {"status": "completed"}


# ------------------------------------------------------------------ manager side


class JobManager:
    """Submits jobs and tracks their state.

    Args:
        settings: Settings passed to workers.
        max_workers: Worker count; 0 uses a thread pool of one (no subprocesses).
        use_processes: Use processes (default) or threads.
    """

    def __init__(
        self, settings: Settings, max_workers: int = 2, use_processes: bool = True
    ) -> None:
        self._settings_json = settings.model_dump_json()
        self._jobs: dict[str, Job] = {}
        self._futures: dict[str, Future[dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._subscribers: dict[
            str, list[tuple[asyncio.AbstractEventLoop, asyncio.Queue[dict[str, Any]]]]
        ] = {}
        self._executor: Executor
        self._manager: Any = None
        if use_processes and max_workers > 0:
            ctx = mp.get_context("spawn")
            self._manager = ctx.Manager()
            self._events: Any = self._manager.Queue()
            self._cancel: Any = self._manager.dict()
            self._executor = ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)
        else:
            self._events = queue.Queue()
            self._cancel = {}
            self._executor = ThreadPoolExecutor(max_workers=max(max_workers, 1))
        self._stop = threading.Event()
        self._pump = threading.Thread(target=self._pump_events, name="job-pump", daemon=True)
        self._pump.start()

    # ---- public API

    def submit(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        run_id: str | None = None,
        summary: dict[str, Any] | None = None,
    ) -> Job:
        """Queue a job and return immediately."""
        job = Job(
            id=uuid.uuid4().hex[:12], kind=kind, run_id=run_id, payload_summary=dict(summary or {})
        )
        with self._lock:
            self._jobs[job.id] = job
        future = self._executor.submit(
            run_job, kind, payload, self._settings_json, job.id, self._events, self._cancel
        )
        future.add_done_callback(functools.partial(self._on_done, job.id))
        self._futures[job.id] = future
        return job

    def get(self, job_id: str) -> Job:
        """Look up a job."""
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise NotFoundError(f"Job '{job_id}' not found")
        return job

    def list(self) -> list[Job]:
        """All jobs, newest first."""
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def cancel(self, job_id: str) -> Job:
        """Request cancellation (immediate if still queued)."""
        job = self.get(job_id)
        if job.status in TERMINAL:
            return job
        self._cancel[job_id] = True
        future = self._futures.get(job_id)
        if future is not None and future.cancel():
            self._update(job_id, {"type": "cancelled"})
        return job

    def cancel_run(self, run_id: str) -> Job | None:
        """Cancel the job executing a given run, if any."""
        for job in self.list():
            if job.run_id == run_id and job.status not in TERMINAL:
                return self.cancel(job.id)
        return None

    def subscribe(self, job_id: str) -> asyncio.Queue[dict[str, Any]]:
        """Async queue receiving this job's events (call from the event loop)."""
        self.get(job_id)
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        with self._lock:
            self._subscribers.setdefault(job_id, []).append((loop, q))
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue[dict[str, Any]]) -> None:
        """Stop receiving events."""
        with self._lock:
            subs = self._subscribers.get(job_id, [])
            self._subscribers[job_id] = [(lp, qq) for lp, qq in subs if qq is not q]

    def wait(self, job_id: str, timeout: float | None = None) -> Job:
        """Block until a job finishes (for CLI and tests)."""
        future = self._futures.get(job_id)
        if future is not None:
            try:
                future.result(timeout=timeout)
            except Exception:
                log.debug("job_future_error", job_id=job_id)
        deadline = threading.Event()
        for _ in range(100):
            if self.get(job_id).status in TERMINAL:
                break
            deadline.wait(0.05)
        return self.get(job_id)

    def shutdown(self) -> None:
        """Stop the pump and the executor."""
        self._stop.set()
        self._executor.shutdown(wait=False, cancel_futures=True)
        if self._manager is not None:
            self._manager.shutdown()

    # ---- internals

    def _on_done(self, job_id: str, future: Future[dict[str, Any]]) -> None:
        if future.cancelled():
            self._update(job_id, {"type": "cancelled"})
            return
        exc = future.exception()
        if exc is not None:
            self._update(
                job_id,
                {
                    "type": "failed",
                    "error": {"code": "worker_crashed", "message": str(exc), "details": {}},
                },
            )

    def _pump_events(self) -> None:
        while not self._stop.is_set():
            try:
                event = self._events.get(timeout=PUMP_TIMEOUT_SECONDS)
            except queue.Empty:
                continue
            except (EOFError, OSError, BrokenPipeError):
                return
            self._update(event["job_id"], event)

    def _update(self, job_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            kind = event.get("type")
            if job.status in TERMINAL and kind != "log":
                return
            if kind == "started":
                job.status = JobStatus.RUNNING
                job.started_at = _now()
            elif kind == "progress":
                job.status = JobStatus.RUNNING
                job.progress = float(event.get("progress", job.progress))
                job.message = str(event.get("message", ""))
                job.logs.append(f"[{job.progress:5.0%}] {job.message}")
            elif kind == "log":
                job.logs.append(str(event.get("line", "")))
            elif kind == "completed":
                job.status = JobStatus.COMPLETED
                job.progress = 1.0
                job.result = event.get("result")
                job.finished_at = _now()
                if job.result and job.result.get("run_id"):
                    job.run_id = str(job.result["run_id"])
            elif kind == "failed":
                job.status = JobStatus.FAILED
                job.error = event.get("error")
                job.finished_at = _now()
            elif kind == "cancelled":
                job.status = JobStatus.CANCELLED
                job.finished_at = _now()
            snapshot = {
                "type": kind,
                "job": job.snapshot(),
                **({"line": event["line"]} if "line" in event else {}),
            }
            subs = list(self._subscribers.get(job_id, []))
        for loop, q in subs:
            loop.call_soon_threadsafe(q.put_nowait, snapshot)

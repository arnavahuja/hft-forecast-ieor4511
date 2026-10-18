"""Composition root: builds and wires every service from settings."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import dataclass

import structlog

from hft_forecast.analytics.service import AnalyticsService
from hft_forecast.core.config import Settings
from hft_forecast.core.errors import BackboneError, JobCancelledError
from hft_forecast.core.logging import bind_run, clear_context
from hft_forecast.core.run_config import BacktestConfig
from hft_forecast.data.service import DataPaths, DataService
from hft_forecast.services.plugins import PluginService
from hft_forecast.services.results import ResultService
from hft_forecast.services.run_store import RunRecord, RunStatus, RunStore
from hft_forecast.services.runner import BacktestRunner, git_commit

log = structlog.get_logger(__name__)

ProgressFn = Callable[[float, str], None]


@dataclass
class Services:
    """All application services, wired together."""

    settings: Settings
    plugins: PluginService
    data: DataService
    store: RunStore
    analytics: AnalyticsService
    runner: BacktestRunner
    results: ResultService

    @classmethod
    def create(cls, settings: Settings | None = None, *, load_plugins: bool = True) -> Services:
        """Build services from settings (plugins are discovered once)."""
        settings = settings or Settings()
        settings.ensure_dirs()
        plugins = PluginService(settings.user_plugins_dir)
        if load_plugins:
            plugins.load()
        data = DataService(DataPaths.under(settings.data_dir), wrds_username=settings.wrds_username)
        analytics = AnalyticsService()
        store = RunStore(settings.runs_dir)
        return cls(
            settings=settings,
            plugins=plugins,
            data=data,
            store=store,
            analytics=analytics,
            runner=BacktestRunner(data, analytics),
            results=ResultService(store, analytics),
        )

    def execute_run(
        self,
        config: BacktestConfig,
        *,
        run_id: str | None = None,
        progress: ProgressFn | None = None,
        cancelled: Callable[[], bool] | None = None,
        job_id: str | None = None,
    ) -> RunRecord:
        """Create (or reuse) a run record, execute it and store the results."""
        record = self.store.get(run_id) if run_id else self.store.create(config, job_id=job_id)
        rid = record.id
        bind_run(rid)
        self.store.mark_running(rid)
        self.store.append_log(rid, f"run {rid} started: strategy={config.strategy.name}")

        def report(fraction: float, message: str) -> None:
            self.store.append_log(rid, f"[{fraction:5.0%}] {message}")
            if progress is not None:
                progress(fraction, message)
            if cancelled is not None and cancelled():
                raise JobCancelledError("Run cancelled")

        try:
            n_trials, sharpes = self.store.trials(config.experiment)
            outcome = self.runner.run(
                config,
                progress=report,
                cancelled=cancelled,
                n_trials=n_trials,
                trial_sharpes=sharpes,
            )
            for warning in outcome.result.metadata.get("warnings", []):
                self.store.append_log(rid, f"warning: {warning}")
            record = self.store.complete(
                rid,
                outcome.result,
                outcome.metrics,
                git_commit=git_commit(),
                plugin_versions=outcome.plugin_versions,
                data_refs=outcome.data_refs,
                timings=outcome.timings,
            )
            self.store.append_log(rid, f"completed in {outcome.timings['total_seconds']}s")
            return record
        except JobCancelledError as exc:
            self.store.mark_failed(rid, str(exc), RunStatus.CANCELLED)
            self.store.append_log(rid, "cancelled")
            raise
        except BackboneError as exc:
            self.store.mark_failed(rid, exc.message)
            self.store.append_log(rid, f"error: {exc.message} {exc.details or ''}")
            raise
        except Exception as exc:
            self.store.mark_failed(rid, f"{type(exc).__name__}: {exc}")
            self.store.append_log(rid, traceback.format_exc())
            raise
        finally:
            clear_context()

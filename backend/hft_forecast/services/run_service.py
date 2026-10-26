"""Launching, cloning, cancelling and deleting runs (used by the API)."""

from __future__ import annotations

import copy
from typing import Any

from hft_forecast.core.errors import CompatibilityError
from hft_forecast.core.run_config import BacktestConfig
from hft_forecast.services.compatibility import CompatibilityReport
from hft_forecast.services.container import Services
from hft_forecast.services.jobs import Job, JobManager
from hft_forecast.services.run_store import RunRecord


def deep_merge(base: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``changes`` into a copy of ``base`` (lists are replaced)."""
    out = copy.deepcopy(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class RunService:
    """Run lifecycle operations.

    Args:
        services: Application services.
        jobs: Background job manager.
    """

    def __init__(self, services: Services, jobs: JobManager) -> None:
        self.services = services
        self.jobs = jobs

    def validate(self, config: BacktestConfig) -> CompatibilityReport:
        """Check a configuration without running it."""
        return self.services.runner.checker.check(config)

    def launch(self, config: BacktestConfig, parent_id: str | None = None) -> tuple[RunRecord, Job]:
        """Validate, create a queued run and submit it as a background job.

        Raises:
            CompatibilityError: If the configuration is invalid.
        """
        report = self.validate(config)
        if not report.ok:
            raise CompatibilityError(
                "Invalid run configuration",
                details={"issues": [i.model_dump() for i in report.issues]},
            )
        record = self.services.store.create(config, parent_id=parent_id)
        job = self.jobs.submit(
            "backtest",
            {"config": config.model_dump(mode="json"), "run_id": record.id},
            run_id=record.id,
            summary={"strategy": config.strategy.name, "name": config.name},
        )
        return record, job

    def clone(
        self, run_id: str, changes: dict[str, Any], run: bool = True
    ) -> tuple[BacktestConfig, RunRecord | None, Job | None]:
        """Clone a run's config with edits; optionally launch it."""
        record = self.services.store.get(run_id)
        merged = deep_merge(record.config, changes)
        if "name" not in changes:
            merged["name"] = f"{record.name or record.strategy} (clone)"
        config = BacktestConfig.model_validate(merged)
        if not run:
            return config, None, None
        new_record, job = self.launch(config, parent_id=run_id)
        return config, new_record, job

    def cancel(self, run_id: str) -> Job | None:
        """Cancel a queued or running run."""
        self.services.store.get(run_id)
        return self.jobs.cancel_run(run_id)

    def delete(self, run_id: str) -> None:
        """Delete a run and its stored results."""
        self.jobs.cancel_run(run_id)
        self.services.store.delete(run_id)
        self.services.results.evict(run_id)

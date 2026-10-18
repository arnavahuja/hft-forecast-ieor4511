"""Job handlers: one function per job kind, executed inside a worker.

A handler receives the worker's :class:`~hft_forecast.services.container.Services`, the JSON
payload, a ``progress(fraction, message)`` callback and a ``cancelled()`` predicate, and
returns a JSON-serializable result.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final

from hft_forecast.core.run_config import BacktestConfig
from hft_forecast.core.types import DataRequest

Progress = Callable[[float, str], None]
Cancelled = Callable[[], bool]


def backtest(
    services: Any, payload: dict[str, Any], progress: Progress, cancelled: Cancelled
) -> dict[str, Any]:
    """Execute a stored (queued) run."""
    config = BacktestConfig.model_validate(payload["config"])
    progress(0.02, "loading data")
    record = services.execute_run(
        config, run_id=payload.get("run_id"), progress=progress, cancelled=cancelled
    )
    return {"run_id": record.id, "headline": record.headline}


def data_pull(
    services: Any, payload: dict[str, Any], progress: Progress, cancelled: Cancelled
) -> dict[str, Any]:
    """Pull data into the cache and catalog."""
    request = DataRequest.model_validate(payload["request"])
    progress(0.1, f"fetching {len(request.instruments)} instruments from {request.source}")
    data = services.data.fetch(request, refresh=bool(payload.get("refresh", False)))
    progress(0.95, "registered in catalog")
    return {
        "rows": len(data),
        "instruments": list(data.instruments),
        "cache_key": data.metadata.get("cache_key"),
    }


def _research(name: str) -> Callable[..., dict[str, Any]]:
    def handler(
        services: Any, payload: dict[str, Any], progress: Progress, cancelled: Cancelled
    ) -> dict[str, Any]:
        from hft_forecast.services import research

        fn: Callable[..., dict[str, Any]] = getattr(research, name)
        return fn(services, payload, progress, cancelled)

    handler.__name__ = name
    return handler


HANDLERS: Final[dict[str, Callable[..., dict[str, Any]]]] = {
    "backtest": backtest,
    "data_pull": data_pull,
    "sweep": _research("run_sweep"),
    "walk_forward": _research("run_walk_forward"),
    "monte_carlo": _research("run_monte_carlo"),
    "sensitivity": _research("run_sensitivity"),
}
"""Job kind -> handler."""

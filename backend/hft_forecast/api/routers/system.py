"""Health, settings and connection tests."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from hft_forecast import __version__
from hft_forecast.api.deps import ServicesDep
from hft_forecast.api.schemas import SettingsOut, WrdsTestOut
from hft_forecast.core.errors import PluginNotFoundError

router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok", "version": __version__}


@router.get("/settings", response_model=SettingsOut)
def settings(services: ServicesDep) -> SettingsOut:
    """Non-secret settings."""
    s = services.settings
    return SettingsOut(
        data_dir=str(s.data_dir.resolve()),
        user_plugins_dir=str(s.user_plugins_dir.resolve()),
        host=s.host,
        port=s.port,
        dev_mode=s.dev_mode,
        default_calendar=s.default_calendar,
        default_benchmark=s.default_benchmark,
        wrds_username=s.wrds_username,
        max_workers=s.max_workers,
    )


@router.post("/settings/wrds-test", response_model=WrdsTestOut)
def wrds_test(services: ServicesDep) -> dict[str, Any]:
    """Try to connect to WRDS with the configured user name and ``~/.pgpass``."""
    try:
        source = services.data.source("wrds")
    except PluginNotFoundError:
        return {"ok": False, "message": "WRDS source is not installed"}
    ok, reason = source.available()
    if not ok:
        return {"ok": False, "message": reason}
    test = getattr(source, "test_connection", None)
    if callable(test):
        result: dict[str, Any] = test()
        return result
    return {"ok": True, "message": "WRDS source available"}

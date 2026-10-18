"""Plugin endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from hft_forecast.api.deps import ServicesDep
from hft_forecast.api.schemas import PluginHealthOut, PluginOut
from hft_forecast.core.registry import PluginKind
from hft_forecast.services.plugins import describe_spec

router = APIRouter(prefix="/plugins", tags=["plugins"])


@router.get("", response_model=list[PluginOut])
def list_plugins(services: ServicesDep) -> list[PluginOut]:
    """All registered plugins."""
    return [PluginOut.model_validate(describe_spec(s)) for s in services.plugins.specs()]


@router.get("/health", response_model=PluginHealthOut)
def plugin_health(services: ServicesDep) -> PluginHealthOut:
    """Counts per kind and import failures."""
    return PluginHealthOut.model_validate(services.plugins.health())


@router.get("/{kind}", response_model=list[PluginOut])
def list_kind(kind: PluginKind, services: ServicesDep) -> list[PluginOut]:
    """Plugins of one kind."""
    return [PluginOut.model_validate(describe_spec(s)) for s in services.plugins.specs(kind)]


@router.get("/{kind}/{name}", response_model=PluginOut)
def get_plugin(kind: PluginKind, name: str, services: ServicesDep) -> PluginOut:
    """One plugin, including the JSON Schema of its parameters and its capabilities."""
    return PluginOut.model_validate(describe_spec(services.plugins.get(kind, name)))

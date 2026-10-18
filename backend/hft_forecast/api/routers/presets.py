"""Preset CRUD."""

from __future__ import annotations

from fastapi import APIRouter

from hft_forecast.api.deps import ServicesDep
from hft_forecast.api.schemas import PresetIn
from hft_forecast.core.run_config import BacktestConfig
from hft_forecast.services.run_store import Preset

router = APIRouter(prefix="/presets", tags=["presets"])


@router.get("", response_model=list[Preset])
def list_presets(services: ServicesDep) -> list[Preset]:
    """All presets."""
    return services.store.presets()


@router.post("", response_model=Preset, status_code=201)
def save_preset(body: PresetIn, services: ServicesDep) -> Preset:
    """Create or replace a preset (the config is validated first)."""
    BacktestConfig.model_validate(body.config)
    return services.store.save_preset(body.name, body.config, body.notes)


@router.get("/{preset_id}", response_model=Preset)
def get_preset(preset_id: str, services: ServicesDep) -> Preset:
    """One preset."""
    return services.store.preset(preset_id)


@router.put("/{preset_id}", response_model=Preset)
def update_preset(preset_id: str, body: PresetIn, services: ServicesDep) -> Preset:
    """Update a preset (renames allowed)."""
    services.store.preset(preset_id)
    BacktestConfig.model_validate(body.config)
    services.store.delete_preset(preset_id)
    return services.store.save_preset(body.name, body.config, body.notes)


@router.delete("/{preset_id}", status_code=204, response_model=None)
def delete_preset(preset_id: str, services: ServicesDep) -> None:
    """Delete a preset."""
    services.store.delete_preset(preset_id)

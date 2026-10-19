"""The model registry: arrays in, arrays out, nothing else."""

from __future__ import annotations

from typing import Final

from hft_forecast.ml.base import Forecaster, MlRegistry

MODELS: Final[MlRegistry[Forecaster]] = MlRegistry("model", Forecaster)
register_model = MODELS.decorator
"""Decorator: ``@register_model(name="logit_ridge", version="1.0.0")``."""

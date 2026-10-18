"""Pydantic base classes for plugin parameters.

Every plugin declares a ``Params`` model deriving from one of these. The registry exposes
the model as JSON Schema, which the front end renders as a form.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class PluginParams(BaseModel):
    """Base class for all plugin parameter models (immutable, no extra keys)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """JSON Schema of the parameters, suitable for auto-generated forms."""
        return cls.model_json_schema()


class StrategyParams(PluginParams):
    """Parameters of a strategy."""


class ConstructorParams(PluginParams):
    """Parameters of a portfolio constructor."""


class OverlayParams(PluginParams):
    """Parameters of an overlay."""


class CostParams(PluginParams):
    """Parameters of a cost, slippage or fill model."""


class SourceParams(PluginParams):
    """Parameters of a data source (never secrets)."""


class MetricParams(PluginParams):
    """Parameters of a metric group."""


class ChartParams(PluginParams):
    """Parameters of a chart builder."""


class UniverseParams(PluginParams):
    """Parameters of a universe provider."""

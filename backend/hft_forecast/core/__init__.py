"""Core domain types, plugin interfaces and the plugin registry.

Plugin authors normally only import from here::

    from hft_forecast.core import Strategy, StrategyParams, TargetFrame, register
"""

from hft_forecast.core import columns
from hft_forecast.core.errors import BackboneError
from hft_forecast.core.interfaces import (
    ChartBuilder,
    ChartInput,
    ConstructionContext,
    CostContext,
    CostModel,
    DataRequirements,
    DataSource,
    Metric,
    MetricContext,
    Overlay,
    OverlayContext,
    PortfolioConstructor,
    SlippageModel,
    SourceEnvironment,
    Strategy,
    UniverseProvider,
)
from hft_forecast.core.params import (
    ChartParams,
    ConstructorParams,
    CostParams,
    MetricParams,
    OverlayParams,
    PluginParams,
    SourceParams,
    StrategyParams,
    UniverseParams,
)
from hft_forecast.core.registry import PluginKind, register, registry
from hft_forecast.core.types import (
    AssetClass,
    Frequency,
    Instrument,
    MarketData,
    TargetFrame,
    TargetKind,
)

__all__ = [
    "AssetClass",
    "BackboneError",
    "ChartBuilder",
    "ChartInput",
    "ChartParams",
    "ConstructionContext",
    "ConstructorParams",
    "CostContext",
    "CostModel",
    "CostParams",
    "DataRequirements",
    "DataSource",
    "Frequency",
    "Instrument",
    "MarketData",
    "Metric",
    "MetricContext",
    "MetricParams",
    "Overlay",
    "OverlayContext",
    "OverlayParams",
    "PluginKind",
    "PluginParams",
    "PortfolioConstructor",
    "SlippageModel",
    "SourceEnvironment",
    "SourceParams",
    "Strategy",
    "StrategyParams",
    "TargetFrame",
    "TargetKind",
    "UniverseParams",
    "UniverseProvider",
    "columns",
    "register",
    "registry",
]

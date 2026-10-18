"""Interfaces for every plugin kind.

Plugins depend only on this module (plus ``core`` value types). All plugins are
instantiated with a validated ``Params`` model: ``SomePlugin(params)``.

Capability vocabulary (see design doc section 5.3):
    ``engine:vectorized``, ``engine:event``, ``asset:equity``, ``asset:future``,
    ``asset:option``, ``freq:daily``, ``freq:intraday``, ``needs:fundamentals``,
    ``needs:options_chain``, ``needs:benchmark``, ``needs:multi_asset``, ``supports:short``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Final, Protocol, runtime_checkable

import numpy as np
import polars as pl

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
from hft_forecast.core.registry import PluginKind
from hft_forecast.core.results import BacktestResult
from hft_forecast.core.specs import ChartSpec, MetricDescriptor
from hft_forecast.core.types import (
    DataRequest,
    DatasetInfo,
    FloatArray,
    Instrument,
    MarketData,
    Order,
    OrderType,
    PortfolioState,
    TargetFrame,
    TargetKind,
    TimeArray,
    TimeInForce,
)

ENGINE_VECTORIZED: Final = "engine:vectorized"
ENGINE_EVENT: Final = "engine:event"
FREQ_DAILY: Final = "freq:daily"
FREQ_INTRADAY: Final = "freq:intraday"
# Sub-minute bars (``Frequency.S1`` .. ``S30``); a plugin without it is rejected at config time.
FREQ_SUBMINUTE: Final = "freq:subminute"
ASSET_EQUITY: Final = "asset:equity"
SUPPORTS_SHORT: Final = "supports:short"
NEEDS_BENCHMARK: Final = "needs:benchmark"
NEEDS_MULTI_ASSET: Final = "needs:multi_asset"


class Plugin:
    """Common base for all plugins.

    Attributes:
        Params: Pydantic parameter model.
        capabilities: Declared capabilities.
        params: The validated parameters of this instance.
    """

    Params: ClassVar[type[PluginParams]] = PluginParams
    capabilities: ClassVar[frozenset[str]] = frozenset()
    plugin_name: ClassVar[str] = ""
    plugin_version: ClassVar[str] = ""

    def __init__(self, params: PluginParams | None = None) -> None:
        self.params = params if params is not None else self.Params()


# --------------------------------------------------------------------------- data sources


@dataclass(frozen=True)
class SourceEnvironment:
    """Non-secret environment handed to data sources.

    Attributes:
        data_dir: Data directory.
        imports_dir: Directory for imported local datasets.
        wrds_username: WRDS user name (password is read from ``~/.pgpass`` by the library).
    """

    data_dir: Path
    imports_dir: Path
    wrds_username: str | None = None


class DataSource(Plugin, ABC):
    """Fetches data from somewhere and maps it to the canonical schema."""

    Params: ClassVar[type[PluginParams]] = SourceParams
    source_version: ClassVar[str] = "1"

    def __init__(
        self, params: PluginParams | None = None, env: SourceEnvironment | None = None
    ) -> None:
        super().__init__(params)
        self.env = env or SourceEnvironment(Path("data"), Path("data/imports"))

    @abstractmethod
    def describe(self) -> list[DatasetInfo]:
        """Datasets offered by this source."""

    @abstractmethod
    def search_instruments(self, query: str) -> list[Instrument]:
        """Search instruments by symbol or name."""

    @abstractmethod
    def fetch(self, request: DataRequest) -> MarketData:
        """Fetch data for a request in the canonical long format."""

    def available(self) -> tuple[bool, str]:
        """Whether the source can currently be used, with a reason if not."""
        return True, ""


# --------------------------------------------------------------------------- strategies


@dataclass(frozen=True)
class DataRequirements:
    """What a strategy needs from the data layer.

    Attributes:
        fields: Required fields (defaults to close).
        lookback: Bars of history needed before the first valid decision.
        needs_benchmark: Whether the benchmark series must be present in the data.
        extra: Free-form requirements (e.g. ``{"dataset": "fundamentals"}``).
    """

    fields: tuple[str, ...] = ("close",)
    lookback: int = 0
    needs_benchmark: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Context(Protocol):
    """Per-bar context given to ``on_bar`` in the event engine.

    Implementations must never expose data after the current bar; attempting to do so raises
    :class:`~hft_forecast.core.errors.LookaheadError`.
    """

    @property
    def now(self) -> np.datetime64:
        """Timestamp of the current bar."""
        ...

    @property
    def bar_index(self) -> int:
        """0-based index of the current bar."""
        ...

    @property
    def instruments(self) -> tuple[str, ...]:
        """Instrument ids in column order."""
        ...

    @property
    def portfolio(self) -> PortfolioState:
        """Current portfolio state."""
        ...

    @property
    def state(self) -> dict[str, Any]:
        """Scratch storage that persists across bars for this plugin instance."""
        ...

    @property
    def periods_per_year(self) -> float:
        """Annualization factor of the bars."""
        ...

    def history(self, field_name: str, lookback: int | None = None) -> FloatArray:
        """Panel ``(bars, instruments)`` up to and including the current bar."""
        ...

    def current(self, field_name: str) -> FloatArray:
        """Values of a field for the current bar ``(instruments,)``."""
        ...

    def timestamps(self, lookback: int | None = None) -> TimeArray:
        """Timestamps up to and including the current bar."""
        ...

    def value_at(self, field_name: str, bar_index: int) -> FloatArray:
        """Values at a past bar; raises ``LookaheadError`` for a future bar."""
        ...

    def portfolio_returns(self, lookback: int | None = None) -> FloatArray:
        """Realized portfolio returns up to the current bar."""
        ...

    def resampled(self, field_name: str, frequency: str, lookback: int | None = None) -> FloatArray:
        """Completed bars of a coarser frequency (e.g. daily from intraday), ``(bars, N)``.

        Only periods that ended before the current bar's period are included.
        """
        ...

    def order(
        self,
        instrument: str,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
        tif: TimeInForce = TimeInForce.GTC,
        tag: str = "",
    ) -> str:
        """Submit an order. Positive quantity buys, negative sells. Returns the order id."""
        ...

    def cancel(self, order_id: str) -> None:
        """Cancel an open order."""
        ...

    def open_orders(self) -> list[Order]:
        """Open orders."""
        ...

    def set_target_weights(self, weights: dict[str, float]) -> None:
        """Rebalance towards target weights (unlisted instruments go to zero)."""
        ...

    def target_weights(self) -> dict[str, float]:
        """Target weights set so far during this bar (used by overlays)."""
        ...


class Strategy(Plugin):
    """Produces signals or target weights (intent), never sizing or risk decisions.

    Implement :meth:`generate_targets` (vectorized, whole history), :meth:`on_bar`
    (event-driven), or both, and declare it through ``capabilities``.
    """

    Params: ClassVar[type[PluginParams]] = StrategyParams
    capabilities: ClassVar[frozenset[str]] = frozenset({ENGINE_VECTORIZED, ENGINE_EVENT})
    output_kind: ClassVar[TargetKind] = TargetKind.WEIGHTS

    def data_requirements(self) -> DataRequirements:
        """Data this strategy needs."""
        return DataRequirements(lookback=self.warmup())

    def warmup(self) -> int:
        """Bars of history needed before the first meaningful decision."""
        return 0

    def generate_targets(self, data: MarketData) -> TargetFrame:
        """Targets for every bar using only information up to that bar."""
        raise NotImplementedError

    def on_bar(self, ctx: Context) -> None:
        """Handle one bar in the event engine."""
        raise NotImplementedError

    @classmethod
    def implements_vectorized(cls) -> bool:
        """Whether :meth:`generate_targets` is implemented."""
        return cls.generate_targets is not Strategy.generate_targets

    @classmethod
    def implements_event(cls) -> bool:
        """Whether :meth:`on_bar` is implemented."""
        return cls.on_bar is not Strategy.on_bar


# --------------------------------------------------------------------------- pipeline


@dataclass(frozen=True)
class ConstructionContext:
    """Context for portfolio constructors."""

    data: MarketData
    periods_per_year: float


class PortfolioConstructor(Plugin, ABC):
    """Turns signals into target weights, using only past information at each row."""

    Params: ClassVar[type[PluginParams]] = ConstructorParams

    @abstractmethod
    def construct(self, signals: TargetFrame, ctx: ConstructionContext) -> TargetFrame:
        """Return target weights."""


@dataclass(frozen=True)
class OverlayContext:
    """Context for overlays in the vectorized pipeline.

    Attributes:
        data: Market data (the overlay must use rows ``<= t`` for the decision at ``t``).
        periods_per_year: Annualization factor.
        benchmark: Benchmark instrument id, if configured.
        lag_bars: Execution lag used by the engine.
        simulate: Zero-cost simulated returns of a target frame on the data grid.
            ``simulate(targets)[t]`` is the return over bar ``t``; it depends on targets
            decided at ``t - lag_bars`` and earlier only.
        notes: Overlays append human-readable notes here.
        initial_capital: Starting equity (for overlays that need currency amounts).
    """

    data: MarketData
    periods_per_year: float
    benchmark: str | None
    lag_bars: int
    simulate: Callable[[TargetFrame], FloatArray]
    notes: list[str] = field(default_factory=list)
    initial_capital: float = 1_000_000.0


class Overlay(Plugin):
    """Modifies proposed targets for risk control or hedging.

    Implement :meth:`apply` (vectorized), :meth:`on_bar` (event), or both.
    """

    Params: ClassVar[type[PluginParams]] = OverlayParams
    capabilities: ClassVar[frozenset[str]] = frozenset({ENGINE_VECTORIZED})

    def apply(self, targets: TargetFrame, ctx: OverlayContext) -> TargetFrame:
        """Return modified targets."""
        raise NotImplementedError

    def on_bar(self, ctx: Context) -> None:
        """Adjust ``ctx.target_weights()`` in the event engine."""
        raise NotImplementedError

    @classmethod
    def implements_vectorized(cls) -> bool:
        """Whether :meth:`apply` is implemented."""
        return cls.apply is not Overlay.apply

    @classmethod
    def implements_event(cls) -> bool:
        """Whether :meth:`on_bar` is implemented."""
        return cls.on_bar is not Overlay.on_bar


# --------------------------------------------------------------------------- costs & fills


@dataclass(frozen=True)
class CostContext:
    """Inputs for vectorized cost and slippage models. All weights are fractions of equity.

    Attributes:
        timestamps: Bar timestamps ``(T,)``.
        instruments: Instrument ids ``(N,)``.
        trades: Signed weight traded at each bar ``(T, N)``.
        holdings: Weights held over each bar (start-of-bar, after drift) ``(T, N)``.
        prices: Execution prices ``(T, N)``.
        volumes: Bar volumes ``(T, N)`` or ``None``.
        equity: Equity at each bar before costs ``(T,)``.
        returns: Close-to-close instrument returns ``(T, N)`` (for volatility estimates;
            models must only use rows ``<= t`` at bar ``t``).
        multipliers: Contract multipliers ``(N,)``.
        asset_classes: Asset class per instrument ``(N,)``.
        periods_per_year: Annualization factor.
    """

    timestamps: TimeArray
    instruments: tuple[str, ...]
    trades: FloatArray
    holdings: FloatArray
    prices: FloatArray
    volumes: FloatArray | None
    equity: FloatArray
    returns: FloatArray
    multipliers: FloatArray
    asset_classes: tuple[str, ...]
    periods_per_year: float

    def traded_units(self) -> FloatArray:
        """Absolute units (shares/contracts) traded at each bar."""
        with np.errstate(divide="ignore", invalid="ignore"):
            units = (
                np.abs(self.trades)
                * self.equity[:, None]
                / (self.prices * self.multipliers[None, :])
            )
        return np.nan_to_num(units, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass(frozen=True)
class BarSnapshot:
    """One instrument's bar as seen by fill and slippage models in the event engine."""

    timestamp: np.datetime64
    open: float
    high: float
    low: float
    close: float
    volume: float
    volatility: float = float("nan")


class CostModel(Plugin):
    """Commissions, fees, borrow and financing costs.

    Attributes:
        category: Cost category used for reporting.
    """

    Params: ClassVar[type[PluginParams]] = CostParams
    category: ClassVar[str] = "commission"

    def vectorized(self, ctx: CostContext) -> FloatArray:
        """Cost at each bar as a fraction of equity ``(T, N)`` (positive = cost)."""
        return np.zeros_like(ctx.trades)

    def commission(self, quantity: float, price: float, instrument: Instrument) -> float:
        """Commission in currency for a fill of ``quantity`` units (signed) at ``price``."""
        return 0.0

    def holding_cost(
        self, market_value: float, instrument: Instrument, year_fraction: float
    ) -> float:
        """Cost in currency of holding a position of ``market_value`` for ``year_fraction``."""
        return 0.0


class SlippageModel(Plugin):
    """Price impact and spread costs."""

    Params: ClassVar[type[PluginParams]] = CostParams

    def vectorized(self, ctx: CostContext) -> FloatArray:
        """Slippage at each bar as a fraction of equity ``(T, N)``."""
        return np.zeros_like(ctx.trades)

    def price_adjustment(self, quantity: float, price: float, bar: BarSnapshot) -> float:
        """Adverse per-unit price adjustment (>= 0) for an order of ``quantity`` units."""
        return 0.0


class FillModel(Plugin):
    """Decides when and at what price an order fills in the event engine."""

    Params: ClassVar[type[PluginParams]] = CostParams
    capabilities: ClassVar[frozenset[str]] = frozenset({ENGINE_EVENT})

    @abstractmethod
    def fill(self, order: Order, bar: BarSnapshot, max_quantity: float) -> tuple[float, float]:
        """Return ``(price, quantity)`` filled on ``bar``; quantity 0 means no fill.

        ``max_quantity`` is the unsigned cap from the volume participation limit.
        """


# --------------------------------------------------------------------------- analytics


@dataclass(frozen=True)
class MetricContext:
    """Inputs for metric computation.

    Attributes:
        result: The backtest result (for the benchmark pass: the strategy's result).
        returns: Return series being measured (strategy or benchmark).
        benchmark_returns: Benchmark returns (``None`` if unavailable or on benchmark pass).
        periods_per_year: Annualization factor.
        is_benchmark: True when computing the benchmark column.
        risk_free_rate: Annual risk-free rate.
        factors: Optional factor returns frame (``timestamp`` + factor columns).
        n_trials: Number of trials in the experiment (for deflated Sharpe).
        trial_sharpes: Sharpe ratios of all trials in the experiment (per period).

    The per-bar risk-free series, when the run has one, is ``result.risk_free``; use
    :meth:`excess_returns` for Sharpe-type ratios.
    """

    result: BacktestResult
    returns: FloatArray
    benchmark_returns: FloatArray | None
    periods_per_year: float
    is_benchmark: bool = False
    risk_free_rate: float = 0.0
    factors: pl.DataFrame | None = None
    n_trials: int = 1
    trial_sharpes: tuple[float, ...] = ()

    def excess_returns(self) -> tuple[FloatArray, float]:
        """Returns for Sharpe-type ratios and the annual rate still to subtract.

        With a per-bar risk-free series the returns are net of it (rate 0); otherwise the
        raw returns and the constant annual ``risk_free_rate``.
        """
        rf = self.result.risk_free
        if rf is not None and len(rf) == len(self.returns):
            return self.returns - np.nan_to_num(rf, nan=0.0), 0.0
        return self.returns, self.risk_free_rate


MetricOutput = float | None | list[dict[str, Any]]


class Metric(Plugin, ABC):
    """A group of related metrics.

    Attributes:
        descriptors: Outputs of this metric group.
    """

    Params: ClassVar[type[PluginParams]] = MetricParams
    descriptors: ClassVar[tuple[MetricDescriptor, ...]] = ()

    @abstractmethod
    def compute(self, ctx: MetricContext) -> dict[str, MetricOutput]:
        """Compute every descriptor's value (``None`` when undefined)."""


@dataclass(frozen=True)
class ChartInput:
    """One run as seen by a chart builder."""

    run_id: str
    label: str
    result: BacktestResult
    metrics: dict[str, float | None] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


class ChartBuilder(Plugin, ABC):
    """Builds a chart spec from one or more runs.

    Attributes:
        title: Chart title.
        group: UI group (Performance, Drawdown, ...).
        scopes: ``single`` (run detail) and/or ``compare`` (compare page).
    """

    Params: ClassVar[type[PluginParams]] = ChartParams
    title: ClassVar[str] = ""
    group: ClassVar[str] = "Performance"
    scopes: ClassVar[frozenset[str]] = frozenset({"single"})

    @abstractmethod
    def build(self, runs: Sequence[ChartInput]) -> ChartSpec:
        """Return the chart spec."""

    def applicable(self, runs: Sequence[ChartInput]) -> bool:
        """Whether the chart has something to show for these runs."""
        return len(runs) > 0


# --------------------------------------------------------------------------- universes


class UniverseProvider(Plugin, ABC):
    """Point-in-time list of tradable instruments."""

    Params: ClassVar[type[PluginParams]] = UniverseParams

    def candidates(self, configured: list[str]) -> list[str]:
        """All instruments that may ever be members (these are fetched)."""
        return configured

    def request_universe(self) -> str | None:
        """Source-side universe code to request (e.g. ``"sp500"``), if any."""
        return None

    @abstractmethod
    def membership(self, data: MarketData) -> FloatArray:
        """Boolean-valued ``(T, N)`` float mask: 1.0 where tradable at ``t`` (point-in-time)."""


BASE_FOR_KIND: Final[dict[PluginKind, type[Plugin]]] = {
    PluginKind.DATA_SOURCE: DataSource,
    PluginKind.STRATEGY: Strategy,
    PluginKind.PORTFOLIO_CONSTRUCTOR: PortfolioConstructor,
    PluginKind.OVERLAY: Overlay,
    PluginKind.COST_MODEL: CostModel,
    PluginKind.SLIPPAGE_MODEL: SlippageModel,
    PluginKind.FILL_MODEL: FillModel,
    PluginKind.METRIC: Metric,
    PluginKind.CHART: ChartBuilder,
    PluginKind.UNIVERSE: UniverseProvider,
}
"""Base class each plugin kind must subclass."""

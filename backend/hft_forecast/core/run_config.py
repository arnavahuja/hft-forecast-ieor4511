"""Run configuration: everything needed to reproduce a backtest.

``BacktestConfig`` lives in ``core`` (not ``engine``) because plugins, analytics and services
all read it and ``core`` is the only package everyone may depend on.
``hft_forecast.engine.base`` re-exports it.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hft_forecast.core.types import Adjustment, DataRequest, Frequency

DEFAULT_CAPITAL = 1_000_000.0
DEFAULT_SEED = 42


class PluginRef(BaseModel):
    """Reference to a plugin plus its parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    version: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class DataSpec(BaseModel):
    """Which data a run uses."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = "yahoo"
    dataset: str = "daily"
    instruments: list[str] = Field(default_factory=list)
    universe: PluginRef | None = None
    start: date
    end: date
    frequency: Frequency = Frequency.D1
    adjustment: Adjustment = Adjustment.SPLIT
    fields: list[str] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> DataSpec:
        if self.end < self.start:
            raise ValueError("end must not be before start")
        return self

    def to_request(
        self, instruments: list[str] | None = None, universe: str | None = None
    ) -> DataRequest:
        """Build a :class:`DataRequest` for the given (or configured) instruments."""
        return DataRequest(
            source=self.source,
            dataset=self.dataset,
            universe=universe,
            instruments=tuple(instruments if instruments is not None else self.instruments),
            start=self.start,
            end=self.end,
            frequency=self.frequency,
            fields=tuple(self.fields),
            adjustment=self.adjustment,
            options=dict(self.options),
        )


class ExecutionPrice(StrEnum):
    """Price at which the vectorized engine trades."""

    CLOSE = "close"
    OPEN = "open"


class RebalanceRule(StrEnum):
    """When the portfolio is rebalanced to the latest targets.

    ``on_change`` trades only when the targets differ from the previous decision, so a
    constant target is bought once and then drifts (buy and hold). ``every_bar`` restores the
    target weights on every bar (constant-mix).
    """

    ON_CHANGE = "on_change"
    EVERY_BAR = "every_bar"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    EVERY_N = "every_n"


class ExecutionSpec(BaseModel):
    """Engine and execution settings.

    Attributes:
        engine: Engine name (``vectorized`` or ``event``).
        lag_bars: Bars between the decision bar and the trade bar. Must be >= 1 so a decision
            made at the close of ``t`` can never trade at that same close.
        price: Trade at the close or the open of the trade bar.
        rebalance: Rebalance schedule.
        rebalance_every_n: Bars between rebalances for ``every_n``.
        cash_rate: Annual interest earned on positive cash (and paid on negative cash).
        max_participation: Max fraction of bar volume an order may take (event engine).
        latency_bars: Extra bars between order submission and eligibility (event engine).
        calendar: Exchange calendar code used for annualization.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    engine: str = "vectorized"
    lag_bars: int = Field(1, ge=1, le=20)
    price: ExecutionPrice = ExecutionPrice.CLOSE
    rebalance: RebalanceRule = RebalanceRule.ON_CHANGE
    rebalance_every_n: int = Field(1, ge=1)
    cash_rate: float = Field(0.0, ge=-0.05, le=0.5)
    max_participation: float = Field(1.0, gt=0.0, le=1.0)
    latency_bars: int = Field(0, ge=0, le=20)
    calendar: str = "XNYS"


class ResearchSplit(BaseModel):
    """Train/validation/test split. The test period is locked until explicitly unlocked."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    train_end: date | None = None
    validation_end: date | None = None
    test_unlocked: bool = False


class BacktestConfig(BaseModel):
    """Full, serializable description of a run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = ""
    data: DataSpec
    strategy: PluginRef
    constructor: PluginRef | None = None
    overlays: list[PluginRef] = Field(default_factory=list)
    costs: list[PluginRef] = Field(default_factory=list)
    slippage: PluginRef | None = None
    fill_model: PluginRef | None = None
    execution: ExecutionSpec = Field(default_factory=ExecutionSpec)
    benchmark: str | None = None
    risk_free: str | None = Field(
        None,
        description="'source:dataset[:field]' of per-period risk-free returns: cash earns it "
        "and Sharpe-type metrics use excess returns",
    )
    initial_capital: float = Field(DEFAULT_CAPITAL, gt=0)
    seed: int = DEFAULT_SEED
    split: ResearchSplit | None = None
    tags: list[str] = Field(default_factory=list)
    notes: str = ""
    experiment: str | None = None

    def reproducibility_payload(self) -> dict[str, Any]:
        """Fields that determine results (excludes labels such as name, tags, notes)."""
        return self.model_dump(mode="json", exclude={"name", "tags", "notes", "experiment"})

    def config_hash(self) -> str:
        """Stable hash of the result-determining configuration."""
        blob = json.dumps(self.reproducibility_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

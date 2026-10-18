"""Commission and fee models."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from pydantic import BaseModel, Field

from hft_forecast.core.interfaces import ENGINE_EVENT, ENGINE_VECTORIZED, CostContext, CostModel
from hft_forecast.core.params import CostParams
from hft_forecast.core.registry import register
from hft_forecast.core.results import COST_COMMISSION, COST_FEES
from hft_forecast.core.types import AssetClass, FloatArray, Instrument

BPS = 1e-4
BOTH_ENGINES = frozenset({ENGINE_VECTORIZED, ENGINE_EVENT})


def _per_equity(currency: FloatArray, equity: FloatArray) -> FloatArray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = currency / equity[:, None]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


class ZeroParams(CostParams):
    """No parameters."""


@register("cost_model", name="zero", version="1.0.0", tags=["basic"])
class ZeroCost(CostModel):
    """No costs at all (useful for parity tests and upper bounds)."""

    Params = ZeroParams
    capabilities: ClassVar[frozenset[str]] = BOTH_ENGINES


class FixedParams(CostParams):
    """Fixed fee per order."""

    fee: float = Field(1.0, ge=0, description="Currency per order")


@register("cost_model", name="fixed_per_trade", version="1.0.0", tags=["commission"])
class FixedPerTrade(CostModel):
    """A fixed fee for every order (every instrument traded at a bar)."""

    Params = FixedParams
    params: FixedParams
    capabilities: ClassVar[frozenset[str]] = BOTH_ENGINES
    category: ClassVar[str] = COST_COMMISSION

    def vectorized(self, ctx: CostContext) -> FloatArray:
        """Fee where a trade happens."""
        traded = (np.abs(ctx.trades) > 0).astype(np.float64) * self.params.fee
        return _per_equity(traded, ctx.equity)

    def commission(self, quantity: float, price: float, instrument: Instrument) -> float:
        """Fixed fee."""
        return self.params.fee if quantity else 0.0


class PerShareParams(CostParams):
    """Per-share commission."""

    per_share: float = Field(0.005, ge=0, description="Currency per share")
    minimum: float = Field(1.0, ge=0, description="Minimum per order")
    max_pct_notional: float = Field(0.01, ge=0, le=1, description="Cap as fraction of notional")


@register("cost_model", name="per_share", version="1.0.0", tags=["commission"])
class PerShare(CostModel):
    """Per-share commission with a per-order minimum and a notional cap (IB-style)."""

    Params = PerShareParams
    params: PerShareParams
    capabilities: ClassVar[frozenset[str]] = BOTH_ENGINES
    category: ClassVar[str] = COST_COMMISSION

    def _fee(self, units: FloatArray, notional: FloatArray) -> FloatArray:
        p = self.params
        fee = np.maximum(units * p.per_share, p.minimum)
        fee = np.minimum(fee, notional * p.max_pct_notional)
        return np.where(units > 0, fee, 0.0)

    def vectorized(self, ctx: CostContext) -> FloatArray:
        """Per-share fee on traded units."""
        units = ctx.traded_units()
        notional = np.abs(ctx.trades) * ctx.equity[:, None]
        return _per_equity(self._fee(units, notional), ctx.equity)

    def commission(self, quantity: float, price: float, instrument: Instrument) -> float:
        """Per-share fee for one fill."""
        units = np.array([abs(quantity)])
        notional = units * price * instrument.multiplier
        return float(self._fee(units, notional)[0])


class BpsParams(CostParams):
    """Basis points of notional."""

    bps: float = Field(5.0, ge=0, le=500, description="Basis points of traded notional")


@register("cost_model", name="bps_notional", version="1.0.0", tags=["commission", "basic"])
class BpsNotional(CostModel):
    """A fixed number of basis points of traded notional."""

    Params = BpsParams
    params: BpsParams
    capabilities: ClassVar[frozenset[str]] = BOTH_ENGINES
    category: ClassVar[str] = COST_COMMISSION

    def vectorized(self, ctx: CostContext) -> FloatArray:
        """Bps x |traded weight|."""
        return np.abs(ctx.trades) * self.params.bps * BPS

    def commission(self, quantity: float, price: float, instrument: Instrument) -> float:
        """Bps x notional."""
        return abs(quantity) * price * instrument.multiplier * self.params.bps * BPS


class Tier(BaseModel):
    """One tier of a tiered schedule."""

    up_to_shares: float = Field(description="Upper bound of order size for this tier")
    per_share: float = Field(ge=0, description="Rate per share within this tier")


class TieredParams(CostParams):
    """Tiered per-share schedule (marginal rates by order size)."""

    tiers: list[Tier] = Field(
        default_factory=lambda: [
            Tier(up_to_shares=500, per_share=0.0035),
            Tier(up_to_shares=5_000, per_share=0.002),
            Tier(up_to_shares=1e12, per_share=0.001),
        ],
        description="Marginal per-share rates by cumulative order size",
    )
    minimum: float = Field(0.35, ge=0, description="Minimum per order")


@register("cost_model", name="tiered", version="1.0.0", tags=["commission"])
class Tiered(CostModel):
    """Tiered per-share schedule: marginal rates by order size, with a minimum."""

    Params = TieredParams
    params: TieredParams
    capabilities: ClassVar[frozenset[str]] = BOTH_ENGINES
    category: ClassVar[str] = COST_COMMISSION

    def _fee(self, units: FloatArray) -> FloatArray:
        fee = np.zeros_like(units)
        lower = 0.0
        for tier in sorted(self.params.tiers, key=lambda t: t.up_to_shares):
            in_tier = np.clip(units - lower, 0.0, tier.up_to_shares - lower)
            fee += in_tier * tier.per_share
            lower = tier.up_to_shares
        return np.where(units > 0, np.maximum(fee, self.params.minimum), 0.0)

    def vectorized(self, ctx: CostContext) -> FloatArray:
        """Tiered fee on traded units."""
        return _per_equity(self._fee(ctx.traded_units()), ctx.equity)

    def commission(self, quantity: float, price: float, instrument: Instrument) -> float:
        """Tiered fee for one fill."""
        return float(self._fee(np.array([abs(quantity)]))[0])


class ContractParams(CostParams):
    """Per-contract fee."""

    per_contract: float = Field(2.0, ge=0, description="Currency per contract")


def _contract_fee(ctx: CostContext, per_contract: float, asset_class: AssetClass) -> FloatArray:
    mask = np.array([c == asset_class.value for c in ctx.asset_classes], dtype=np.float64)
    return _per_equity(ctx.traded_units() * per_contract * mask[None, :], ctx.equity)


@register("cost_model", name="futures_per_contract", version="1.0.0", tags=["futures"])
class FuturesPerContract(CostModel):
    """Exchange + broker fee per futures contract traded (futures only)."""

    Params = ContractParams
    params: ContractParams
    capabilities: ClassVar[frozenset[str]] = BOTH_ENGINES | {"asset:future"}
    category: ClassVar[str] = COST_FEES

    def vectorized(self, ctx: CostContext) -> FloatArray:
        """Fee per contract on futures columns."""
        return _contract_fee(ctx, self.params.per_contract, AssetClass.FUTURE)

    def commission(self, quantity: float, price: float, instrument: Instrument) -> float:
        """Fee per contract."""
        if instrument.asset_class is not AssetClass.FUTURE:
            return 0.0
        return abs(quantity) * self.params.per_contract


class OptionContractParams(CostParams):
    """Per-contract option fee."""

    per_contract: float = Field(0.65, ge=0, description="Currency per option contract")


@register("cost_model", name="options_per_contract", version="1.0.0", tags=["options"])
class OptionsPerContract(CostModel):
    """Fee per option contract traded (options only)."""

    Params = OptionContractParams
    params: OptionContractParams
    capabilities: ClassVar[frozenset[str]] = BOTH_ENGINES | {"asset:option"}
    category: ClassVar[str] = COST_FEES

    def vectorized(self, ctx: CostContext) -> FloatArray:
        """Fee per contract on option columns."""
        return _contract_fee(ctx, self.params.per_contract, AssetClass.OPTION)

    def commission(self, quantity: float, price: float, instrument: Instrument) -> float:
        """Fee per contract."""
        if instrument.asset_class is not AssetClass.OPTION:
            return 0.0
        return abs(quantity) * self.params.per_contract

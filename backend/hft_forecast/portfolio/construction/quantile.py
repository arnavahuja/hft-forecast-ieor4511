"""Top-N and quantile long-short constructor."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field

from hft_forecast.core.errors import DataError
from hft_forecast.core.interfaces import ConstructionContext, PortfolioConstructor
from hft_forecast.core.params import ConstructorParams
from hft_forecast.core.registry import register
from hft_forecast.core.types import BoolArray, FloatArray, IntArray, TargetFrame, TargetKind


class QuantileParams(ConstructorParams):
    """Quantile long-short parameters."""

    quantiles: int = Field(10, ge=2, le=100, description="Number of quantile buckets")
    top_n: int = Field(0, ge=0, le=5000, description="If > 0, hold top/bottom N instead")
    long_only: bool = Field(False, description="Hold only the top bucket")
    gross: float = Field(1.0, gt=0, le=5, description="Target gross exposure")
    min_names: int = Field(4, ge=1, description="Minimum instruments with a signal to trade")
    weighting: Literal["equal", "value"] = Field(
        "equal", description="Weights within each leg: equal, or proportional to a size field"
    )
    weight_field: str = Field("market_cap", description="Size field for value weighting")
    max_leg_weight: float = Field(
        1.0,
        gt=0,
        le=1,
        description="Cap on any single name as a fraction of its leg (excess is spread "
        "over the other names)",
    )


MAX_CAP_ITERATIONS = 100


def leg_weights(member: BoolArray, size: FloatArray, cap: float) -> FloatArray:
    """Row-wise weights summing to 1 over ``member``, proportional to ``size``, capped.

    Names above the cap are set to it and the excess is redistributed pro rata over the
    rest until no name exceeds it (or every name is capped when the cap is infeasible).
    """
    raw = np.where(member & np.isfinite(size) & (size > 0), size, 0.0)
    total = raw.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(total > 0, raw / total, 0.0)
    if cap >= 1.0:
        return w
    for _ in range(MAX_CAP_ITERATIONS):
        over = w > cap + 1e-12
        if not over.any():
            break
        capped = w >= cap - 1e-12
        excess = np.where(over, w - cap, 0.0).sum(axis=1, keepdims=True)
        free = np.where(capped, 0.0, w)
        free_total = free.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(free_total > 0, free / free_total, 0.0)
        w = np.where(capped, np.minimum(w, cap), w + excess * share)
    return w


def rank_rows(values: FloatArray) -> IntArray:
    """Row-wise 0-based rank of finite values (NaN rows get -1)."""
    filled = np.where(np.isfinite(values), values, np.inf)
    order = np.argsort(filled, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(values.shape[1])[None, :], axis=1)
    return np.where(np.isfinite(values), ranks, -1)


@register(
    "portfolio_constructor",
    name="quantile_long_short",
    version="1.0.0",
    tags=["cross-sectional"],
    capabilities={"supports:short"},
)
class QuantileLongShort(PortfolioConstructor):
    """Long the top quantile (or top N), short the bottom one.

    Within each leg names are equal- or value-weighted (optionally capped per name). With
    ``gross`` 2 each leg is 100% of equity (dollar neutral, 200% gross).
    """

    Params = QuantileParams
    params: QuantileParams

    def construct(self, signals: TargetFrame, ctx: ConstructionContext) -> TargetFrame:
        """Rank signals cross-sectionally each bar."""
        p = self.params
        s = signals.values
        count = np.isfinite(s).sum(axis=1, keepdims=True)
        ranks = rank_rows(s)
        if p.top_n > 0:
            n = np.minimum(p.top_n, count // 2 if not p.long_only else count)
            long = (ranks >= count - n) & (ranks >= 0)
            short = (ranks >= 0) & (ranks < n)
        else:
            bucket = np.where(ranks >= 0, (ranks * p.quantiles) // np.maximum(count, 1), -1)
            long = bucket == p.quantiles - 1
            short = bucket == 0
        if p.long_only:
            short = np.zeros_like(short)
        enough = count >= p.min_names
        legs = 1.0 if p.long_only else 2.0
        if p.weighting == "value":
            if not ctx.data.has_field(p.weight_field):
                raise DataError(f"Value weighting needs the '{p.weight_field}' field")
            size = ctx.data.aligned_panel(p.weight_field, signals.timestamps, signals.instruments)
            w = (
                leg_weights(long, size, p.max_leg_weight)
                - leg_weights(short, size, p.max_leg_weight)
            ) * (p.gross / legs)
        else:
            n_long = long.sum(axis=1, keepdims=True)
            n_short = short.sum(axis=1, keepdims=True)
            ones = np.ones_like(s)
            if p.max_leg_weight < 1.0:
                w = (
                    leg_weights(long, ones, p.max_leg_weight)
                    - leg_weights(short, ones, p.max_leg_weight)
                ) * (p.gross / legs)
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    w = np.where(long, p.gross / legs / n_long, 0.0) - np.where(
                        short, p.gross / legs / n_short, 0.0
                    )
        w = np.where(enough, np.nan_to_num(w), 0.0)
        return signals.replace(values=w, kind=TargetKind.WEIGHTS)

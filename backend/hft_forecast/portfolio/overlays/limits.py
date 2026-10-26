"""Portfolio limit overlays: position, group, exposure, leverage, turnover, liquidity."""

from __future__ import annotations

import numpy as np
from pydantic import Field, model_validator

from hft_forecast.core import columns as C
from hft_forecast.core.interfaces import (
    ENGINE_VECTORIZED,
    Overlay,
    OverlayContext,
)
from hft_forecast.core.numeric import rolling_mean
from hft_forecast.core.params import OverlayParams
from hft_forecast.core.registry import register
from hft_forecast.core.types import FloatArray, TargetFrame

EPS = 1e-12
VECTORIZED = frozenset({ENGINE_VECTORIZED})


class MaxWeightParams(OverlayParams):
    """Maximum absolute position weight."""

    max_weight: float = Field(0.10, gt=0, le=5, description="Cap on |weight| per instrument")


@register(
    "overlay",
    name="max_position_weight",
    version="1.0.0",
    tags=["limits"],
    capabilities=VECTORIZED,
)
class MaxPositionWeight(Overlay):
    """Clip each position to a maximum absolute weight (excess goes to cash)."""

    Params = MaxWeightParams
    params: MaxWeightParams

    def apply(self, targets: TargetFrame, ctx: OverlayContext) -> TargetFrame:
        """Clip weights."""
        cap = self.params.max_weight
        return targets.replace(values=np.clip(targets.values, -cap, cap))


class GroupParams(OverlayParams):
    """Group (e.g. sector) limit."""

    field: str = Field("sector", description="Text data field holding each instrument's group")
    groups: dict[str, str] = Field(
        default_factory=dict, description="Explicit instrument -> group map (overrides field)"
    )
    max_group_weight: float = Field(0.30, gt=0, le=5, description="Cap on gross weight per group")


def group_labels(
    instruments: tuple[str, ...], ctx: OverlayContext, field: str, explicit: dict[str, str]
) -> list[str]:
    """Group label per instrument (explicit map, else the data field, else 'other')."""
    from_data = ctx.data.latest_text(field)
    return [explicit.get(i) or from_data.get(i) or "other" for i in instruments]


@register("overlay", name="max_group_weight", version="1.0.0", tags=["limits", "sector"])
class MaxGroupWeight(Overlay):
    """Scale down every group whose gross weight exceeds the cap."""

    Params = GroupParams
    params: GroupParams

    def apply(self, targets: TargetFrame, ctx: OverlayContext) -> TargetFrame:
        """Scale groups proportionally."""
        labels = np.array(
            group_labels(targets.instruments, ctx, self.params.field, self.params.groups)
        )
        w = targets.filled()
        out = w.copy()
        for g in np.unique(labels):
            cols = labels == g
            gross = np.abs(w[:, cols]).sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                scale = np.where(
                    gross > self.params.max_group_weight, self.params.max_group_weight / gross, 1.0
                )
            out[:, cols] = w[:, cols] * scale[:, None]
        return targets.replace(values=out)


class ExposureParams(OverlayParams):
    """Gross and net exposure caps."""

    max_gross: float = Field(1.0, gt=0, le=20, description="Maximum gross exposure")
    max_net: float = Field(1.0, ge=-20, le=20, description="Maximum net exposure")
    min_net: float = Field(-1.0, ge=-20, le=20, description="Minimum net exposure")

    @model_validator(mode="after")
    def _check(self) -> ExposureParams:
        if self.min_net > self.max_net:
            raise ValueError("min_net must not exceed max_net")
        return self


def cap_gross(w: FloatArray, cap: float) -> FloatArray:
    """Scale rows whose gross exposure exceeds ``cap``."""
    gross = np.abs(w).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        scale = np.where(gross > cap, cap / gross, 1.0)
    return w * scale[:, None]


@register("overlay", name="exposure_limits", version="1.0.0", tags=["limits"])
class ExposureLimits(Overlay):
    """Cap gross exposure and keep net exposure within bounds (by shrinking the larger side)."""

    Params = ExposureParams
    params: ExposureParams

    def apply(self, targets: TargetFrame, ctx: OverlayContext) -> TargetFrame:
        """Apply gross, then net limits."""
        p = self.params
        w = cap_gross(targets.filled(), p.max_gross)
        longs = np.clip(w, 0, None)
        shorts = np.clip(w, None, 0)
        lsum, ssum = longs.sum(axis=1), -shorts.sum(axis=1)
        net = lsum - ssum
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            long_scale = np.where(net > p.max_net, (p.max_net + ssum) / lsum, 1.0)
            short_scale = np.where(net < p.min_net, (lsum - p.min_net) / ssum, 1.0)
        long_scale = np.clip(np.nan_to_num(long_scale, nan=1.0), 0.0, 1.0)
        short_scale = np.clip(np.nan_to_num(short_scale, nan=1.0), 0.0, 1.0)
        return targets.replace(values=longs * long_scale[:, None] + shorts * short_scale[:, None])


class LeverageParams(OverlayParams):
    """Leverage cap."""

    max_leverage: float = Field(1.0, gt=0, le=20, description="Maximum gross leverage")


@register("overlay", name="leverage_cap", version="1.0.0", tags=["limits"], capabilities=VECTORIZED)
class LeverageCap(Overlay):
    """Scale rows so gross leverage never exceeds a cap."""

    Params = LeverageParams
    params: LeverageParams

    def apply(self, targets: TargetFrame, ctx: OverlayContext) -> TargetFrame:
        """Cap gross leverage."""
        return targets.replace(values=cap_gross(targets.filled(), self.params.max_leverage))


class TurnoverParams(OverlayParams):
    """Turnover cap."""

    max_turnover: float = Field(
        0.10, gt=0, le=10, description="Maximum one-way turnover per bar (fraction)"
    )


@register("overlay", name="turnover_cap", version="1.0.0", tags=["limits", "costs"])
class TurnoverCap(Overlay):
    """Move only part of the way towards new targets when the change is too large.

    Path dependent (each bar starts from the previous capped weights), so it iterates over
    bars; each step is vectorized across instruments.
    """

    Params = TurnoverParams
    params: TurnoverParams

    def apply(self, targets: TargetFrame, ctx: OverlayContext) -> TargetFrame:
        """Blend towards targets with at most ``max_turnover`` one-way turnover per bar."""
        w = targets.filled()
        out = np.zeros_like(w)
        prev = np.zeros(w.shape[1])
        limit = 2.0 * self.params.max_turnover
        capped = 0
        for t in range(len(w)):
            delta = w[t] - prev
            traded = np.abs(delta).sum()
            if traded > limit:
                delta *= limit / traded
                capped += 1
            prev = prev + delta
            out[t] = prev
        ctx.notes.append(f"{capped} bars capped")
        return targets.replace(values=out)


class LiquidityParams(OverlayParams):
    """Liquidity cap."""

    max_adv_pct: float = Field(
        0.05, gt=0, le=1, description="Max position as a fraction of average daily volume"
    )
    adv_window: int = Field(20, ge=1, le=252, description="Average volume window (bars)")


@register("overlay", name="liquidity_cap", version="1.0.0", tags=["limits", "capacity"])
class LiquidityCap(Overlay):
    """Cap each position's value at a fraction of trailing average dollar volume.

    Equity is approximated by the zero-cost equity path of the incoming targets.
    """

    Params = LiquidityParams
    params: LiquidityParams

    def apply(self, targets: TargetFrame, ctx: OverlayContext) -> TargetFrame:
        """Clip |weight| to ``pct x ADV$ / equity``."""
        data = ctx.data
        if not data.has_field(C.VOLUME):
            ctx.notes.append("no volume field: liquidity cap skipped")
            return targets
        close = data.aligned_panel(C.CLOSE, targets.timestamps, targets.instruments)
        vol = data.aligned_panel(C.VOLUME, targets.timestamps, targets.instruments)
        adv_dollars = rolling_mean(close * vol, self.params.adv_window, 1)
        equity = ctx.initial_capital * np.cumprod(1.0 + ctx.simulate(targets))
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            cap = self.params.max_adv_pct * adv_dollars / equity[:, None]
        cap = np.where(np.isfinite(cap), cap, np.inf)
        w = targets.filled()
        clipped = np.sign(w) * np.minimum(np.abs(w), cap)
        ctx.notes.append(f"{int((np.abs(clipped - w) > EPS).sum())} cells capped")
        return targets.replace(values=clipped)

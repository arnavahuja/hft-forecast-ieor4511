"""Position sizing overlays: volatility targeting, fractional Kelly, fixed fractional, ATR."""

from __future__ import annotations

import numpy as np
from pydantic import Field

from hft_forecast.core import columns as C
from hft_forecast.core.errors import ConfigError
from hft_forecast.core.interfaces import Context, Overlay, OverlayContext
from hft_forecast.core.numeric import rolling_mean, rolling_std, shift
from hft_forecast.core.params import OverlayParams
from hft_forecast.core.registry import register
from hft_forecast.core.types import FloatArray, TargetFrame


def scale_rows(targets: TargetFrame, scale: FloatArray) -> TargetFrame:
    """Multiply each row of the targets by a scalar."""
    return targets.replace(values=targets.values * scale[:, None])


class VolTargetParams(OverlayParams):
    """Volatility targeting parameters."""

    target_vol: float = Field(0.10, gt=0, le=2, description="Annualized target volatility")
    window: int = Field(63, ge=5, le=1260, description="Realized volatility window (bars)")
    max_leverage: float = Field(2.0, gt=0, le=10, description="Maximum scale factor")


@register(
    "overlay",
    name="vol_target",
    version="1.0.0",
    tags=["sizing"],
    capabilities={"engine:vectorized", "engine:event"},
)
class VolTarget(Overlay):
    """Scale exposure so the strategy's trailing realized volatility matches a target."""

    Params = VolTargetParams
    params: VolTargetParams

    def apply(self, targets: TargetFrame, ctx: OverlayContext) -> TargetFrame:
        """Scale by target / trailing realized vol of the incoming targets."""
        p = self.params
        r = ctx.simulate(targets)
        vol = rolling_std(r, p.window) * np.sqrt(ctx.periods_per_year)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.clip(p.target_vol / vol, 0.0, p.max_leverage)
        scale = np.where(np.isfinite(scale), scale, 1.0)
        ctx.notes.append(f"average scale {float(np.mean(scale)):.2f}")
        return scale_rows(targets, scale)

    def on_bar(self, ctx: Context) -> None:
        """Scale staged targets by target / realized portfolio vol so far."""
        staged = ctx.target_weights()
        rets = ctx.portfolio_returns(self.params.window)
        if not staged or len(rets) < self.params.window:
            return
        vol = float(np.std(rets, ddof=1)) * np.sqrt(ctx.periods_per_year)
        if vol <= 0:
            return
        scale = min(self.params.target_vol / vol, self.params.max_leverage)
        ctx.set_target_weights({k: v * scale for k, v in staged.items()})


class KellyParams(OverlayParams):
    """Fractional Kelly parameters."""

    fraction: float = Field(0.5, gt=0, le=1, description="Fraction of full Kelly")
    window: int = Field(252, ge=20, le=2520, description="Estimation window (bars)")
    max_leverage: float = Field(2.0, gt=0, le=10, description="Maximum scale factor")


@register("overlay", name="fractional_kelly", version="1.0.0", tags=["sizing"])
class FractionalKelly(Overlay):
    """Scale exposure by a fraction of the Kelly leverage ``mu / sigma^2`` (trailing)."""

    Params = KellyParams
    params: KellyParams

    def apply(self, targets: TargetFrame, ctx: OverlayContext) -> TargetFrame:
        """Scale by fraction x trailing Kelly leverage."""
        p = self.params
        r = ctx.simulate(targets)
        mu = rolling_mean(r, p.window)
        var = rolling_std(r, p.window) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            kelly = mu / var
        scale = np.clip(p.fraction * np.where(np.isfinite(kelly), kelly, 0.0), 0.0, p.max_leverage)
        warm = np.isnan(mu)
        scale = np.where(warm, 1.0, scale)
        ctx.notes.append(f"average scale {float(np.mean(scale)):.2f}")
        return scale_rows(targets, scale)


class FixedFractionalParams(OverlayParams):
    """Fixed fractional parameters."""

    fraction: float = Field(0.5, gt=0, le=10, description="Constant exposure multiplier")


@register("overlay", name="fixed_fractional", version="1.0.0", tags=["sizing"])
class FixedFractional(Overlay):
    """Scale all targets by a constant fraction."""

    Params = FixedFractionalParams
    params: FixedFractionalParams

    def apply(self, targets: TargetFrame, ctx: OverlayContext) -> TargetFrame:
        """Multiply by the fraction."""
        return targets.replace(values=targets.values * self.params.fraction)


class AtrParams(OverlayParams):
    """ATR sizing parameters."""

    risk_per_position: float = Field(
        0.01, gt=0, le=0.2, description="Equity risked per position per ATR stop"
    )
    atr_window: int = Field(14, ge=2, le=252, description="ATR window (bars)")
    atr_multiple: float = Field(2.0, gt=0, le=20, description="Stop distance in ATRs")
    max_weight: float = Field(0.25, gt=0, le=5, description="Cap per position")


@register("overlay", name="atr_sizing", version="1.0.0", tags=["sizing"])
class AtrSizing(Overlay):
    """Size each position so an ``atr_multiple`` x ATR move costs ``risk_per_position``."""

    Params = AtrParams
    params: AtrParams

    def apply(self, targets: TargetFrame, ctx: OverlayContext) -> TargetFrame:
        """Weight = sign(target) x risk / (multiple x ATR / close)."""
        p = self.params
        data = ctx.data.select_instruments(list(targets.instruments))
        if not (data.has_field(C.HIGH) and data.has_field(C.LOW)):
            raise ConfigError("ATR sizing needs high and low prices")
        tf = targets.reindex(data.timestamps, data.instruments)
        close = data.panel(C.CLOSE)
        prev = shift(close, 1)
        tr = np.fmax(
            data.panel(C.HIGH) - data.panel(C.LOW),
            np.fmax(np.abs(data.panel(C.HIGH) - prev), np.abs(data.panel(C.LOW) - prev)),
        )
        atr_pct = rolling_mean(tr, p.atr_window) / close
        with np.errstate(divide="ignore", invalid="ignore"):
            size = np.minimum(p.risk_per_position / (p.atr_multiple * atr_pct), p.max_weight)
        w = np.sign(np.nan_to_num(tf.values)) * np.nan_to_num(size)
        out = tf.replace(values=w)
        return out.reindex(targets.timestamps, targets.instruments)

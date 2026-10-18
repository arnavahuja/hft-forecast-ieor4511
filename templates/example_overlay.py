"""Example overlay: a user overlay."""

from __future__ import annotations

import numpy as np
from pydantic import Field

from hft_forecast.core import Overlay, OverlayContext, OverlayParams, TargetFrame, register


class ExampleOverlayParams(OverlayParams):
    """Parameters."""

    scale: float = Field(0.5, gt=0, le=5, description="Exposure multiplier")


@register("overlay", name="example_overlay", version="0.1.0", tags=["user"])
class ExampleOverlay(Overlay):
    """Scale all targets by a constant (replace with your own rule)."""

    Params = ExampleOverlayParams
    params: ExampleOverlayParams

    def apply(self, targets: TargetFrame, ctx: OverlayContext) -> TargetFrame:
        """Row t may only use data up to t; ``ctx.simulate(targets)`` gives zero-cost returns."""
        return targets.replace(values=np.asarray(targets.values) * self.params.scale)

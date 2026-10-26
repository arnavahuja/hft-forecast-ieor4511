"""Example cost model: a user cost model."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from pydantic import Field

from hft_forecast.core import CostContext, CostModel, CostParams, register
from hft_forecast.core.types import FloatArray


class ExampleCostModelParams(CostParams):
    """Parameters."""

    bps: float = Field(3.0, ge=0, description="Basis points of traded notional")


@register(
    "cost_model",
    name="example_cost_model",
    version="0.1.0",
    tags=["user"],
    capabilities={"engine:vectorized"},
)
class ExampleCostModel(CostModel):
    """Flat bps cost (vectorized and event forms)."""

    Params = ExampleCostModelParams
    params: ExampleCostModelParams
    category: ClassVar[str] = "commission"

    def vectorized(self, ctx: CostContext) -> FloatArray:
        """Cost as a fraction of equity, shape (T, N)."""
        return np.abs(ctx.trades) * self.params.bps * 1e-4

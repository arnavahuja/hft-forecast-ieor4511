"""Point-in-time size universe: the N largest instruments by a size field (market cap)."""

from __future__ import annotations

import numpy as np
from pydantic import Field

from hft_forecast.core import columns as C
from hft_forecast.core.errors import DataError
from hft_forecast.core.interfaces import UniverseProvider
from hft_forecast.core.params import UniverseParams
from hft_forecast.core.registry import register
from hft_forecast.core.types import FloatArray, MarketData

MEMBER_FIELD = "universe_member"


class SizeParams(UniverseParams):
    """Parameters for the size universe."""

    top_n: int = Field(1000, ge=1, description="Number of largest instruments to keep")
    size_field: str = Field("market_cap", description="Field ranked each bar")
    min_price: float = Field(5.0, ge=0, description="Minimum close price at formation")
    source_universe: str = Field(
        "",
        description="Optional source-side universe to request first (e.g. 'crsp_common' or "
        "'sp500' on WRDS); only its members are ranked",
    )


@register(
    "universe",
    name="market_cap_top_n",
    version="1.0.0",
    tags=["size", "point-in-time", "survivorship-free"],
)
class MarketCapTopN(UniverseProvider):
    """Top N instruments by size at each bar, among those above a minimum price.

    Uses only the values at each bar, so membership is point-in-time. With a source
    universe, the source returns every instrument that was ever a member (including later
    delistings) and ranking is restricted to the members at each bar.
    """

    Params = SizeParams
    params: SizeParams

    def request_universe(self) -> str | None:
        """Source-side universe, if configured."""
        return self.params.source_universe or None

    def membership(self, data: MarketData) -> FloatArray:
        """Point-in-time membership mask."""
        p = self.params
        if not data.has_field(p.size_field):
            raise DataError(f"The data has no '{p.size_field}' field to rank by size")
        close = data.panel(C.CLOSE)
        size = data.panel(p.size_field)
        ok = np.isfinite(close) & (np.nan_to_num(close) >= p.min_price) & np.isfinite(size)
        if p.source_universe and data.has_field(MEMBER_FIELD):
            ok &= np.nan_to_num(data.panel(MEMBER_FIELD), nan=0.0) > 0
        score = np.where(ok, size, -np.inf)
        k = min(p.top_n, score.shape[1])
        order = np.argsort(-score, axis=1, kind="stable")
        top = np.zeros(score.shape, dtype=bool)
        np.put_along_axis(top, order[:, :k], values=True, axis=1)
        return (top & ok).astype(np.float64)

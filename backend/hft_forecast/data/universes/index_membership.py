"""Point-in-time index membership provided by the data source (e.g. CRSP S&P 500 list)."""

from __future__ import annotations

import numpy as np
from pydantic import Field

from hft_forecast.core import columns as C
from hft_forecast.core.errors import DataError
from hft_forecast.core.interfaces import UniverseProvider
from hft_forecast.core.params import UniverseParams
from hft_forecast.core.registry import register
from hft_forecast.core.types import FloatArray, MarketData


class IndexParams(UniverseParams):
    """Index membership parameters."""

    index: str = Field("sp500", description="Source-side universe code")
    field: str = Field("universe_member", description="Membership flag field in the data")


@register(
    "universe",
    name="index_membership",
    version="1.0.0",
    tags=["point-in-time", "survivorship-free"],
)
class IndexMembership(UniverseProvider):
    """Members of an index on each date, using the source's historical membership lists.

    The data source returns every instrument that was ever a member in the date range
    (including later-delisted ones) plus a 0/1 membership field.
    """

    Params = IndexParams
    params: IndexParams

    def request_universe(self) -> str | None:
        """Ask the source for the index."""
        return self.params.index

    def membership(self, data: MarketData) -> FloatArray:
        """Membership flag (and a price) at each bar."""
        if not data.has_field(self.params.field):
            raise DataError(
                f"The data has no '{self.params.field}' field; use a source that "
                f"provides the '{self.params.index}' universe"
            )
        flag = np.nan_to_num(data.panel(self.params.field), nan=0.0)
        return ((flag > 0) & np.isfinite(data.panel(C.CLOSE))).astype(np.float64)

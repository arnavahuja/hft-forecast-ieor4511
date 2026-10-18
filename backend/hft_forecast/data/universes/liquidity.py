"""Point-in-time liquidity universe: top N by trailing average dollar volume."""

from __future__ import annotations

import numpy as np
from pydantic import Field

from hft_forecast.core import columns as C
from hft_forecast.core.interfaces import UniverseProvider
from hft_forecast.core.params import UniverseParams
from hft_forecast.core.registry import register
from hft_forecast.core.types import FloatArray, MarketData


class LiquidityParams(UniverseParams):
    """Parameters for the liquidity universe."""

    top_n: int = Field(100, ge=1, description="Number of most liquid instruments to keep")
    window: int = Field(63, ge=1, description="Trailing window (bars) for dollar volume")
    min_price: float = Field(5.0, ge=0, description="Minimum close price")
    rebalance_every: int = Field(21, ge=1, description="Recompute membership every N bars")


@register("universe", name="liquidity_top_n", version="1.0.0", tags=["liquidity"])
class LiquidityUniverse(UniverseProvider):
    """Top N instruments by trailing dollar volume, recomputed on a schedule.

    Uses only data up to each bar (trailing window), so membership is point-in-time.
    """

    Params = LiquidityParams
    params: LiquidityParams

    def membership(self, data: MarketData) -> FloatArray:
        """Point-in-time membership mask."""
        close = data.panel(C.CLOSE)
        if not data.has_field(C.VOLUME):
            return np.isfinite(close).astype(np.float64)
        dollar = np.nan_to_num(close * data.panel(C.VOLUME), nan=0.0)
        csum = np.cumsum(dollar, axis=0)
        window = self.params.window
        lagged = np.vstack([np.zeros((window, dollar.shape[1])), csum[:-window]])[: len(csum)]
        avg = (csum - lagged) / window
        ok = np.isfinite(close) & (np.nan_to_num(close) >= self.params.min_price)
        score = np.where(ok, avg, -np.inf)
        n_t = score.shape[0]
        rank_rows = np.arange(0, n_t, self.params.rebalance_every)
        order = np.argsort(-score[rank_rows], axis=1)
        top = np.zeros((len(rank_rows), score.shape[1]), dtype=bool)
        k = min(self.params.top_n, score.shape[1])
        np.put_along_axis(top, order[:, :k], values=True, axis=1)
        top &= np.isfinite(score[rank_rows])
        idx = np.repeat(np.arange(len(rank_rows)), np.diff(np.append(rank_rows, n_t)))
        return (top[idx] & ok).astype(np.float64)

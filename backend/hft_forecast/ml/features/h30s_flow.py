"""h30s group: level-1 order flow and tick-regime features for the 5 s grid.

Adds what ``basic`` leaves out:

- order-flow imbalance (Cont, Kukanov & Stoikov 2014) between consecutive bar-end NBBO
  snapshots, summed over several windows and scaled by the average depth;
- the persistence of the queue imbalance (window means and changes);
- the tick regime (spread in ticks, the share of recent bars at a one-tick spread, the tick
  size relative to the price) and the imbalance interacted with it: queue imbalance predicts
  the next mid move far better when the spread is pinned at one tick;
- intra-bar information (mid versus its time-weighted average, the mid range) and quote
  intensity relative to its recent level.

Every expression is a non-negative shift or a trailing window taken over the session, as
:class:`~hft_forecast.ml.base.FeatureSet` requires. Trade-flow features (signed volume) need
the trade bars joined to the quote bars, which the shared dataset builder does not do yet.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Final

import polars as pl
from pydantic import Field

from hft_forecast.core import columns as C
from hft_forecast.core.params import PluginParams
from hft_forecast.ml.base import BPS, FeatureSet
from hft_forecast.ml.features.registry import register_feature_set, session_group

OFI_WINDOWS: Final = (1, 2, 6, 12, 60)  # 5 s, 10 s, 30 s, 1 min, 5 min on the 5 s grid
IMBALANCE_WINDOWS: Final = (6, 12, 60)
IMBALANCE_CHANGES: Final = (1, 6)
REGIME_WINDOW: Final = 60
TICK: Final = 0.01
ONE_TICK: Final = 1.5  # spreads below 1.5 ticks count as one tick
EPS: Final = 1e-12


class FlowParams(PluginParams):
    """Window lengths in bars (on the 5 s grid: 6 bars = 30 s, 60 bars = 5 min)."""

    ofi_windows: list[Annotated[int, Field(ge=1)]] = Field(
        default_factory=lambda: list(OFI_WINDOWS), description="OFI sum windows in bars"
    )
    imbalance_windows: list[Annotated[int, Field(ge=2)]] = Field(
        default_factory=lambda: list(IMBALANCE_WINDOWS),
        description="Windows of the mean queue imbalance, in bars",
    )
    imbalance_changes: list[Annotated[int, Field(ge=1)]] = Field(
        default_factory=lambda: list(IMBALANCE_CHANGES),
        description="Lags of the queue-imbalance change, in bars",
    )
    regime_window: int = Field(
        REGIME_WINDOW, ge=2, description="Window of the one-tick share and quote intensity"
    )


@register_feature_set(name="h30s_flow", version="1.0.0")
class H30sFlow(FeatureSet):
    """L1 order-flow imbalance, imbalance persistence, tick regime and intra-bar features."""

    Params = FlowParams
    params: FlowParams
    lookback_bars: ClassVar[int] = max(*OFI_WINDOWS, REGIME_WINDOW)
    requires: ClassVar[tuple[str, ...]] = (
        C.BID,
        C.ASK,
        C.BID_SIZE,
        C.ASK_SIZE,
        C.MID,
        C.MID_TWAP,
        C.MICROPRICE,
        C.SPREAD,
        C.HIGH,
        C.LOW,
        C.N_QUOTES,
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Add the ``f_*`` columns."""
        group = session_group()
        p = self.params
        bid, ask = pl.col(C.BID), pl.col(C.ASK)
        qb, qa = pl.col(C.BID_SIZE), pl.col(C.ASK_SIZE)

        def prev(e: pl.Expr) -> pl.Expr:
            """The previous bar's value in the same session (null on the first bar)."""
            return e.shift(1).over(group)

        ofi = (
            pl.when(bid >= prev(bid)).then(qb).otherwise(0.0)
            - pl.when(bid <= prev(bid)).then(prev(qb)).otherwise(0.0)
            - pl.when(ask <= prev(ask)).then(qa).otherwise(0.0)
            + pl.when(ask >= prev(ask)).then(prev(qa)).otherwise(0.0)
        )
        spread_ticks = pl.col(C.SPREAD) / TICK
        out = frame.with_columns(
            _ofi=ofi,  # null on a session's first bar: no previous snapshot
            _depth=(qb + qa) / 2,
            _imb=(qb - qa) / (qb + qa + EPS),
            _one_tick=(spread_ticks < ONE_TICK).cast(pl.Float64),
        )
        imb, depth = pl.col("_imb"), pl.col("_depth")
        one_tick_share = (
            pl.col("_one_tick").rolling_mean(p.regime_window, min_samples=1).over(group)
        )
        features = [
            *(
                (
                    pl.col("_ofi").rolling_sum(w, min_samples=1).over(group)
                    / (depth.rolling_mean(w, min_samples=1).over(group) + EPS)
                ).alias(f"f_ofi_{w}")
                for w in p.ofi_windows
            ),
            *(
                imb.rolling_mean(w, min_samples=1).over(group).alias(f"f_imb_mean_{w}")
                for w in p.imbalance_windows
            ),
            *(
                (imb - imb.shift(k).over(group)).alias(f"f_imb_change_{k}")
                for k in p.imbalance_changes
            ),
            spread_ticks.alias("f_spread_ticks"),
            one_tick_share.alias("f_one_tick_share"),
            (TICK / pl.col(C.MID) * BPS).alias("f_tick_bps"),
            (imb * one_tick_share).alias("f_imb_x_one_tick"),
            ((pl.col(C.MICROPRICE) - pl.col(C.MID)) / TICK).alias("f_microprice_ticks"),
            ((pl.col(C.MID) - pl.col(C.MID_TWAP)) / pl.col(C.MID) * BPS).alias("f_twap_dev_bps"),
            ((pl.col(C.HIGH) - pl.col(C.LOW)) / pl.col(C.MID) * BPS).alias("f_range_bps"),
            (
                pl.col(C.N_QUOTES)
                / (
                    pl.col(C.N_QUOTES).rolling_mean(p.regime_window, min_samples=1).over(group)
                    + EPS
                )
            ).alias("f_quotes_rel"),
        ]
        return out.with_columns(features).drop("_ofi", "_depth", "_imb", "_one_tick")

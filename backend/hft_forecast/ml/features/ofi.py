"""Order flow imbalance at the best quotes (Cont, Kukanov and Stoikov, 2014).

The paper's contribution is that the *best-quote* events alone explain most of the
short-horizon price move: for consecutive observations of the best bid and ask, the
contribution of one update is

    e_n = 1{P_b(n) >= P_b(n-1)} q_b(n) - 1{P_b(n) <= P_b(n-1)} q_b(n-1)
        - 1{P_a(n) <= P_a(n-1)} q_a(n) + 1{P_a(n) >= P_a(n-1)} q_a(n-1)

which reads as: a better bid adds its whole size, a worse bid removes the old size, and an
unchanged price contributes the change in depth. The ask enters with the opposite sign, so a
positive ``e_n`` is buying pressure. The order flow imbalance over an interval is the sum of
``e_n`` over the updates in it, and mid-price changes are close to linear in it with a slope
that falls as depth rises.

Level 1 only, which is the point: this needs the NBBO and nothing deeper, so it runs on TAQ
quote bars and on any level-1 file.

**Bar sampling.** The paper works in event time, one ``e_n`` per quote update. A bar carries
only the last NBBO of the bar, so the sum here is taken over bar-to-bar observations and
whatever happened inside a bar is invisible: an update that lifts the bid and another that
puts it back nets to zero. ``n_quotes`` tells you how many updates a bar hid. Use the finest
grid the data allows, and read ``hftf ml impact`` as a lower bound on the fit.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Final

import polars as pl
from pydantic import Field

from hft_forecast.core import columns as C
from hft_forecast.core.params import PluginParams
from hft_forecast.ml.base import FeatureSet
from hft_forecast.ml.features.registry import register_feature_set, session_group

OFI: Final = "ofi"
"""Raw per-bar order flow imbalance, in shares. Not an ``f_`` column: the regression in
``ml.impact`` wants it in the units the paper's slope is quoted in."""

DEPTH: Final = "depth"
"""Average of the two displayed sizes, the ``D`` the slope scales with."""

MIN_WINDOW: Final = 1
DEFAULT_WINDOWS: Final = (1, 5, 10)
DEFAULT_DEPTH_WINDOW: Final = 30
EPS: Final = 1e-12


def order_flow_imbalance() -> pl.Expr:
    """Equation (1) of the paper, over consecutive rows of one instrument-session.

    The caller is responsible for the grouping; :func:`add_order_flow` applies it.
    """
    bid, ask = pl.col(C.BID), pl.col(C.ASK)
    bid_size, ask_size = pl.col(C.BID_SIZE), pl.col(C.ASK_SIZE)
    prev_bid, prev_ask = bid.shift(1), ask.shift(1)
    prev_bid_size, prev_ask_size = bid_size.shift(1), ask_size.shift(1)
    bid_flow = (bid >= prev_bid).cast(pl.Float64) * bid_size - (bid <= prev_bid).cast(
        pl.Float64
    ) * prev_bid_size
    ask_flow = (ask <= prev_ask).cast(pl.Float64) * ask_size - (ask >= prev_ask).cast(
        pl.Float64
    ) * prev_ask_size
    return bid_flow - ask_flow


def add_order_flow(frame: pl.DataFrame) -> pl.DataFrame:
    """Add the raw ``ofi`` and ``depth`` columns, never crossing a session.

    The first bar of a session has no predecessor, so its imbalance is null rather than the
    whole displayed size: the overnight gap is not an order book event.
    """
    group = session_group()
    return frame.with_columns(
        order_flow_imbalance().over(group).alias(OFI),
        ((pl.col(C.BID_SIZE) + pl.col(C.ASK_SIZE)) / 2.0).alias(DEPTH),
    )


class OfiParams(PluginParams):
    """Aggregation windows, in bars.

    The windows are the paper's interval length: it reports 10-minute intervals, which is 10
    bars on a 1 min grid and 120 on a 5 s one. Set them to the horizon you forecast.
    """

    windows: list[Annotated[int, Field(ge=MIN_WINDOW)]] = Field(
        default_factory=lambda: list(DEFAULT_WINDOWS),
        description="Bars to sum the imbalance over; one feature per window",
    )
    depth_window: int = Field(
        DEFAULT_DEPTH_WINDOW,
        ge=1,
        description="Bars of trailing average depth the imbalance is divided by",
    )


@register_feature_set(name="ofi", version="1.0.0")
class Ofi(FeatureSet):
    """Best-quote order flow imbalance, summed over several windows and scaled by depth."""

    Params = OfiParams
    params: OfiParams
    lookback_bars: ClassVar[int] = max(DEFAULT_WINDOWS)
    requires: ClassVar[tuple[str, ...]] = (C.BID, C.ASK, C.BID_SIZE, C.ASK_SIZE)

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Add ``f_ofi_<w>`` for each window, plus the depth it was scaled by.

        Dividing by trailing depth is what makes the feature comparable across names and
        across the day: the paper's slope is inversely proportional to depth, so the raw
        imbalance of a thin name and a thick one mean different things for the price.
        """
        group = session_group()
        out = add_order_flow(frame)
        # min_samples=1: depth is an average, so it is defined from the first bar of the
        # session, and nulling it would cost a window's worth of rows for nothing
        scale = (
            pl.col(DEPTH).rolling_mean(self.params.depth_window, min_samples=1).over(group) + EPS
        )
        features = [
            (pl.col(OFI).rolling_sum(window).over(group) / scale).alias(f"f_ofi_{window}")
            for window in self.params.windows
        ]
        return out.with_columns(*features, pl.col(DEPTH).log1p().alias("f_log_depth")).drop(
            OFI, DEPTH
        )

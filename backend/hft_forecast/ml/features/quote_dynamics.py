"""How the best quote is *changing*, as opposed to where it is.

``basic`` describes the state of the top of the book (spread, imbalance, microprice tilt);
this set describes its motion: the spread widening, the imbalance tilting, size arriving or
leaving, and size refilling at a price that did not move. Together with ``ofi`` it covers
the level-1 feature families surveyed in the group's WRDS TAQ feature-research notebook,
which is where the replenishment proxies come from.

Every column is a difference or a ratio of the current bar and an earlier one, taken
``.over`` the session, so the first ``delta_bars`` of a session are null rather than carrying
yesterday's close across the overnight gap.

Level 1 only. Nothing here needs depth beyond the best quote, so it runs on TAQ NBBO bars
and on any imported level-1 file.
"""

from __future__ import annotations

from typing import ClassVar, Final

import polars as pl
from pydantic import Field

from hft_forecast.core import columns as C
from hft_forecast.core.params import PluginParams
from hft_forecast.ml.base import FeatureSet
from hft_forecast.ml.features.registry import register_feature_set, session_group

EPS: Final = 1e-12
DEFAULT_DELTA_BARS: Final = 1


class QuoteDynamicsParams(PluginParams):
    """How far back the differences look.

    One bar is the natural choice and the only one the notebook used. A group on a coarse
    grid may widen it to span the same wall-clock interval: 12 bars of a 5 s grid and 1 bar
    of a 1 min grid are the same minute of quote motion.
    """

    delta_bars: int = Field(
        DEFAULT_DELTA_BARS, ge=1, description="Bars back the differences are taken over"
    )


@register_feature_set(name="quote_dynamics", version="1.0.0")
class QuoteDynamics(FeatureSet):
    """Spread change, imbalance tilt, size changes and replenishment at an unmoved price."""

    Params = QuoteDynamicsParams
    params: QuoteDynamicsParams
    lookback_bars: ClassVar[int] = DEFAULT_DELTA_BARS
    requires: ClassVar[tuple[str, ...]] = (
        C.BID,
        C.ASK,
        C.BID_SIZE,
        C.ASK_SIZE,
        C.SPREAD_BPS,
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Add the ``f_*`` columns."""
        group = session_group()
        lag = self.params.delta_bars
        bid_size, ask_size = pl.col(C.BID_SIZE), pl.col(C.ASK_SIZE)
        prev_bid_size = bid_size.shift(lag).over(group)
        prev_ask_size = ask_size.shift(lag).over(group)
        imbalance = (bid_size - ask_size) / (bid_size + ask_size + EPS)
        notional = pl.col(C.BID) * bid_size + pl.col(C.ASK) * ask_size
        return frame.with_columns(
            (pl.col(C.SPREAD_BPS) - pl.col(C.SPREAD_BPS).shift(lag).over(group)).alias(
                "f_spread_change_bps"
            ),
            ((bid_size + EPS) / (ask_size + EPS)).log().alias("f_size_log_ratio"),
            notional.log1p().alias("f_log_notional"),
            (imbalance - imbalance.shift(lag).over(group)).alias("f_imbalance_delta"),
            _log_size_change(bid_size, prev_bid_size).alias("f_bid_size_change"),
            _log_size_change(ask_size, prev_ask_size).alias("f_ask_size_change"),
            _replenishment(pl.col(C.BID), bid_size, lag, group).alias("f_bid_replenishment"),
            _replenishment(pl.col(C.ASK), ask_size, lag, group).alias("f_ask_replenishment"),
        )


def _log_size_change(size: pl.Expr, previous: pl.Expr) -> pl.Expr:
    """Log growth in displayed size.

    ``log1p`` of each side rather than of the ratio: a quote that empties to zero size is a
    real event, and the ratio would be undefined for it.
    """
    return size.log1p() - previous.log1p()


def _replenishment(price: pl.Expr, size: pl.Expr, lag: int, group: list[str]) -> pl.Expr:
    """Size added at a price that did not move, as a share of the size already there.

    The notebook's proxy for queue replenishment, and the reason it is worth having beside
    :mod:`~hft_forecast.ml.features.ofi`: the imbalance of equation (1) counts the same
    arrival, but mixes it with price-level changes, so a model cannot tell the two apart.
    Zero when the price moved (the queue is a different one) or size left.
    """
    previous_price = price.shift(lag).over(group)
    previous_size = size.shift(lag).over(group)
    added = size - previous_size
    return (
        # null at the start of a session, like every other lagged column here
        pl.when(previous_price.is_null())
        .then(None)
        .when((price == previous_price) & (added > 0))
        .then(added / (previous_size + EPS))
        .otherwise(0.0)
    )

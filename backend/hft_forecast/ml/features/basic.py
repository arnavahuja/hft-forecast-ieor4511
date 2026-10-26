"""The one feature set the template ships: level-1 microstructure, nothing clever.

Every expression is a non-negative shift or a rolling window taken over the session, so the
first ``lookback_bars`` of each session are null rather than contaminated by yesterday's
close. Groups add their own sets beside this one; they do not edit it.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Final

import polars as pl
from pydantic import Field

from hft_forecast.core import columns as C
from hft_forecast.core.params import PluginParams
from hft_forecast.ml.base import BPS, SECONDS_SINCE_OPEN, FeatureSet
from hft_forecast.ml.features.registry import register_feature_set, session_group

RETURN_LAGS: Final = (1, 5, 10)
VOL_WINDOWS: Final = (20, 100)
MIN_VOL_BARS: Final = 2
EPS: Final = 1e-12


class BasicParams(PluginParams):
    """Window lengths of the basic set.

    The windows are counted in bars, so they mean different things on different grids: 100
    bars is eight minutes of a 5 s grid but more than one session of a 5 min grid. A group on
    a coarse grid shrinks them rather than getting an all-null column.
    """

    vol_windows: list[Annotated[int, Field(ge=MIN_VOL_BARS)]] = Field(
        default_factory=lambda: list(VOL_WINDOWS),
        description="Realized-volatility windows in bars; each must fit inside a session",
    )
    coarse_seconds: int = Field(
        60, ge=1, description="Bar size of the completed coarse-grid volatility feature"
    )


@register_feature_set(name="basic", version="1.0.0")
class Basic(FeatureSet):
    """Mid returns, spread, level-1 imbalance, microprice tilt, quote intensity and vol."""

    Params = BasicParams
    params: BasicParams
    lookback_bars: ClassVar[int] = max(VOL_WINDOWS)  # with the default vol_windows
    requires: ClassVar[tuple[str, ...]] = (
        C.MID,
        C.SPREAD_BPS,
        C.BID_SIZE,
        C.ASK_SIZE,
        C.MICROPRICE,
        C.N_QUOTES,
        C.IS_STALE,
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Add the ``f_*`` columns."""
        group = session_group()
        log_mid = pl.col(C.MID).log()
        ret1 = (log_mid - log_mid.shift(1).over(group)).alias("_r1")
        out = frame.with_columns(ret1)
        size_sum = pl.col(C.BID_SIZE) + pl.col(C.ASK_SIZE)
        features = [
            *(
                (log_mid - log_mid.shift(lag).over(group)).alias(f"f_logret_{lag}")
                for lag in RETURN_LAGS
            ),
            pl.col(C.SPREAD_BPS).alias("f_spread_bps"),
            ((pl.col(C.BID_SIZE) - pl.col(C.ASK_SIZE)) / (size_sum + EPS)).alias("f_imbalance"),
            ((pl.col(C.MICROPRICE) - pl.col(C.MID)) / pl.col(C.MID) * BPS).alias(
                "f_microprice_bps"
            ),
            (pl.col(C.N_QUOTES) + 1.0).log().alias("f_log_quotes"),
            pl.col(C.IS_STALE).alias("f_is_stale"),
            pl.col(SECONDS_SINCE_OPEN).cast(pl.Float64).alias("f_seconds_since_open"),
            *(
                pl.col("_r1").rolling_std(window).over(group).alias(f"f_vol_{window}")
                for window in self.params.vol_windows
            ),
            self._completed_coarse_vol(group, bar_seconds(frame)),
        ]
        return out.with_columns(features).drop("_r1")

    def _completed_coarse_vol(self, group: list[str], bar_seconds: int) -> pl.Expr:
        """Realized vol over the last completed coarse bar (1 min by default).

        Shifting by one coarse bar's worth of fine bars is what makes it *completed*: at bar
        ``t`` the window ends at the previous coarse boundary, never inside the current one.
        """
        # at least two bars: a rolling std over one sample is null for every row, which
        # would make the whole column useless on a grid as coarse as the feature
        bars = max(self.params.coarse_seconds // bar_seconds, MIN_VOL_BARS)
        return (
            pl.col("_r1").rolling_std(bars).shift(bars).over(group).alias("f_vol_coarse_completed")
        )


def bar_seconds(frame: pl.DataFrame) -> int:
    """Fine-grid bar length, read off the frame as the smallest forward step."""
    diffs = frame.get_column(C.TIMESTAMP).diff().dt.total_seconds().to_numpy()
    positive = diffs[diffs > 0]
    return int(positive.min()) if positive.size else 1

"""Where we are in the session, and what the session and the ones before it have done.

``basic``, ``ofi`` and ``quote_dynamics`` all describe the last few bars of the top of the
book. Nothing in them knows what time it is, what the stock did overnight, or what it
usually does at this hour. This set adds that context, in five families:

* **Time of day.** The phase of the session and its sine/cosine, plus flags for the first
  and last half hour. Intraday volatility and spreads are U-shaped, so a model with no
  clock cannot tell the open from lunchtime and has to average the two.
* **Longer lookback.** A 30 min return and a 30 min realized volatility, against the 5 s to
  1 min windows the other sets stop at.
* **Overnight gap.** The jump from the previous session's close to this session's open, its
  size, and whether the price is since moving back towards the close it gapped away from.
  Large opening moves tend to revert over the rest of the day (Grant, Wolf and Yu 2005, on
  index futures; the same reversal is documented for US equity overnight gaps).
* **Range so far.** Where the price sits inside the session's running high-low range, and
  how far it has come off each end. "The high is too far up, so it should come back" is a
  statement about this feature.
* **Same time of day, historically.** The mean return and realized volatility of this
  half-hour bucket over the previous sessions. Heston, Korajczyk and Sadka (2010) show a
  stock's return in a half-hour interval predicts its return in the same interval on later
  days, which is the periodicity this estimates from the data instead of assuming.

Causality. Every column is a non-negative shift, an expanding or rolling window taken over
the session, or an aggregate of *strictly earlier* sessions. The overnight and historical
families are the only features in the project that cross a session boundary, and they only
ever look backwards: ``check_feature_lookahead`` covers them like any other set.
"""

from __future__ import annotations

import math
from typing import ClassVar, Final

import polars as pl
from pydantic import Field

from hft_forecast.core import columns as C
from hft_forecast.core.params import PluginParams
from hft_forecast.ml.base import (
    BPS,
    SECONDS_SINCE_OPEN,
    SECONDS_TO_CLOSE,
    SESSION_ID,
    FeatureSet,
)
from hft_forecast.ml.features.registry import register_feature_set, session_group

EPS: Final = 1e-12
MINUTE: Final = 60
DEFAULT_LONG_BARS: Final = 360
"""30 min on a 5 s grid: the lookback the group was asked for."""

DEFAULT_BUCKET_SECONDS: Final = 1800
"""Half-hour buckets, the interval Heston, Korajczyk and Sadka (2010) measure."""

DEFAULT_HIST_SESSIONS: Final = 20
DEFAULT_EDGE_MINUTES: Final = 30
ROW: Final = "_row"


class SessionContextParams(PluginParams):
    """Window lengths, all in the units they are named in.

    ``long_bars`` is in bars and so depends on the grid: 360 bars is 30 min of a 5 s grid
    and a session and a half of a 5 min one. The rest are in seconds, minutes or sessions
    and mean the same thing on every grid.
    """

    long_bars: int = Field(
        DEFAULT_LONG_BARS, ge=2, description="Bars in the long return and volatility windows"
    )
    bucket_seconds: int = Field(
        DEFAULT_BUCKET_SECONDS, ge=1, description="Width of the time-of-day buckets"
    )
    hist_sessions: int = Field(
        DEFAULT_HIST_SESSIONS, ge=1, description="Earlier sessions averaged per time-of-day bucket"
    )
    edge_minutes: int = Field(
        DEFAULT_EDGE_MINUTES, ge=1, description="Width of the opening and closing flags"
    )


@register_feature_set(name="session_context", version="1.0.0")
class SessionContext(FeatureSet):
    """Time of day, a 30 min lookback, the overnight gap, the day's range and its history."""

    Params = SessionContextParams
    params: SessionContextParams
    lookback_bars: ClassVar[int] = DEFAULT_LONG_BARS
    requires: ClassVar[tuple[str, ...]] = (
        C.MID,
        C.HIGH,
        C.LOW,
        SESSION_ID,
        SECONDS_SINCE_OPEN,
        SECONDS_TO_CLOSE,
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Add the ``f_*`` columns."""
        group = session_group()
        since, to_close = pl.col(SECONDS_SINCE_OPEN), pl.col(SECONDS_TO_CLOSE)
        out = frame.with_columns(
            (pl.col(C.MID).log() - pl.col(C.MID).log().shift(1).over(group)).alias("_r1"),
            # the session's own length, so a half day still runs 0 to 1
            (since / (since + to_close + EPS)).alias("_phase"),
        )
        out = out.with_columns(
            *self._clock(),
            *self._long_lookback(group),
            *self._range(group),
        )
        out = self._join_session_history(out, group)
        return out.drop("_r1", "_phase", "_bucket")

    def _clock(self) -> list[pl.Expr]:
        """Time of day, as a phase in ``[0, 1]`` rather than a count of seconds."""
        since, to_close = pl.col(SECONDS_SINCE_OPEN), pl.col(SECONDS_TO_CLOSE)
        edge = self.params.edge_minutes * MINUTE
        return [
            pl.col("_phase").alias("f_session_phase"),
            to_close.cast(pl.Float64).alias("f_seconds_to_close"),
            (2 * math.pi * pl.col("_phase")).sin().alias("f_tod_sin"),
            (2 * math.pi * pl.col("_phase")).cos().alias("f_tod_cos"),
            (since <= edge).cast(pl.Float64).alias("f_is_opening"),
            (to_close <= edge).cast(pl.Float64).alias("f_is_closing"),
        ]

    def _long_lookback(self, group: list[str]) -> list[pl.Expr]:
        """The 30 min return and realized volatility."""
        bars = self.params.long_bars
        log_mid = pl.col(C.MID).log()
        return [
            (log_mid - log_mid.shift(bars).over(group)).alias(f"f_logret_{bars}"),
            pl.col("_r1").rolling_std(bars).over(group).alias(f"f_vol_{bars}"),
        ]

    def _range(self, group: list[str]) -> list[pl.Expr]:
        """Position inside the session's running high-low range, and the distance to each end.

        The extremes are expanding (``cum_max``/``cum_min``), so they are the day's high and
        low *so far* and never the whole day's.
        """
        high = pl.col(C.HIGH).cum_max().over(group)
        low = pl.col(C.LOW).cum_min().over(group)
        mid = pl.col(C.MID)
        session_open = mid.first().over(group)
        return [
            (high / low).log().mul(BPS).alias("f_range_bps"),
            # the bar high/low and the snapshot mid come from slightly different reads of the
            # same quotes, so the mid can sit a hair outside the range it should bound
            ((mid - low) / (high - low + EPS)).clip(0.0, 1.0).alias("f_range_position"),
            (mid / high).log().mul(BPS).alias("f_drawdown_bps"),
            (mid / low).log().mul(BPS).alias("f_runup_bps"),
            (mid / session_open).log().mul(BPS).alias("f_logret_since_open_bps"),
        ]

    def _join_session_history(self, frame: pl.DataFrame, group: list[str]) -> pl.DataFrame:
        """Add the overnight-gap and same-time-of-day families.

        Both need one row per session (or per session and bucket) rather than per bar, so
        they are aggregated, shifted back by at least one session and joined on. The join is
        bracketed by a row index because a feature set must return its rows in the order it
        got them, and a left join does not promise that.
        """
        frame = frame.with_columns(
            pl.int_range(pl.len()).alias(ROW),
            (pl.col(SECONDS_SINCE_OPEN) // self.params.bucket_seconds).alias("_bucket"),
        )
        overnight = self._overnight(frame)
        history = self._time_of_day_history(frame)
        joined = frame.join(overnight, on=[C.INSTRUMENT, SESSION_ID], how="left").join(
            history, on=[C.INSTRUMENT, SESSION_ID, "_bucket"], how="left"
        )
        gap = pl.col("f_overnight_gap_bps")
        return (
            joined.sort(ROW)
            .with_columns(
                gap.abs().alias("f_overnight_gap_abs_bps"),
                # positive when the day is giving the gap back: the reversion the supervisor
                # asked about, as one column, because a linear model cannot form the product
                (-gap.sign() * pl.col("f_logret_since_open_bps")).alias("f_gap_reversion_bps"),
            )
            .drop(ROW)
        )

    def _overnight(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Per session: the gap from the previous session's close, and that session's return."""
        per_session = (
            frame.group_by([C.INSTRUMENT, SESSION_ID])
            .agg(
                pl.col(C.MID).first().alias("_first"),
                pl.col(C.MID).last().alias("_last"),
            )
            .sort([C.INSTRUMENT, SESSION_ID])
        )
        previous_close = pl.col("_last").shift(1).over(C.INSTRUMENT)
        previous_first = pl.col("_first").shift(1).over(C.INSTRUMENT)
        return per_session.select(
            C.INSTRUMENT,
            SESSION_ID,
            (pl.col("_first") / previous_close).log().mul(BPS).alias("f_overnight_gap_bps"),
            (previous_close / previous_first).log().mul(BPS).alias("f_prev_session_logret_bps"),
        )

    def _time_of_day_history(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Per session and bucket: what this bucket did on the previous sessions.

        ``shift(1)`` after the rolling mean is what keeps the current session out of its own
        history, which is the whole point: the feature is what the stock *used* to do at this
        hour, known before the bucket starts.
        """
        per_bucket = (
            frame.group_by([C.INSTRUMENT, SESSION_ID, "_bucket"])
            .agg(
                (pl.col(C.MID).last() / pl.col(C.MID).first()).log().alias("_ret"),
                pl.col("_r1").std().alias("_vol"),
            )
            .sort([C.INSTRUMENT, "_bucket", SESSION_ID])
        )
        window = self.params.hist_sessions
        bucket_group = [C.INSTRUMENT, "_bucket"]
        return per_bucket.select(
            C.INSTRUMENT,
            SESSION_ID,
            "_bucket",
            pl.col("_ret")
            .rolling_mean(window, min_samples=1)
            .shift(1)
            .over(bucket_group)
            .mul(BPS)
            .alias("f_tod_hist_logret_bps"),
            pl.col("_vol")
            .rolling_mean(window, min_samples=1)
            .shift(1)
            .over(bucket_group)
            .alias("f_tod_hist_vol"),
        )

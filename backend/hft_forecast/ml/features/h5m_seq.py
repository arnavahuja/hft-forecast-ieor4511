"""h5m group: the raw recent history of each stock, laid out for a sequence model.

For each bar t and lag l = 0 .. length-1 (l = 0 is bar t itself, the newest), six raw L1
variables of bar t-l become columns ``f_seq_<var>_l<lag>``:

    ret       1-bar log mid return, bps
    imb       queue imbalance (bid_size - ask_size) / (bid_size + ask_size)
    spr       quoted spread, bps of the mid
    micro     microprice - mid, bps of the mid
    twap      mid - time-weighted mid of the bar, bps of the mid
    lquotes   log(1 + number of quote updates in the bar)

Nothing here is engineered beyond one bar: the point is to let the model (``h5m_gru``) learn
its own summaries of the last ``length`` minutes. Only non-negative shifts taken over the
session, so the set is causal and the history restarts every session.
"""

from __future__ import annotations

from typing import ClassVar, Final

import polars as pl
from pydantic import Field

from hft_forecast.core import columns as C
from hft_forecast.core.params import PluginParams
from hft_forecast.ml.base import BPS, FeatureSet
from hft_forecast.ml.features.registry import register_feature_set, session_group

VARIABLES: Final = ("ret", "imb", "spr", "micro", "twap", "lquotes")
EPS: Final = 1e-12


class SeqParams(PluginParams):
    """How many minutes of history each row carries."""

    length: int = Field(20, ge=2, le=60, description="Bars of history, newest first")


@register_feature_set(name="h5m_seq", version="1.0.0")
class H5mSeq(FeatureSet):
    """Lagged raw L1 variables: ``len(VARIABLES) x length`` columns."""

    Params = SeqParams
    params: SeqParams
    lookback_bars: ClassVar[int] = 60
    requires: ClassVar[tuple[str, ...]] = (
        C.BID,
        C.ASK,
        C.BID_SIZE,
        C.ASK_SIZE,
        C.MID,
        C.MID_TWAP,
        C.MICROPRICE,
        C.N_QUOTES,
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Add the ``f_seq_*`` columns."""
        group = session_group()
        mid = pl.col(C.MID)
        qb, qa = pl.col(C.BID_SIZE), pl.col(C.ASK_SIZE)
        raw = {
            "ret": (mid.log() - mid.log().shift(1).over(group)) * BPS,
            "imb": (qb - qa) / (qb + qa + EPS),
            "spr": (pl.col(C.ASK) - pl.col(C.BID)) / mid * BPS,
            "micro": (pl.col(C.MICROPRICE) - mid) / mid * BPS,
            "twap": (mid - pl.col(C.MID_TWAP)) / mid * BPS,
            "lquotes": pl.col(C.N_QUOTES).log1p(),
        }
        out = frame.with_columns(**{f"_seq_{k}": v for k, v in raw.items()})
        columns = [
            pl.col(f"_seq_{var}").shift(lag).over(group).alias(f"f_seq_{var}_l{lag:02d}")
            for var in VARIABLES
            for lag in range(self.params.length)
        ]
        return out.with_columns(columns).drop([f"_seq_{k}" for k in raw])

"""Forward-return labels with a session guard (§4.2).

The shift that produces the target mid is taken ``.over`` the session, so a label whose
target bar falls in a later session is null. That is the only session-crossing mechanism;
nothing downstream has to re-check it.
"""

from __future__ import annotations

from typing import Final

import polars as pl

from hft_forecast.core import columns as C
from hft_forecast.ml.base import (
    BPS,
    DIRECTION,
    FWD_LOGRET,
    LABEL_END,
    LABEL_VALID,
    MAGNITUDE_BPS,
    SECONDS_SINCE_OPEN,
    SECONDS_TO_CLOSE,
    SESSION_ID,
    HorizonSpec,
    require_columns,
)
from hft_forecast.ml.config import LabelParams

GROUP: Final = (C.INSTRUMENT, SESSION_ID)
NEEDS: Final = (C.TIMESTAMP, C.INSTRUMENT, SESSION_ID, C.MID, SECONDS_SINCE_OPEN, SECONDS_TO_CLOSE)
DROP_REASONS: Final = (
    "missing_mid",
    "crosses_session",
    "bar_gap",
    "stale_target",
    "opening_cross",
    "closing_cross",
)


def make_labels(frame: pl.DataFrame, spec: HorizonSpec, params: LabelParams) -> pl.DataFrame:
    """Add the label columns to a bar frame, keeping every row.

    Rows are never dropped here because the feature windows need the full grid; ``label_valid``
    marks the usable ones and the CV filters on it.

    Raises:
        ConfigError: If the frame is missing a column the labels need.
    """
    require_columns(frame, NEEDS, "make_labels")
    steps = spec.steps
    bar_seconds = int(spec.delta.astype("timedelta64[s]").astype(int))
    stale = pl.col(C.IS_STALE) if C.IS_STALE in frame.columns else pl.lit(0.0)

    target_mid = pl.col(C.MID).shift(-steps).over(GROUP)
    target_ts = pl.col(C.TIMESTAMP).shift(-steps).over(GROUP)
    fwd = (target_mid.log() - pl.col(C.MID).log()).alias(FWD_LOGRET)
    # a bar missing from the grid (before the session's first quote) makes the interval
    # longer than steps bars, which is the gap guard
    contiguous = (target_ts - pl.col(C.TIMESTAMP)).dt.total_seconds() == steps * bar_seconds
    reasons = {
        "missing_mid": pl.col(C.MID).is_null() | target_mid.is_null(),
        "crosses_session": target_ts.is_null(),
        "bar_gap": target_ts.is_not_null() & ~contiguous,
        "stale_target": (stale.shift(-steps).over(GROUP) == 1.0)
        if params.drop_stale_targets
        else pl.lit(False),
        "opening_cross": pl.col(SECONDS_SINCE_OPEN) < params.warmup_seconds,
        "closing_cross": pl.col(SECONDS_TO_CLOSE) < params.cooldown_seconds,
    }
    tau = params.dead_zone_bps / BPS
    out = frame.with_columns(
        fwd,
        pl.coalesce(target_ts, pl.col(C.TIMESTAMP)).alias(LABEL_END),
        *[r.fill_null(True).alias(f"_drop_{k}") for k, r in reasons.items()],
    )
    dropped = pl.any_horizontal([pl.col(f"_drop_{k}") for k in reasons])
    return out.with_columns(
        (pl.col(FWD_LOGRET).abs() * BPS).alias(MAGNITUDE_BPS),
        pl.when(pl.col(FWD_LOGRET) > tau)
        .then(1)
        .when(pl.col(FWD_LOGRET) < -tau)
        .then(-1)
        .otherwise(0)
        .cast(pl.Int8)
        .alias(DIRECTION),
        (~dropped).alias(LABEL_VALID),
    )


def label_report(frame: pl.DataFrame) -> dict[str, float]:
    """Class balance and the share of rows each drop rule removed."""
    n = max(frame.height, 1)
    valid = frame.filter(pl.col(LABEL_VALID))
    counts = {
        f"dropped_{k}": float(frame.get_column(f"_drop_{k}").sum()) / n
        for k in DROP_REASONS
        if f"_drop_{k}" in frame.columns
    }
    balance = {}
    if valid.height:
        for name, value in (("down", -1), ("flat", 0), ("up", 1)):
            balance[f"class_{name}"] = float(
                (valid.get_column(DIRECTION).to_numpy() == value).mean()
            )
    return {
        "n_rows": float(frame.height),
        "n_valid": float(valid.height),
        **balance,
        **counts,
    }


def drop_reason_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Remove the per-rule bookkeeping columns before the dataset is written."""
    return frame.drop([c for c in frame.columns if c.startswith("_drop_")])

"""The price impact regression of Cont, Kukanov and Stoikov (2014).

The paper's test: cut the session into equal intervals, sum the best-quote order flow
imbalance over each one, and regress the contemporaneous mid-price change on it,

    dP_k = alpha + beta * OFI_k + eps_k,

per stock and per day. Three results are worth reproducing on new data: the fit is close to
linear with a high R-squared, the slope is positive, and it falls roughly as one over the
displayed depth, so ``beta * depth`` is the quantity that is stable across names.

This is a **contemporaneous** regression, not a forecast: both sides are measured over the
same interval. It says how much of the price move the order flow accounts for, which is the
ceiling any forecast built on the same information is working under. The forecasting side of
the project uses the same imbalance as a feature (``ml.features.ofi``) over *past* bars.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import polars as pl

from hft_forecast.core import columns as C
from hft_forecast.core.types import FloatArray
from hft_forecast.ml.base import SESSION_ID, require_columns
from hft_forecast.ml.features.ofi import DEPTH, OFI, add_order_flow

BLOCK: Final = "_block"
DELTA_MID: Final = "_dmid"
MIN_INTERVALS: Final = 10
"""Fewer intervals than this in a session and the slope is noise, so the session is skipped."""
MIN_OBS_FOR_T: Final = 3
SIGNIFICANT_T: Final = 2.0
NEEDS: Final = (C.TIMESTAMP, C.INSTRUMENT, SESSION_ID, C.MID, C.BID, C.ASK, C.BID_SIZE, C.ASK_SIZE)


@dataclass(frozen=True)
class ImpactFit:
    """One stock-day regression of the mid-price change on the order flow imbalance."""

    instrument: str
    session: int
    n_intervals: int
    beta: float
    """Price move per unit of imbalance, in currency per share."""
    t_stat: float
    r2: float
    depth: float
    """Average displayed depth over the session, the ``D`` the slope scales with."""

    @property
    def beta_depth(self) -> float:
        """``beta * depth``: what the paper finds stable across names and days."""
        return self.beta * self.depth

    def as_row(self) -> dict[str, Any]:
        """Flat record for a CSV."""
        return {
            "instrument": self.instrument,
            "session": self.session,
            "n_intervals": self.n_intervals,
            "beta": self.beta,
            "t_stat": self.t_stat,
            "r2": self.r2,
            "depth": self.depth,
            "beta_depth": self.beta_depth,
        }


def interval_table(frame: pl.DataFrame, interval_bars: int) -> pl.DataFrame:
    """Sum the imbalance and the mid change over blocks of ``interval_bars`` bars.

    Blocks are cut inside an instrument-session, so no interval spans the overnight gap. The
    first block of a session is dropped: its first bar has no predecessor, so neither its
    imbalance nor its price change is defined.

    Raises:
        ConfigError: If a column the regression needs is missing.
    """
    require_columns(frame, NEEDS, "price impact")
    group = [C.INSTRUMENT, SESSION_ID]
    with_flow = add_order_flow(frame.sort([C.INSTRUMENT, SESSION_ID, C.TIMESTAMP]))
    blocked = with_flow.with_columns(
        pl.col(C.MID).diff().over(group).alias(DELTA_MID),
        (pl.int_range(pl.len()).over(group) // interval_bars).alias(BLOCK),
    )
    aggregated = blocked.group_by([*group, BLOCK], maintain_order=True).agg(
        pl.col(OFI).sum().alias(OFI),
        pl.col(DELTA_MID).sum().alias(DELTA_MID),
        pl.col(DEPTH).mean().alias(DEPTH),
        pl.col(OFI).null_count().alias("_nulls"),
        pl.len().alias("_bars"),
    )
    # a block holding a null is the one at the session open; a short block is the tail
    return aggregated.filter((pl.col("_nulls") == 0) & (pl.col("_bars") == interval_bars)).drop(
        "_nulls", "_bars"
    )


def _regress(x: FloatArray, y: FloatArray) -> tuple[float, float, float]:
    """OLS of ``y`` on ``x`` with an intercept: ``(slope, t statistic, r squared)``."""
    n = len(x)
    if n < MIN_OBS_FOR_T or np.ptp(x) == 0:
        return float("nan"), float("nan"), float("nan")
    design = np.column_stack([np.ones(n), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    residual = y - fitted
    total = float(((y - y.mean()) ** 2).sum())
    rss = float((residual**2).sum())
    r2 = 1.0 - rss / total if total > 0 else float("nan")
    variance = rss / (n - 2)
    sxx = float(((x - x.mean()) ** 2).sum())
    standard_error = float(np.sqrt(variance / sxx)) if sxx > 0 and variance > 0 else float("nan")
    slope = float(coefficients[1])
    return slope, slope / standard_error if standard_error else float("nan"), r2


def price_impact(
    frame: pl.DataFrame, interval_bars: int, min_intervals: int = MIN_INTERVALS
) -> list[ImpactFit]:
    """One regression per instrument-session, in the paper's stock-day form."""
    table = interval_table(frame, interval_bars)
    fits: list[ImpactFit] = []
    for (instrument, session), rows in table.group_by(
        [C.INSTRUMENT, SESSION_ID], maintain_order=True
    ):
        if rows.height < min_intervals:
            continue
        x = rows.get_column(OFI).to_numpy().astype(float)
        y = rows.get_column(DELTA_MID).to_numpy().astype(float)
        slope, t_stat, r2 = _regress(x, y)
        fits.append(
            ImpactFit(
                instrument=str(instrument),
                session=int(str(session)),
                n_intervals=rows.height,
                beta=slope,
                t_stat=t_stat,
                r2=r2,
                depth=float(rows.get_column(DEPTH).to_numpy().mean()),
            )
        )
    return fits


def depth_scaling(fits: list[ImpactFit]) -> float:
    """Slope of ``log beta`` on ``log depth`` across fits; the paper's claim is about -1.

    Only fits with a positive slope enter, since the logarithm of a negative impact is not
    defined and a negative slope is a failed day rather than a point on the curve.
    """
    usable = [f for f in fits if f.beta > 0 and f.depth > 0 and np.isfinite(f.beta)]
    if len(usable) < MIN_OBS_FOR_T:
        return float("nan")
    slope, _, _ = _regress(
        np.log(np.array([f.depth for f in usable])),
        np.log(np.array([f.beta for f in usable])),
    )
    return slope


def impact_summary(fits: list[ImpactFit]) -> dict[str, Any]:
    """Medians across stock-days plus the two claims the paper makes about the slope."""
    if not fits:
        return {"n_fits": 0}
    r2 = np.array([f.r2 for f in fits])
    beta = np.array([f.beta for f in fits])
    t_stat = np.array([f.t_stat for f in fits])
    return {
        "n_fits": len(fits),
        "n_instruments": len({f.instrument for f in fits}),
        "median_r2": float(np.nanmedian(r2)),
        "median_beta": float(np.nanmedian(beta)),
        "median_beta_depth": float(np.nanmedian([f.beta_depth for f in fits])),
        "share_positive_beta": float(np.mean(beta > 0)),
        "share_significant": float(np.mean(np.abs(t_stat) > SIGNIFICANT_T)),
        "depth_scaling_slope": depth_scaling(fits),
    }

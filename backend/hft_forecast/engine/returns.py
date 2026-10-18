"""Instrument return panels used by the engines.

Return basis, in order of preference:

1. A ``ret`` field (e.g. CRSP total returns including delisting returns ``dlret``).
2. Close prices plus explicit ``dividend`` (and ``split_ratio`` for raw prices).
3. Close prices alone (already adjusted, or no corporate actions available).

The chosen basis is recorded in the result metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hft_forecast.core import columns as C
from hft_forecast.core.types import Adjustment, FloatArray, MarketData


@dataclass(frozen=True)
class ReturnPanels:
    """Per-bar instrument returns.

    Attributes:
        close_to_close: Total return from close ``t-1`` to close ``t`` ``(T, N)``.
        overnight: Return from close ``t-1`` to open ``t`` (includes dividends) ``(T, N)``.
        intraday: Return from open ``t`` to close ``t`` ``(T, N)``.
        basis: Description of the return basis.
    """

    close_to_close: FloatArray
    overnight: FloatArray
    intraday: FloatArray
    basis: str


def _safe_ratio(num: FloatArray, den: FloatArray) -> FloatArray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den - 1.0
    out[~np.isfinite(out)] = np.nan
    return out


def _shift(panel: FloatArray) -> FloatArray:
    return np.vstack([np.full((1, panel.shape[1]), np.nan), panel[:-1]])


def compute_returns(data: MarketData) -> ReturnPanels:
    """Compute return panels from market data."""
    close = data.panel(C.CLOSE)
    prev_close = _shift(close)
    adjustment = str(data.metadata.get("adjustment", Adjustment.SPLIT.value))

    split = np.ones_like(close)
    if adjustment == Adjustment.RAW.value and data.has_field(C.SPLIT):
        split = np.nan_to_num(data.panel(C.SPLIT), nan=1.0)
        split[split <= 0] = 1.0
    dividend = np.zeros_like(close)
    has_div = data.has_field(C.DIVIDEND) and adjustment != Adjustment.TOTAL_RETURN.value
    if has_div:
        dividend = np.nan_to_num(data.panel(C.DIVIDEND), nan=0.0)

    c2c = _safe_ratio(split * (close + dividend), prev_close)
    basis = "close + dividends" if has_div else f"close ({adjustment})"
    if data.has_field(C.RETURN):
        # an explicit return field wins where present (CRSP returns with delistings, option
        # positions); instruments without it keep price-based returns
        ret = data.panel(C.RETURN)
        has_ret = np.isfinite(ret).any(axis=0)
        c2c = np.where(has_ret[None, :], ret, c2c)
        basis = (
            "ret field (total return incl. delisting)"
            if has_ret.all()
            else f"ret field where present, else {basis}"
        )

    if data.has_field(C.OPEN):
        open_ = data.panel(C.OPEN)
        overnight = _safe_ratio(split * (open_ + dividend), prev_close)
        intraday = _safe_ratio(close, open_)
        # keep the two legs consistent with the close-to-close basis
        with np.errstate(invalid="ignore"):
            implied = (1.0 + overnight) * (1.0 + intraday) - 1.0
        fix = np.isfinite(c2c) & ~np.isfinite(implied)
        overnight[fix] = c2c[fix]
        intraday[fix] = 0.0
    else:
        overnight = c2c.copy()
        intraday = np.zeros_like(c2c)
    return ReturnPanels(c2c, overnight, intraday, basis)

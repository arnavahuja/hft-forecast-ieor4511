"""Vectorized portfolio simulation kernels (no Python loops over bars).

The portfolio is rebalanced to target weights at rebalance steps and drifts with prices in
between. Between two rebalances holdings in units are constant, so the value of each
position relative to the portfolio value right after the rebalance at step ``s`` is::

    a_i(s, t) = E_i(s) * I_i(t) / I_i(s)

where ``I`` is the cumulative total-return index. The cash fraction ``1 - sum_i E_i(s)``
(which includes short-sale proceeds and is negative when levered) grows at the cash rate.
This closed form lets the whole path be computed with array indexing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from hft_forecast.core.types import BoolArray, FloatArray, IntArray

MIN_GROWTH: Final = 1e-12
"""Floor on index levels so a -100% return does not produce division by zero."""


@dataclass(frozen=True)
class SegmentPath:
    """Output of :func:`simulate_segments` on an ``M``-step grid.

    Attributes:
        returns: Portfolio return over each step, before costs ``(M,)``.
        pre_trade: Weights at the end of each step before any trade ``(M, N)``.
        post_trade: Weights at the end of each step after trading ``(M, N)``.
        trades: Signed weight traded at each step ``(M, N)``.
        start_of_step: Weights held during each step ``(M, N)`` (``post_trade`` shifted).
    """

    returns: FloatArray
    pre_trade: FloatArray
    post_trade: FloatArray
    trades: FloatArray
    start_of_step: FloatArray


def last_true_before(mask: BoolArray) -> IntArray:
    """For each step ``t``, the index of the last ``True`` strictly before ``t`` (or -1)."""
    idx = np.where(mask, np.arange(len(mask)), -1)
    last_incl = np.maximum.accumulate(idx) if len(idx) else idx
    return np.concatenate(([-1], last_incl[:-1])) if len(mask) else last_incl


def simulate_segments(
    asset_returns: FloatArray,
    targets: FloatArray,
    rebalance: BoolArray,
    cash_returns: FloatArray | None = None,
) -> SegmentPath:
    """Simulate a drifting portfolio rebalanced at selected steps.

    Args:
        asset_returns: Per-step instrument returns ``(M, N)``; ``NaN`` treated as 0.
        targets: Target weights ``(M, N)``; only rows where ``rebalance`` is true are used.
        rebalance: Boolean ``(M,)``; trading happens at the end of these steps.
        cash_returns: Per-step return on cash ``(M,)`` (default zero).

    Returns:
        The simulated path.
    """
    m, n = asset_returns.shape
    rets = np.nan_to_num(asset_returns, nan=0.0)
    cash = np.zeros(m) if cash_returns is None else np.nan_to_num(cash_returns, nan=0.0)
    tgt = np.nan_to_num(targets, nan=0.0)

    # Cumulative indices with a virtual leading row (index 0) = 1.0.
    index = np.vstack([np.ones((1, n)), np.cumprod(1.0 + rets, axis=0)])
    index = np.maximum(index, MIN_GROWTH)
    cash_index = np.concatenate(([1.0], np.cumprod(1.0 + cash)))

    # Targets at each rebalance, with a virtual all-cash row for "no rebalance yet".
    held_targets = np.vstack([np.zeros((1, n)), tgt])
    seg = last_true_before(rebalance) + 1  # 0 = virtual row
    t_idx = np.arange(1, m + 1)

    e = held_targets[seg]  # (M, N) weights set at the segment start
    cash_frac = 1.0 - e.sum(axis=1)
    growth_now = index[t_idx] / index[seg]
    growth_prev = index[t_idx - 1] / index[seg]
    cash_now = cash_index[t_idx] / cash_index[seg]
    cash_prev = cash_index[t_idx - 1] / cash_index[seg]

    pos_now = e * growth_now
    value_now = pos_now.sum(axis=1) + cash_frac * cash_now
    value_prev = (e * growth_prev).sum(axis=1) + cash_frac * cash_prev
    with np.errstate(divide="ignore", invalid="ignore"):
        step_ret = np.where(np.abs(value_prev) > MIN_GROWTH, value_now / value_prev - 1.0, 0.0)
        pre = np.where(np.abs(value_now)[:, None] > MIN_GROWTH, pos_now / value_now[:, None], 0.0)

    post = np.where(rebalance[:, None], tgt, pre)
    trades = np.where(rebalance[:, None], tgt - pre, 0.0)
    start = np.vstack([np.zeros((1, n)), post[:-1]]) if m else post
    return SegmentPath(step_ret, pre, post, trades, start)


def compound_pairs(first: FloatArray, second: FloatArray) -> FloatArray:
    """Combine two sub-step returns into one: ``(1 + a)(1 + b) - 1``."""
    return (1.0 + first) * (1.0 + second) - 1.0

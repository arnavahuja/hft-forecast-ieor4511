"""Rebalance schedules."""

from __future__ import annotations

import numpy as np

from hft_forecast.core.run_config import RebalanceRule
from hft_forecast.core.types import BoolArray, FloatArray, TimeArray


def _period_change(timestamps: TimeArray, unit: str) -> BoolArray:
    periods = timestamps.astype(f"datetime64[{unit}]")
    if unit == "W":
        # numpy weeks start on Thursday (epoch); shift so weeks start on Monday
        periods = (timestamps.astype("datetime64[D]") + np.timedelta64(3, "D")).astype(
            "datetime64[W]"
        )
    change = np.ones(len(timestamps), dtype=bool)
    if len(timestamps) > 1:
        change[1:] = periods[1:] != periods[:-1]
    return change


CHANGE_TOLERANCE = 1e-12


def rebalance_mask(
    timestamps: TimeArray,
    rule: RebalanceRule,
    every_n: int = 1,
    first_bar: int = 0,
    targets: FloatArray | None = None,
) -> BoolArray:
    """Boolean mask of bars at which a rebalance happens.

    Calendar rules fire on the first bar of each new week/month/quarter. ``on_change`` fires
    when ``targets`` (already lagged to trade bars) differ from the previous bar. ``first_bar``
    bars at the start are excluded (no decision exists yet because of the execution lag).
    """
    n = len(timestamps)
    if rule is RebalanceRule.ON_CHANGE:
        if targets is None:
            raise ValueError("on_change rebalancing needs targets")
        filled = np.nan_to_num(targets, nan=0.0)
        mask = np.zeros(n, dtype=bool)
        if n:
            mask[0] = bool(np.any(np.abs(filled[0]) > CHANGE_TOLERANCE))
            mask[1:] = np.any(np.abs(np.diff(filled, axis=0)) > CHANGE_TOLERANCE, axis=1)
        mask[: min(first_bar, n)] = False
        return mask
    if rule is RebalanceRule.EVERY_BAR:
        mask = np.ones(n, dtype=bool)
    elif rule is RebalanceRule.EVERY_N:
        mask = (np.arange(n) - first_bar) % every_n == 0
    elif rule is RebalanceRule.WEEKLY:
        mask = _period_change(timestamps, "W")
    elif rule is RebalanceRule.MONTHLY:
        mask = _period_change(timestamps, "M")
    else:
        months = timestamps.astype("datetime64[M]").astype(np.int64)
        quarters = months // 3
        mask = np.ones(n, dtype=bool)
        if n > 1:
            mask[1:] = quarters[1:] != quarters[:-1]
    mask[: min(first_bar, n)] = False
    if first_bar < n and rule is not RebalanceRule.EVERY_N:
        mask[first_bar] = True
    return mask

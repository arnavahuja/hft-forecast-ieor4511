"""Resampling bars to coarser frequencies (vectorized)."""

from __future__ import annotations

from typing import Final

import numpy as np

from hft_forecast.core import columns as C
from hft_forecast.core.errors import ConfigError
from hft_forecast.core.types import FloatArray, Frequency, IntArray, TimeArray

_UNITS: Final = {
    Frequency.S1: ("s", 1),
    Frequency.S5: ("s", 5),
    Frequency.S10: ("s", 10),
    Frequency.S30: ("s", 30),
    Frequency.MIN1: ("m", 1),
    Frequency.MIN5: ("m", 5),
    Frequency.MIN15: ("m", 15),
    Frequency.H1: ("h", 1),
    Frequency.D1: ("D", 1),
    Frequency.W1: ("W", 1),
    Frequency.MO1: ("M", 1),
}
WEEK_SHIFT_DAYS: Final = 3  # numpy weeks start on Thursday; shift so weeks start Monday


def period_codes(timestamps: TimeArray, frequency: Frequency) -> IntArray:
    """Integer period id per timestamp for the target frequency."""
    try:
        unit, size = _UNITS[frequency]
    except KeyError:
        raise ConfigError(f"Cannot resample to {frequency}") from None
    ts: TimeArray = timestamps
    if unit == "W":
        shifted = timestamps.astype("datetime64[D]") + np.timedelta64(WEEK_SHIFT_DAYS, "D")
        ts = np.asarray(shifted, dtype="datetime64[D]")
    codes = ts.astype(f"datetime64[{unit}]").astype(np.int64)
    return codes // size


def aggregate_blocks(block: FloatArray, starts: IntArray, field_name: str) -> FloatArray:
    """Aggregate consecutive row groups (``starts`` = first row of each group) of a panel.

    ``open``: first, ``high``: max, ``low``: min, ``volume``: sum, anything else: last.
    """
    ends = np.append(starts[1:], len(block)) - 1
    if field_name == C.OPEN:
        return block[starts]
    if field_name == C.HIGH:
        return np.fmax.reduceat(block, starts, axis=0)
    if field_name == C.LOW:
        return np.fmin.reduceat(block, starts, axis=0)
    if field_name == C.VOLUME:
        return np.add.reduceat(np.nan_to_num(block), starts, axis=0)
    return block[ends]

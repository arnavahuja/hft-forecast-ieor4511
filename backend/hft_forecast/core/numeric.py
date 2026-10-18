"""Pure, vectorized, NaN-aware numerical helpers for plugins.

All rolling functions are *trailing*: the value at row ``t`` uses rows ``t-window+1 .. t``
only, so they are safe to use in ``generate_targets`` without lookahead.
"""

from __future__ import annotations

import numpy as np

from hft_forecast.core.types import FloatArray


def _as2d(x: FloatArray) -> tuple[FloatArray, bool]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:, None], True
    return arr, False


def _restore(out: FloatArray, was_1d: bool) -> FloatArray:
    return out[:, 0] if was_1d else out


def _window_sum(values: FloatArray, window: int) -> FloatArray:
    csum = np.cumsum(values, axis=0)
    lagged = np.zeros_like(csum)
    if window < len(csum):
        lagged[window:] = csum[:-window]
    return csum - lagged


def rolling_sum(x: FloatArray, window: int, min_periods: int | None = None) -> FloatArray:
    """Trailing sum ignoring NaN; NaN where fewer than ``min_periods`` valid values."""
    arr, was_1d = _as2d(x)
    valid = np.isfinite(arr)
    total = _window_sum(np.where(valid, arr, 0.0), window)
    count = _window_sum(valid.astype(np.float64), window)
    out = np.where(count >= (min_periods or window), total, np.nan)
    return _restore(out, was_1d)


def rolling_mean(x: FloatArray, window: int, min_periods: int | None = None) -> FloatArray:
    """Trailing mean ignoring NaN."""
    arr, was_1d = _as2d(x)
    valid = np.isfinite(arr)
    total = _window_sum(np.where(valid, arr, 0.0), window)
    count = _window_sum(valid.astype(np.float64), window)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(count >= (min_periods or window), total / count, np.nan)
    return _restore(out, was_1d)


def rolling_std(
    x: FloatArray, window: int, min_periods: int | None = None, ddof: int = 1
) -> FloatArray:
    """Trailing standard deviation ignoring NaN."""
    arr, was_1d = _as2d(x)
    valid = np.isfinite(arr)
    clean = np.where(valid, arr, 0.0)
    count = _window_sum(valid.astype(np.float64), window)
    s1 = _window_sum(clean, window)
    s2 = _window_sum(clean * clean, window)
    with np.errstate(divide="ignore", invalid="ignore"):
        var = (s2 - s1 * s1 / count) / (count - ddof)
    var = np.where(var < 0, 0.0, var)
    out = np.where(count >= max(min_periods or window, ddof + 1), np.sqrt(var), np.nan)
    return _restore(out, was_1d)


def rolling_max(x: FloatArray, window: int) -> FloatArray:
    """Trailing maximum (NaN until ``window`` rows exist)."""
    arr, was_1d = _as2d(x)
    n = len(arr)
    out = np.full_like(arr, np.nan)
    if n >= window:
        view = np.lib.stride_tricks.sliding_window_view(arr, window, axis=0)
        out[window - 1 :] = np.nanmax(view, axis=-1)
    return _restore(out, was_1d)


def rolling_min(x: FloatArray, window: int) -> FloatArray:
    """Trailing minimum (NaN until ``window`` rows exist)."""
    arr, was_1d = _as2d(x)
    n = len(arr)
    out = np.full_like(arr, np.nan)
    if n >= window:
        view = np.lib.stride_tricks.sliding_window_view(arr, window, axis=0)
        out[window - 1 :] = np.nanmin(view, axis=-1)
    return _restore(out, was_1d)


def ewm_mean(x: FloatArray, halflife: float) -> FloatArray:
    """Exponentially weighted trailing mean with adjusted weights, ignoring NaN.

    Uses the recurrence ``y_t = d * y_{t-1} + x_t`` (evaluated in C by ``lfilter``) for both
    the weighted sum and the weight total.
    """
    from scipy.signal import lfilter

    arr, was_1d = _as2d(x)
    decay = 0.5 ** (1.0 / halflife)
    valid = np.isfinite(arr)
    num = lfilter([1.0], [1.0, -decay], np.where(valid, arr, 0.0), axis=0)
    den = lfilter([1.0], [1.0, -decay], valid.astype(np.float64), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return _restore(np.asarray(out, dtype=np.float64), was_1d)


def pct_change(x: FloatArray, periods: int = 1) -> FloatArray:
    """Percentage change over ``periods`` rows (NaN for the first rows)."""
    arr, was_1d = _as2d(x)
    out = np.full_like(arr, np.nan)
    if periods < len(arr):
        with np.errstate(divide="ignore", invalid="ignore"):
            out[periods:] = arr[periods:] / arr[:-periods] - 1.0
    out[~np.isfinite(out)] = np.nan
    return _restore(out, was_1d)


def shift(x: FloatArray, periods: int = 1) -> FloatArray:
    """Shift rows down by ``periods`` (NaN-filled)."""
    arr, was_1d = _as2d(x)
    out = np.full_like(arr, np.nan)
    if 0 < periods < len(arr):
        out[periods:] = arr[:-periods]
    elif periods == 0:
        out = arr.copy()
    return _restore(out, was_1d)


def ffill(x: FloatArray) -> FloatArray:
    """Forward-fill NaN along rows."""
    arr, was_1d = _as2d(x)
    idx = np.where(np.isfinite(arr), np.arange(len(arr))[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = arr[idx, np.arange(arr.shape[1])[None, :]]
    return _restore(out, was_1d)


def cross_sectional_rank(x: FloatArray) -> FloatArray:
    """Row-wise percentile rank in ``(0, 1]`` ignoring NaN (NaN stays NaN)."""
    arr = np.asarray(x, dtype=np.float64)
    valid = np.isfinite(arr)
    filled = np.where(valid, arr, np.inf)
    order = np.argsort(filled, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    np.put_along_axis(ranks, order, np.arange(1, arr.shape[1] + 1, dtype=np.float64), axis=1)
    count = valid.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = ranks / count
    return np.where(valid, pct, np.nan)


def normalize_gross(weights: FloatArray, gross: float = 1.0) -> FloatArray:
    """Scale each row so the sum of absolute weights equals ``gross`` (all-zero rows stay 0)."""
    arr = np.nan_to_num(np.asarray(weights, dtype=np.float64), nan=0.0)
    total = np.abs(arr).sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0, arr / total * gross, 0.0)

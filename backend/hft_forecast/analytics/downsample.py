"""Largest-Triangle-Three-Buckets downsampling for chart series."""

from __future__ import annotations

import numpy as np

from hft_forecast.core.types import FloatArray, IntArray


def lttb_indices(x: FloatArray, y: FloatArray, n_out: int) -> IntArray:
    """Indices of points kept by LTTB (Steinarsson 2013).

    The first and last points are always kept. Loops over output buckets (not input rows).
    """
    n = len(x)
    if n_out >= n or n_out < 3:
        return np.arange(n)
    y = np.nan_to_num(y, nan=0.0)
    edges = np.linspace(1, n - 1, n_out - 1).astype(np.int64)
    keep = np.empty(n_out, dtype=np.int64)
    keep[0], keep[-1] = 0, n - 1
    prev = 0
    for b in range(n_out - 2):
        lo, hi = edges[b], edges[b + 1]
        nxt_lo, nxt_hi = edges[b + 1], edges[b + 2] if b + 2 < len(edges) else n
        avg_x = x[nxt_lo:nxt_hi].mean() if nxt_hi > nxt_lo else x[-1]
        avg_y = y[nxt_lo:nxt_hi].mean() if nxt_hi > nxt_lo else y[-1]
        seg_x, seg_y = x[lo:hi], y[lo:hi]
        area = np.abs((x[prev] - avg_x) * (seg_y - y[prev]) - (x[prev] - seg_x) * (avg_y - y[prev]))
        prev = lo + int(np.argmax(area)) if area.size else lo
        keep[b + 1] = prev
    return keep

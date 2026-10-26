"""Pure robustness math: walk-forward windows, PBO (CSCV) and the breakeven cost multiple.

Used by ``ml.cv`` for the fold windows and by ``ml.deflate`` for overfitting and the
breakeven search; nothing here touches the engine or the run store.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from hft_forecast.core.types import FloatArray, IntArray

MIN_CSCV_SPLITS: Final = 2


# --------------------------------------------------------------------------- walk-forward


@dataclass(frozen=True)
class Window:
    """Train ``[train_start, train_end)`` and test ``[train_end, test_end)`` bar indices."""

    train_start: int
    train_end: int
    test_end: int


def walk_forward_windows(
    n: int, train: int, test: int, anchored: bool, start: int = 0
) -> list[Window]:
    """Rolling (or anchored) train/test windows covering ``[start, n)``."""
    out = []
    train_end = start + train
    while train_end < n:
        test_end = min(train_end + test, n)
        out.append(Window(start if anchored else train_end - train, train_end, test_end))
        train_end = test_end
    return out


# --------------------------------------------------------------------------- overfitting


def pbo_cscv(trial_returns: FloatArray, n_splits: int = 16) -> dict[str, Any]:
    """Probability of backtest overfitting via combinatorially symmetric cross-validation.

    Bailey, Borwein, Lopez de Prado and Zhu (2017). ``trial_returns`` is ``(T, N)``: one column
    per trial. The rows are split into ``n_splits`` blocks; for every half/half combination the
    best in-sample trial's out-of-sample rank gives a logit; PBO is the share of logits <= 0.

    Returns:
        ``pbo``, the ``logits`` and the number of combinations.
    """
    t, n = trial_returns.shape
    s = max(MIN_CSCV_SPLITS, n_splits - n_splits % 2)
    if n < MIN_CSCV_SPLITS or t < s * 2:
        return {"pbo": None, "logits": [], "combinations": 0}
    blocks: list[IntArray] = np.array_split(np.arange(t), s)
    combos = list(itertools.combinations(range(s), s // 2))
    logits = []
    for combo in combos:
        is_rows = np.concatenate([blocks[i] for i in combo])
        oos_rows = np.concatenate([blocks[i] for i in range(s) if i not in combo])
        is_sr = _sharpes(trial_returns[is_rows])
        oos_sr = _sharpes(trial_returns[oos_rows])
        best = int(np.nanargmax(is_sr))
        rank = (np.sum(oos_sr <= oos_sr[best]) - 0.5) / n  # relative rank in (0, 1)
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(math.log(rank / (1.0 - rank)))
    arr = np.array(logits)
    return {"pbo": float((arr <= 0).mean()), "logits": arr.tolist(), "combinations": len(combos)}


def _sharpes(block: FloatArray) -> FloatArray:
    sd = np.nanstd(block, axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.nanmean(block, axis=0) / sd
    return np.where(np.isfinite(sr), sr, -np.inf)


def breakeven(multiples: list[float], values: list[float | None]) -> float | None:
    """Linearly interpolated multiple at which a metric crosses zero (``None`` if never)."""
    pts = [(m, v) for m, v in zip(multiples, values, strict=True) if v is not None]
    for (m0, v0), (m1, v1) in itertools.pairwise(pts):
        if v0 >= 0 > v1 or v0 <= 0 < v1:
            return m0 + (0 - v0) * (m1 - m0) / (v1 - v0)
    return None

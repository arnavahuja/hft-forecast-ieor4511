"""Pure robustness math: parameter grids, walk-forward windows, Monte Carlo, PBO (CSCV)."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from hft_forecast.analytics import stats as S
from hft_forecast.core.types import FloatArray, IntArray

FAN_PERCENTILES: Final = (5.0, 25.0, 50.0, 75.0, 95.0)
MIN_CSCV_SPLITS: Final = 2


# --------------------------------------------------------------------------- grids


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of parameter values (sorted keys for determinism)."""
    keys = sorted(grid)
    return [
        dict(zip(keys, combo, strict=True)) for combo in itertools.product(*(grid[k] for k in keys))
    ]


def random_grid(
    space: dict[str, dict[str, Any]], n: int, rng: np.random.Generator
) -> list[dict[str, Any]]:
    """Random search points.

    ``space[p]`` is ``{"min", "max", "type": "int" | "float" | "log"}`` or ``{"values": [...]}``.
    """
    points = []
    for _ in range(n):
        point: dict[str, Any] = {}
        for name in sorted(space):
            spec = space[name]
            if "values" in spec:
                point[name] = spec["values"][int(rng.integers(len(spec["values"])))]
                continue
            lo, hi = float(spec["min"]), float(spec["max"])
            kind = spec.get("type", "float")
            if kind == "int":
                point[name] = int(rng.integers(int(lo), int(hi) + 1))
            elif kind == "log":
                point[name] = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
            else:
                point[name] = float(rng.uniform(lo, hi))
        points.append(point)
    return points


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


# --------------------------------------------------------------------------- monte carlo


def bootstrap_paths(
    returns: FloatArray, n_paths: int, horizon: int, mean_block: float, seed: int
) -> FloatArray:
    """Stationary-bootstrap return paths ``(n_paths, horizon)``."""
    r = returns[np.isfinite(returns)]
    rng = np.random.default_rng(seed)
    idx = S.stationary_bootstrap_indices(len(r), n_paths, mean_block, rng)
    if horizon <= len(r):
        return r[idx[:, :horizon]]
    reps = math.ceil(horizon / len(r))
    return np.tile(r[idx], reps)[:, :horizon]


def shuffle_trade_paths(trade_returns: FloatArray, n_paths: int, seed: int) -> FloatArray:
    """Random permutations of trade order ``(n_paths, n_trades)``."""
    rng = np.random.default_rng(seed)
    order = np.argsort(rng.random((n_paths, len(trade_returns))), axis=1)
    return trade_returns[order]


def path_stats(paths: FloatArray) -> dict[str, FloatArray]:
    """Terminal wealth and max drawdown per path, plus a percentile fan of wealth."""
    wealth = np.cumprod(1.0 + paths, axis=1)
    peak = np.maximum.accumulate(
        np.concatenate([np.ones((len(paths), 1)), wealth], axis=1), axis=1
    )[:, 1:]
    mdd = (wealth / peak - 1.0).min(axis=1)
    fan = np.percentile(wealth, FAN_PERCENTILES, axis=0)
    return {"terminal": wealth[:, -1], "max_drawdown": mdd, "fan": fan}


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

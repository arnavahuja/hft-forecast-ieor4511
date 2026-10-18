"""Pure portfolio construction math shared by constructor and overlay plugins.

Every estimate at row ``t`` uses rows ``<= t`` only (trailing windows).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Final

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

from hft_forecast.core.types import BoolArray, FloatArray, IntArray

MIN_VARIANCE: Final = 1e-12
RISK_PARITY_ITERATIONS: Final = 500
RISK_PARITY_TOL: Final = 1e-10
SHRINKAGE: Final = 0.1
"""Default shrinkage of sample covariance towards its diagonal."""


def simple_returns(prices: FloatArray) -> FloatArray:
    """Close-to-close returns with NaN for the first row and missing prices."""
    out = np.full_like(prices, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[1:] = prices[1:] / prices[:-1] - 1.0
    out[~np.isfinite(out)] = np.nan
    return out


def trailing_cov(
    returns: FloatArray, t: int, window: int, cols: IntArray, shrinkage: float = SHRINKAGE
) -> FloatArray:
    """Shrunk sample covariance of ``cols`` over rows ``t-window+1..t`` (NaN rows dropped)."""
    block = returns[max(0, t - window + 1) : t + 1][:, cols]
    block = block[np.all(np.isfinite(block), axis=1)]
    k = len(cols)
    if len(block) < max(k + 1, 3):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns are expected
            var = np.nanvar(returns[max(0, t - window + 1) : t + 1][:, cols], axis=0)
        return np.diag(np.where(np.isfinite(var) & (var > 0), var, MIN_VARIANCE))
    cov = np.cov(block, rowvar=False).reshape(k, k)
    diag = np.diag(np.diag(cov))
    return (1.0 - shrinkage) * cov + shrinkage * diag


def recompute_rows(signals: FloatArray, every: int) -> IntArray:
    """Rows where weights are recomputed: every ``every`` bars and when active signals change."""
    n = len(signals)
    active = np.sign(np.nan_to_num(signals, nan=0.0))
    change = np.zeros(n, dtype=bool)
    if n:
        change[0] = True
        change[1:] = np.any(active[1:] != active[:-1], axis=1)
    scheduled = (np.arange(n) % max(every, 1)) == 0
    return np.flatnonzero(change | scheduled)


def hold_between(values: FloatArray, rows: IntArray, n_rows: int) -> FloatArray:
    """Place ``values`` (one row per entry of ``rows``) and forward-fill to ``n_rows`` rows."""
    out = np.full((n_rows, values.shape[1]), np.nan)
    if len(rows) == 0:
        return out
    idx = np.searchsorted(rows, np.arange(n_rows), side="right") - 1
    valid = idx >= 0
    out[valid] = values[idx[valid]]
    return out


def inverse_vol_weights(cov: FloatArray) -> FloatArray:
    """Weights proportional to 1 / volatility (summing to 1)."""
    vol = np.sqrt(np.maximum(np.diag(cov), MIN_VARIANCE))
    inv = 1.0 / vol
    return inv / inv.sum()


def risk_parity_weights(cov: FloatArray) -> FloatArray:
    """Equal risk contribution weights (long-only, sum 1) by fixed-point iteration."""
    k = cov.shape[0]
    w = inverse_vol_weights(cov)
    for _ in range(RISK_PARITY_ITERATIONS):
        marginal = cov @ w
        port_var = float(w @ marginal)
        if port_var <= 0:
            break
        target = port_var / k
        new = w * np.sqrt(target / np.maximum(w * marginal, MIN_VARIANCE))
        new /= new.sum()
        if np.max(np.abs(new - w)) < RISK_PARITY_TOL:
            w = new
            break
        w = new
    return w


def min_variance_weights(cov: FloatArray, max_weight: float = 1.0) -> FloatArray:
    """Long-only minimum variance weights summing to 1 with an upper bound."""
    k = cov.shape[0]
    bound = max(max_weight, 1.0 / k)
    x0 = np.full(k, 1.0 / k)
    res = minimize(
        lambda w: float(w @ cov @ w),
        x0,
        jac=lambda w: 2.0 * cov @ w,
        bounds=[(0.0, bound)] * k,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        method="SLSQP",
    )
    w = res.x if res.success else x0
    return np.clip(w, 0.0, None) / max(np.clip(w, 0.0, None).sum(), MIN_VARIANCE)


def mean_variance_weights(
    mu: FloatArray,
    cov: FloatArray,
    risk_aversion: float,
    max_weight: float,
    long_only: bool,
    gross: float,
) -> FloatArray:
    """Maximize ``mu'w - risk_aversion/2 w'Cw`` subject to ``sum|w| <= gross`` and bounds."""
    k = len(mu)
    lo = 0.0 if long_only else -max_weight
    x0 = np.full(k, min(gross / k, max_weight)) if long_only else np.zeros(k)

    def objective(w: FloatArray) -> float:
        return float(-(mu @ w) + 0.5 * risk_aversion * (w @ cov @ w))

    def jac(w: FloatArray) -> FloatArray:
        return -mu + risk_aversion * (cov @ w)

    res = minimize(
        objective,
        x0,
        jac=jac,
        bounds=[(lo, max_weight)] * k,
        method="SLSQP",
        constraints=[{"type": "ineq", "fun": lambda w: gross - np.abs(w).sum()}],
    )
    return np.clip(res.x, lo, max_weight) if res.success else x0


def hrp_weights(cov: FloatArray) -> FloatArray:
    """Hierarchical risk parity (Lopez de Prado 2016), long-only, sum 1."""
    k = cov.shape[0]
    if k == 1:
        return np.ones(1)
    std = np.sqrt(np.maximum(np.diag(cov), MIN_VARIANCE))
    corr = np.clip(cov / np.outer(std, std), -1.0, 1.0)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    order = leaves_list(linkage(squareform(dist, checks=False), method="single"))
    w = np.ones(k)
    clusters: list[IntArray] = [order]
    while clusters:
        nxt: list[IntArray] = []
        for c in clusters:
            if len(c) <= 1:
                continue
            half = len(c) // 2
            left, right = c[:half], c[half:]
            var_l = _cluster_var(cov, left)
            var_r = _cluster_var(cov, right)
            alpha = 1.0 - var_l / (var_l + var_r) if var_l + var_r > 0 else 0.5
            w[left] *= alpha
            w[right] *= 1.0 - alpha
            nxt.extend([left, right])
        clusters = nxt
    return w / w.sum()


def _cluster_var(cov: FloatArray, idx: IntArray) -> float:
    sub = cov[np.ix_(idx, idx)]
    ivp = inverse_vol_weights(sub)
    return float(ivp @ sub @ ivp)


def signed_selection(row: FloatArray, allow_short: bool) -> tuple[IntArray, FloatArray]:
    """Columns with an active signal and their signs (+1/-1)."""
    finite = np.isfinite(row) & (row != 0)
    if not allow_short:
        finite &= row > 0
    cols = np.flatnonzero(finite)
    return cols, np.sign(row[cols])


def group_runs(mask: BoolArray) -> IntArray:
    """Run id per row that increments whenever ``mask`` is True (run starts)."""
    return np.cumsum(mask.astype(np.int64), axis=0)


WeighFn = Callable[[FloatArray, FloatArray, FloatArray], FloatArray]
"""``(cov, signs, signal_values) -> signed weights`` for the selected columns."""


def rolling_construct(
    signals: FloatArray,
    returns: FloatArray,
    weigh: WeighFn,
    *,
    every: int,
    window: int,
    allow_short: bool,
    gross: float,
    normalize: bool = True,
) -> FloatArray:
    """Compute weights at recompute rows from trailing covariance and hold them in between.

    Args:
        signals: ``(T, N)`` signals (NaN = no opinion).
        returns: ``(T, N)`` instrument returns used for covariance estimates.
        weigh: Function mapping ``(cov, signs, values)`` to signed weights.
        every: Recompute every N bars (and whenever the active set changes).
        window: Trailing window for covariance.
        allow_short: Keep negative signals as shorts.
        gross: Target gross exposure.
        normalize: Rescale each row to ``gross`` (False keeps the weigher's scale).

    Returns:
        ``(T, N)`` weights (0 where not held).
    """
    n_t, n_i = signals.shape
    rows = recompute_rows(signals, every)
    computed = np.zeros((len(rows), n_i))
    for k, t in enumerate(rows):
        cols, signs = signed_selection(signals[t], allow_short)
        if len(cols) == 0:
            continue
        cov = trailing_cov(returns, int(t), window, cols)
        w = weigh(cov, signs, signals[t, cols])
        total = np.abs(w).sum()
        if not normalize:
            computed[k, cols] = w
        elif total > 0:
            computed[k, cols] = w / total * gross
    held = hold_between(computed, rows, n_t)
    return np.nan_to_num(held, nan=0.0)

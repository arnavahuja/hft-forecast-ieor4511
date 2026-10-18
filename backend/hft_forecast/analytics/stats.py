"""Pure performance statistics on return arrays.

Conventions: ``returns`` are simple per-period returns; annualization uses the supplied
``periods_per_year`` (derived from the calendar, never hard-coded). Functions return ``nan``
when a statistic is undefined; the metric layer converts ``nan`` to ``None``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl
from scipy import stats as sps

from hft_forecast.core import columns as C
from hft_forecast.core.types import FloatArray, IntArray, TimeArray, np_times

EULER_GAMMA: Final = 0.5772156649015329
DEFAULT_VAR_LEVEL: Final = 0.05
MIN_OBS: Final = 2


def _clean(returns: FloatArray) -> FloatArray:
    r = np.asarray(returns, dtype=np.float64)
    return r[np.isfinite(r)]


# --------------------------------------------------------------------------- returns


def total_return(returns: FloatArray) -> float:
    """Compounded total return."""
    r = _clean(returns)
    return float(np.prod(1.0 + r) - 1.0) if r.size else math.nan


def cagr(returns: FloatArray, periods_per_year: float) -> float:
    """Compound annual growth rate."""
    r = _clean(returns)
    if r.size == 0:
        return math.nan
    growth = float(np.prod(1.0 + r))
    if growth <= 0:
        return -1.0
    return growth ** (periods_per_year / r.size) - 1.0


def annualized_mean(returns: FloatArray, periods_per_year: float) -> float:
    """Arithmetic mean times periods per year."""
    r = _clean(returns)
    return float(r.mean() * periods_per_year) if r.size else math.nan


def annualized_vol(returns: FloatArray, periods_per_year: float) -> float:
    """Sample standard deviation times sqrt(periods per year)."""
    r = _clean(returns)
    if r.size < MIN_OBS:
        return math.nan
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def downside_deviation(
    returns: FloatArray, periods_per_year: float, threshold: float = 0.0
) -> float:
    """Annualized root-mean-square of returns below ``threshold``."""
    r = _clean(returns)
    if r.size == 0:
        return math.nan
    down = np.minimum(r - threshold, 0.0)
    return float(np.sqrt(np.mean(down**2)) * np.sqrt(periods_per_year))


def pct_positive(returns: FloatArray) -> float:
    """Fraction of periods with a positive return (among non-zero periods)."""
    r = _clean(returns)
    active = r[r != 0]
    return float((active > 0).mean()) if active.size else math.nan


# --------------------------------------------------------------------------- periods


def period_codes(timestamps: TimeArray, unit: str) -> IntArray:
    """Integer period code per timestamp (``D``, ``M`` or ``Y``)."""
    return timestamps.astype(f"datetime64[{unit}]").astype(np.int64)


def aggregate_returns(
    returns: FloatArray, timestamps: TimeArray, unit: str
) -> tuple[IntArray, FloatArray]:
    """Compound returns within calendar periods. Returns ``(period_codes, returns)``."""
    r = np.nan_to_num(np.asarray(returns, dtype=np.float64), nan=0.0)
    if r.size == 0:
        return np.array([], dtype=np.int64), np.array([])
    codes = period_codes(timestamps, unit)
    uniq, first = np.unique(codes, return_index=True)
    log_growth = np.log1p(np.maximum(r, -1.0 + 1e-15))
    sums = np.add.reduceat(log_growth, first)
    return uniq, np.expm1(sums)


# --------------------------------------------------------------------------- risk


def skewness(returns: FloatArray) -> float:
    """Sample skewness."""
    r = _clean(returns)
    return float(sps.skew(r, bias=False)) if r.size > MIN_OBS else math.nan


def excess_kurtosis(returns: FloatArray) -> float:
    """Sample excess kurtosis."""
    r = _clean(returns)
    return float(sps.kurtosis(r, bias=False)) if r.size > MIN_OBS + 1 else math.nan


def var_historical(returns: FloatArray, level: float = DEFAULT_VAR_LEVEL) -> float:
    """Historical value at risk as a positive loss fraction."""
    r = _clean(returns)
    return float(-np.quantile(r, level)) if r.size else math.nan


def cvar_historical(returns: FloatArray, level: float = DEFAULT_VAR_LEVEL) -> float:
    """Historical conditional VaR (expected shortfall) as a positive loss fraction."""
    r = _clean(returns)
    if r.size == 0:
        return math.nan
    cutoff = np.quantile(r, level)
    tail = r[r <= cutoff]
    return float(-tail.mean()) if tail.size else math.nan


def var_parametric(returns: FloatArray, level: float = DEFAULT_VAR_LEVEL) -> float:
    """Gaussian VaR as a positive loss fraction."""
    r = _clean(returns)
    if r.size < MIN_OBS:
        return math.nan
    return float(-(r.mean() + sps.norm.ppf(level) * r.std(ddof=1)))


def cvar_parametric(returns: FloatArray, level: float = DEFAULT_VAR_LEVEL) -> float:
    """Gaussian expected shortfall as a positive loss fraction."""
    r = _clean(returns)
    if r.size < MIN_OBS:
        return math.nan
    z = sps.norm.ppf(level)
    return float(-(r.mean() - r.std(ddof=1) * sps.norm.pdf(z) / level))


def tail_ratio(returns: FloatArray, level: float = DEFAULT_VAR_LEVEL) -> float:
    """|95th percentile| / |5th percentile|."""
    r = _clean(returns)
    if r.size == 0:
        return math.nan
    low = abs(np.quantile(r, level))
    high = abs(np.quantile(r, 1.0 - level))
    return float(high / low) if low > 0 else math.nan


# --------------------------------------------------------------------------- risk-adjusted


def sharpe(returns: FloatArray, periods_per_year: float, risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio of excess returns."""
    r = _clean(returns) - risk_free_rate / periods_per_year
    if r.size < MIN_OBS:
        return math.nan
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else math.nan


def sortino(returns: FloatArray, periods_per_year: float, risk_free_rate: float = 0.0) -> float:
    """Annualized Sortino ratio (downside deviation below the risk-free rate)."""
    rf = risk_free_rate / periods_per_year
    r = _clean(returns)
    dd = downside_deviation(r, periods_per_year, rf)
    if not np.isfinite(dd) or dd == 0:
        return math.nan
    return float((r.mean() - rf) * periods_per_year / dd)


def omega(returns: FloatArray, threshold: float = 0.0) -> float:
    """Omega ratio at a per-period threshold."""
    r = _clean(returns) - threshold
    losses = -r[r < 0].sum()
    return float(r[r > 0].sum() / losses) if losses > 0 else math.nan


def gain_to_pain(returns: FloatArray) -> float:
    """Sum of returns divided by the absolute sum of negative returns."""
    r = _clean(returns)
    pain = -r[r < 0].sum()
    return float(r.sum() / pain) if pain > 0 else math.nan


def probabilistic_sharpe(returns: FloatArray, sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe ratio (Bailey and Lopez de Prado 2012).

    Args:
        returns: Per-period returns.
        sr_benchmark: Benchmark Sharpe ratio *per period* (not annualized).

    Returns:
        Probability that the true per-period Sharpe exceeds ``sr_benchmark``.
    """
    r = _clean(returns)
    n = r.size
    if n < MIN_OBS + 2:
        return math.nan
    sd = r.std(ddof=1)
    if sd == 0:
        return math.nan
    sr = r.mean() / sd
    g3 = sps.skew(r, bias=False)
    g4 = sps.kurtosis(r, fisher=False, bias=False)
    denom = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2
    if denom <= 0:
        return math.nan
    return float(sps.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / np.sqrt(denom)))


def expected_max_sharpe(trial_sharpes: FloatArray, n_trials: int) -> float:
    """Expected maximum per-period Sharpe among ``n_trials`` unskilled trials."""
    if n_trials <= 1:
        return 0.0
    sharpes = _clean(trial_sharpes)
    var = float(sharpes.var(ddof=1)) if sharpes.size >= MIN_OBS else 0.0
    if var <= 0:
        return 0.0
    z1 = sps.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = sps.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(np.sqrt(var) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe(returns: FloatArray, trial_sharpes: FloatArray, n_trials: int) -> float:
    """Deflated Sharpe ratio (Bailey and Lopez de Prado 2014).

    ``trial_sharpes`` are per-period Sharpe ratios of all trials in the experiment.
    """
    return probabilistic_sharpe(returns, expected_max_sharpe(trial_sharpes, n_trials))


# --------------------------------------------------------------------------- drawdowns


def wealth(returns: FloatArray) -> FloatArray:
    """Wealth index starting at 1 before the first return."""
    return np.cumprod(1.0 + np.nan_to_num(np.asarray(returns, dtype=np.float64), nan=0.0))


def drawdown_series(returns: FloatArray) -> FloatArray:
    """Drawdown from the running peak (<= 0), with the initial capital as first peak."""
    w = wealth(returns)
    peak = np.maximum.accumulate(np.concatenate(([1.0], w)))[1:]
    return w / peak - 1.0


def max_drawdown(returns: FloatArray) -> float:
    """Maximum drawdown as a negative fraction."""
    dd = drawdown_series(returns)
    return float(dd.min()) if dd.size else math.nan


@dataclass(frozen=True)
class DrawdownPeriod:
    """One drawdown episode (indices into the return array)."""

    start: int
    trough: int
    end: int | None
    depth: float

    @property
    def length(self) -> int:
        """Bars from peak to recovery (or to the last bar if unrecovered)."""
        return (self.end if self.end is not None else self.trough) - self.start

    @property
    def recovery(self) -> int | None:
        """Bars from trough to recovery."""
        return None if self.end is None else self.end - self.trough


def drawdown_periods(returns: FloatArray) -> list[DrawdownPeriod]:
    """All drawdown episodes, deepest first."""
    dd = drawdown_series(returns)
    underwater = dd < 0
    if not underwater.any():
        return []
    padded = np.concatenate(([False], underwater, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts, ends = edges[0::2], edges[1::2]
    out = []
    for s, e in zip(starts, ends, strict=True):
        trough = int(s + np.argmin(dd[s:e]))
        recovered = e < len(dd)
        out.append(
            DrawdownPeriod(
                start=int(s - 1) if s > 0 else 0,
                trough=trough,
                end=int(e) if recovered else None,
                depth=float(dd[trough]),
            )
        )
    return sorted(out, key=lambda p: p.depth)


def ulcer_index(returns: FloatArray) -> float:
    """Root-mean-square drawdown."""
    dd = drawdown_series(returns)
    return float(np.sqrt(np.mean(dd**2))) if dd.size else math.nan


# --------------------------------------------------------------------------- relative


@dataclass(frozen=True)
class RelativeStats:
    """Benchmark-relative statistics."""

    alpha: float
    beta: float
    correlation: float
    tracking_error: float
    information_ratio: float
    up_capture: float
    down_capture: float


def relative_stats(
    returns: FloatArray, benchmark: FloatArray, periods_per_year: float, risk_free_rate: float
) -> RelativeStats:
    """Alpha (annualized, CAPM), beta, correlation, tracking error, IR, captures."""
    r = np.asarray(returns, dtype=np.float64)
    b = np.asarray(benchmark, dtype=np.float64)
    ok = np.isfinite(r) & np.isfinite(b)
    r, b = r[ok], b[ok]
    nan = math.nan
    if r.size < MIN_OBS + 1 or b.std() == 0:
        return RelativeStats(nan, nan, nan, nan, nan, nan, nan)
    rf = risk_free_rate / periods_per_year
    cov = np.cov(r - rf, b - rf, ddof=1)
    beta = cov[0, 1] / cov[1, 1]
    alpha = ((r - rf).mean() - beta * (b - rf).mean()) * periods_per_year
    corr = float(np.corrcoef(r, b)[0, 1])
    active = r - b
    te = active.std(ddof=1) * np.sqrt(periods_per_year)
    ir = active.mean() * periods_per_year / te if te > 0 else nan
    up, down = b > 0, b < 0
    up_cap = r[up].mean() / b[up].mean() if up.any() else nan
    down_cap = r[down].mean() / b[down].mean() if down.any() else nan
    return RelativeStats(
        float(alpha), float(beta), corr, float(te), float(ir), float(up_cap), float(down_cap)
    )


# --------------------------------------------------------------------------- statistical


def mean_t_stat(returns: FloatArray) -> float:
    """t-statistic of the mean return."""
    r = _clean(returns)
    if r.size < MIN_OBS:
        return math.nan
    sd = r.std(ddof=1)
    return float(r.mean() / (sd / np.sqrt(r.size))) if sd > 0 else math.nan


def autocorrelation(returns: FloatArray, lag: int = 1) -> float:
    """Lag-k autocorrelation."""
    r = _clean(returns)
    if r.size <= lag + 1:
        return math.nan
    a, b = r[lag:], r[:-lag]
    if a.std() == 0 or b.std() == 0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def stationary_bootstrap_indices(
    n: int, n_samples: int, mean_block: float, rng: np.random.Generator
) -> IntArray:
    """Index matrix ``(n_samples, n)`` for the stationary bootstrap (Politis-Romano 1994)."""
    p = 1.0 / max(mean_block, 1.0)
    starts = rng.integers(0, n, size=(n_samples, n))
    new_block = rng.random((n_samples, n)) < p
    new_block[:, 0] = True
    # position within the current block, then wrap the index around the series
    first_col = np.where(new_block, np.arange(n)[None, :], 0)
    block_start_pos = np.maximum.accumulate(first_col, axis=1)
    offset = np.arange(n)[None, :] - block_start_pos
    start_idx = np.take_along_axis(starts, block_start_pos, axis=1)
    return (start_idx + offset) % n


def bootstrap_ci(
    returns: FloatArray,
    periods_per_year: float,
    seed: int,
    n_samples: int = 1000,
    mean_block: float = 5.0,
    level: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """Stationary-bootstrap confidence intervals for Sharpe and CAGR."""
    r = _clean(returns)
    if r.size < MIN_OBS * 5:
        return {"sharpe": (math.nan, math.nan), "cagr": (math.nan, math.nan)}
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(r.size, n_samples, mean_block, rng)
    samples = r[idx]
    sd = samples.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpes = samples.mean(axis=1) / sd * np.sqrt(periods_per_year)
    growth = np.prod(1.0 + samples, axis=1)
    cagrs = np.where(growth > 0, growth ** (periods_per_year / r.size) - 1.0, -1.0)
    lo, hi = (1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0
    return {
        "sharpe": (float(np.nanquantile(sharpes, lo)), float(np.nanquantile(sharpes, hi))),
        "cagr": (float(np.quantile(cagrs, lo)), float(np.quantile(cagrs, hi))),
    }


def nan_to_none(value: float | None) -> float | None:
    """Convert NaN/inf to ``None`` for JSON output."""
    if value is None:
        return None
    return float(value) if np.isfinite(value) else None


# --------------------------------------------------------------------------- factors

FACTOR_MIN_OBS: Final = 24


def ols(y: FloatArray, x: FloatArray) -> tuple[FloatArray, FloatArray, float]:
    """OLS with intercept. Returns ``(coefs, t_stats, r_squared)``, intercept first."""
    design = np.column_stack([np.ones(len(y)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    dof = max(len(y) - design.shape[1], 1)
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.pinv(design.T @ design)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, coef / se, np.nan)
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - (resid @ resid) / ss_tot if ss_tot > 0 else np.nan
    return coef, t, float(r2)


def _match_dates(r_ts: TimeArray, f_ts: TimeArray) -> tuple[IntArray, IntArray]:
    """Indices pairing return bars with factor rows: same day, else same month.

    Monthly data from different sources is dated differently (first day, calendar month
    end, last trading day); matching by month pairs them when daily matching finds little.
    """
    r_days = r_ts.astype("datetime64[D]")
    f_days = f_ts.astype("datetime64[D]")
    _, ri, fi = np.intersect1d(r_days, f_days, return_indices=True)
    if len(ri) >= FACTOR_MIN_OBS or len(r_days) == 0:
        return ri, fi
    r_months = r_ts.astype("datetime64[M]")
    f_months = f_ts.astype("datetime64[M]")
    if len(np.unique(r_months)) != len(r_months) or len(np.unique(f_months)) != len(f_months):
        return ri, fi
    _, ri_m, fi_m = np.intersect1d(r_months, f_months, return_indices=True)
    return (ri_m, fi_m) if len(ri_m) > len(ri) else (ri, fi)


def align_series(timestamps: TimeArray, other_ts: TimeArray, values: FloatArray) -> FloatArray:
    """``values`` (dated ``other_ts``) on the ``timestamps`` grid (day, else month match)."""
    out = np.full(len(timestamps), np.nan)
    ri, fi = _match_dates(timestamps, other_ts)
    out[ri] = values[fi]
    return out


def align_factors(
    timestamps: TimeArray, returns: FloatArray, factors: pl.DataFrame, names: tuple[str, ...]
) -> tuple[FloatArray, FloatArray, FloatArray] | None:
    """Join factor returns (and ``rf``) on dates. Returns ``(y_excess, X, rf)``."""
    cols = [c for c in names if c in factors.columns]
    if len(cols) != len(names):
        return None
    ri, fi = _match_dates(timestamps, np_times(factors.get_column(C.TIMESTAMP)))
    if len(ri) < FACTOR_MIN_OBS:
        return None
    x = factors.select(cols).to_numpy()[fi]
    rf = factors.get_column("rf").to_numpy()[fi] if "rf" in factors.columns else np.zeros(len(fi))
    y = returns[ri] - rf
    ok = np.isfinite(y) & np.isfinite(x).all(axis=1)
    return y[ok], x[ok], rf[ok]

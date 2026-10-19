"""Trial counting, deflated Sharpe and PBO (§7.4).

The trial count is the honest part: every configuration anyone evaluated under an experiment
is a trial, including the ones that were abandoned. The caller reads it from the ``RunStore``
(``services`` sits above ``ml``) and hands it in.
"""

from __future__ import annotations

import math
from typing import Any, Final

import numpy as np

from hft_forecast.analytics import stats as S
from hft_forecast.analytics.robustness.core import pbo_cscv
from hft_forecast.core.types import FloatArray

MIN_TRIALS_FOR_PBO: Final = 4
MIN_TRIALS_FOR_DSR: Final = 2


def deflated(
    net_returns: FloatArray, trial_sharpes: list[float], n_trials: int
) -> dict[str, float | None]:
    """Deflated and probabilistic Sharpe of one economic-test return series.

    Args:
        net_returns: Per-bar net returns of the run being judged.
        trial_sharpes: Per-bar Sharpe of every trial in the experiment.
        n_trials: How many configurations were evaluated, abandoned ones included.
    """
    if len(net_returns) == 0:
        return {"dsr": None, "psr": None, "n_trials": n_trials}
    psr = S.probabilistic_sharpe(net_returns)
    if n_trials < MIN_TRIALS_FOR_DSR or len(trial_sharpes) < MIN_TRIALS_FOR_DSR:
        return {"dsr": None, "psr": S.nan_to_none(psr), "n_trials": n_trials}
    dsr = S.deflated_sharpe(net_returns, np.asarray(trial_sharpes, dtype=float), n_trials)
    return {"dsr": S.nan_to_none(dsr), "psr": S.nan_to_none(psr), "n_trials": n_trials}


def overfitting(trial_returns: FloatArray) -> dict[str, Any]:
    """PBO over the ``(T, n_trials)`` matrix of aligned per-bar net returns.

    Reported only from four trials up; below that the combinatorial split has nothing to say.
    """
    if trial_returns.ndim != 2 or trial_returns.shape[1] < MIN_TRIALS_FOR_PBO:
        return {"pbo": None, "combinations": 0, "note": "needs at least 4 trials"}
    return pbo_cscv(trial_returns)


def bonferroni(p_value: float, n_trials: int) -> float:
    """``p * n_trials``, capped at 1: the adjustment for having looked ``n_trials`` times."""
    return min(1.0, p_value * max(n_trials, 1))


def t_stat_p_value(t_stat: float, n_obs: int) -> float:
    """Two-sided p-value of a t statistic, normal approximation for large samples."""
    if not math.isfinite(t_stat) or n_obs < MIN_TRIALS_FOR_DSR:
        return float("nan")
    return float(math.erfc(abs(t_stat) / math.sqrt(2.0)))

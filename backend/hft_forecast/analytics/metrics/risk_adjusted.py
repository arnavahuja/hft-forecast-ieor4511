"""Risk-adjusted performance metrics."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np

from hft_forecast.analytics import stats as S
from hft_forecast.core.interfaces import Metric, MetricContext, MetricOutput
from hft_forecast.core.registry import register
from hft_forecast.core.specs import MetricDescriptor
from hft_forecast.core.specs import MetricFormat as F

GROUP = "Risk-adjusted"


def _d(key: str, label: str, fmt: F = F.RATIO, desc: str = "") -> MetricDescriptor:
    return MetricDescriptor(
        key=key, label=label, group=GROUP, format=fmt, higher_is_better=True, description=desc
    )


@register("metric", name="risk_adjusted", version="1.0.0", tags=["core"])
class RiskAdjustedMetrics(Metric):
    """Sharpe, Sortino, Calmar, Omega, gain-to-pain, ulcer, PSR and deflated Sharpe."""

    descriptors: ClassVar[tuple[MetricDescriptor, ...]] = (
        _d("sharpe", "Sharpe"),
        _d("sortino", "Sortino"),
        _d("calmar", "Calmar", desc="CAGR / |max drawdown|"),
        _d("omega", "Omega"),
        _d("gain_to_pain", "Gain to pain"),
        MetricDescriptor(
            key="ulcer_index",
            label="Ulcer index",
            group=GROUP,
            format=F.PERCENT,
            higher_is_better=False,
        ),
        _d("psr", "Probabilistic Sharpe", F.PERCENT, "P(true Sharpe > 0)"),
        _d(
            "dsr",
            "Deflated Sharpe",
            F.PERCENT,
            "PSR against the expected max Sharpe of the experiment's trials",
        ),
    )

    def compute(self, ctx: MetricContext) -> dict[str, MetricOutput]:
        """Compute risk-adjusted metrics."""
        r, ppy = ctx.returns, ctx.periods_per_year
        excess, rf = ctx.excess_returns()
        mdd = S.max_drawdown(r)
        growth = S.cagr(r, ppy)
        calmar = growth / abs(mdd) if mdd < 0 and np.isfinite(growth) else math.nan
        trials = np.asarray(ctx.trial_sharpes, dtype=np.float64)
        nn = S.nan_to_none
        return {
            "sharpe": nn(S.sharpe(excess, ppy, rf)),
            "sortino": nn(S.sortino(excess, ppy, rf)),
            "calmar": nn(calmar),
            "omega": nn(S.omega(r)),
            "gain_to_pain": nn(S.gain_to_pain(r)),
            "ulcer_index": nn(S.ulcer_index(r)),
            "psr": nn(S.probabilistic_sharpe(r)),
            "dsr": nn(S.deflated_sharpe(r, trials, ctx.n_trials)),
        }

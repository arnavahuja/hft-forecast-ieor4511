"""Risk metrics."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from hft_forecast.analytics import stats as S
from hft_forecast.core.interfaces import Metric, MetricContext, MetricOutput
from hft_forecast.core.params import MetricParams
from hft_forecast.core.registry import register
from hft_forecast.core.specs import MetricDescriptor
from hft_forecast.core.specs import MetricFormat as F

GROUP = "Risk"


def _d(key: str, label: str, fmt: F = F.PERCENT, hib: bool | None = False) -> MetricDescriptor:
    return MetricDescriptor(key=key, label=label, group=GROUP, format=fmt, higher_is_better=hib)


class RiskParams(MetricParams):
    """Risk metric parameters."""

    var_level: float = Field(0.05, gt=0, lt=0.5, description="Tail probability for VaR/CVaR")


@register("metric", name="risk", version="1.0.0", tags=["core"])
class RiskMetrics(Metric):
    """Volatility, downside deviation, higher moments, VaR/CVaR and tail ratio."""

    Params = RiskParams
    params: RiskParams
    descriptors: ClassVar[tuple[MetricDescriptor, ...]] = (
        _d("ann_vol", "Annualized volatility"),
        _d("downside_dev", "Downside deviation"),
        _d("skew", "Skewness", F.NUMBER, True),
        _d("kurtosis", "Excess kurtosis", F.NUMBER, False),
        _d("var_hist", "VaR (historical)"),
        _d("cvar_hist", "CVaR (historical)"),
        _d("var_param", "VaR (parametric)"),
        _d("cvar_param", "CVaR (parametric)"),
        _d("tail_ratio", "Tail ratio", F.RATIO, True),
    )

    def compute(self, ctx: MetricContext) -> dict[str, MetricOutput]:
        """Compute risk metrics."""
        r, ppy, lvl = ctx.returns, ctx.periods_per_year, self.params.var_level
        nn = S.nan_to_none
        return {
            "ann_vol": nn(S.annualized_vol(r, ppy)),
            "downside_dev": nn(S.downside_deviation(r, ppy)),
            "skew": nn(S.skewness(r)),
            "kurtosis": nn(S.excess_kurtosis(r)),
            "var_hist": nn(S.var_historical(r, lvl)),
            "cvar_hist": nn(S.cvar_historical(r, lvl)),
            "var_param": nn(S.var_parametric(r, lvl)),
            "cvar_param": nn(S.cvar_parametric(r, lvl)),
            "tail_ratio": nn(S.tail_ratio(r, lvl)),
        }

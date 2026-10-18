"""Statistical significance metrics."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from hft_forecast.analytics import stats as S
from hft_forecast.core.interfaces import Metric, MetricContext, MetricOutput
from hft_forecast.core.params import MetricParams
from hft_forecast.core.registry import register
from hft_forecast.core.specs import MetricDescriptor
from hft_forecast.core.specs import MetricFormat as F

GROUP = "Statistical"


class StatisticalParams(MetricParams):
    """Parameters of the statistical metrics."""

    bootstrap_samples: int = Field(1000, ge=100, le=20000, description="Bootstrap resamples")
    mean_block: float = Field(5.0, ge=1, le=250, description="Mean bootstrap block length")
    seed: int = Field(12345, description="Random seed for the bootstrap")


def _d(key: str, label: str, fmt: F, hib: bool | None = None) -> MetricDescriptor:
    return MetricDescriptor(key=key, label=label, group=GROUP, format=fmt, higher_is_better=hib)


@register("metric", name="statistical", version="1.0.0", tags=["core", "statistics"])
class StatisticalMetrics(Metric):
    """t-stat of the mean, stationary-bootstrap CIs for Sharpe and CAGR, autocorrelation."""

    Params = StatisticalParams
    params: StatisticalParams
    descriptors: ClassVar[tuple[MetricDescriptor, ...]] = (
        _d("mean_t_stat", "t-stat of mean", F.NUMBER, True),
        _d("sharpe_ci_low", "Sharpe 95% CI low", F.RATIO, True),
        _d("sharpe_ci_high", "Sharpe 95% CI high", F.RATIO, True),
        _d("cagr_ci_low", "CAGR 95% CI low", F.PERCENT, True),
        _d("cagr_ci_high", "CAGR 95% CI high", F.PERCENT, True),
        _d("autocorr_1", "Autocorrelation (lag 1)", F.NUMBER),
    )

    def compute(self, ctx: MetricContext) -> dict[str, MetricOutput]:
        """Compute significance statistics."""
        p = self.params
        r = ctx.returns
        ci = S.bootstrap_ci(r, ctx.periods_per_year, p.seed, p.bootstrap_samples, p.mean_block)
        nn = S.nan_to_none
        return {
            "mean_t_stat": nn(S.mean_t_stat(r)),
            "sharpe_ci_low": nn(ci["sharpe"][0]),
            "sharpe_ci_high": nn(ci["sharpe"][1]),
            "cagr_ci_low": nn(ci["cagr"][0]),
            "cagr_ci_high": nn(ci["cagr"][1]),
            "autocorr_1": nn(S.autocorrelation(r, 1)),
        }

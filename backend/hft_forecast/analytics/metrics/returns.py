"""Return metrics."""

from __future__ import annotations

from typing import ClassVar

from hft_forecast.analytics import stats as S
from hft_forecast.core.interfaces import Metric, MetricContext, MetricOutput
from hft_forecast.core.registry import register
from hft_forecast.core.specs import MetricDescriptor
from hft_forecast.core.specs import MetricFormat as F

GROUP = "Return"


def _d(key: str, label: str, fmt: F = F.PERCENT, hib: bool | None = True) -> MetricDescriptor:
    return MetricDescriptor(key=key, label=label, group=GROUP, format=fmt, higher_is_better=hib)


@register("metric", name="returns", version="1.0.0", tags=["core"])
class ReturnMetrics(Metric):
    """Total return, CAGR, annualized mean, best/worst periods and hit rate."""

    descriptors: ClassVar[tuple[MetricDescriptor, ...]] = (
        _d("total_return", "Total return"),
        _d("cagr", "CAGR"),
        _d("ann_mean", "Annualized mean"),
        _d("best_bar", "Best bar"),
        _d("worst_bar", "Worst bar"),
        _d("best_month", "Best month"),
        _d("worst_month", "Worst month"),
        _d("best_year", "Best year"),
        _d("worst_year", "Worst year"),
        _d("pct_positive", "% positive bars"),
        _d("pct_positive_months", "% positive months"),
    )

    def compute(self, ctx: MetricContext) -> dict[str, MetricOutput]:
        """Compute return metrics."""
        r = ctx.returns
        ppy = ctx.periods_per_year
        ts = ctx.result.timestamps
        _, monthly = S.aggregate_returns(r, ts, "M")
        _, yearly = S.aggregate_returns(r, ts, "Y")
        nn = S.nan_to_none
        return {
            "total_return": nn(S.total_return(r)),
            "cagr": nn(S.cagr(r, ppy)),
            "ann_mean": nn(S.annualized_mean(r, ppy)),
            "best_bar": nn(float(r.max())) if r.size else None,
            "worst_bar": nn(float(r.min())) if r.size else None,
            "best_month": nn(float(monthly.max())) if monthly.size else None,
            "worst_month": nn(float(monthly.min())) if monthly.size else None,
            "best_year": nn(float(yearly.max())) if yearly.size else None,
            "worst_year": nn(float(yearly.min())) if yearly.size else None,
            "pct_positive": nn(S.pct_positive(r)),
            "pct_positive_months": nn(S.pct_positive(monthly)),
        }

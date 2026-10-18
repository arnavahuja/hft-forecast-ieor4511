"""Cost metrics."""

from __future__ import annotations

from typing import ClassVar

from hft_forecast.analytics import stats as S
from hft_forecast.core.interfaces import Metric, MetricContext, MetricOutput
from hft_forecast.core.registry import register
from hft_forecast.core.results import (
    COST_BORROW,
    COST_COMMISSION,
    COST_FEES,
    COST_FINANCING,
    COST_SLIPPAGE,
)
from hft_forecast.core.specs import MetricDescriptor
from hft_forecast.core.specs import MetricFormat as F

GROUP = "Costs"


def _d(key: str, label: str, fmt: F, hib: bool | None = False) -> MetricDescriptor:
    return MetricDescriptor(
        key=key, label=label, group=GROUP, format=fmt, higher_is_better=hib, benchmark=False
    )


@register("metric", name="costs", version="1.0.0", tags=["core", "costs"])
class CostMetrics(Metric):
    """Commissions, slippage, financing, cost drag and gross versus net return."""

    descriptors: ClassVar[tuple[MetricDescriptor, ...]] = (
        _d("total_commission", "Total commissions + fees", F.CURRENCY),
        _d("total_slippage", "Total slippage", F.CURRENCY),
        _d("total_financing", "Total borrow + financing", F.CURRENCY),
        _d("cost_drag", "Cost drag per year", F.PERCENT),
        _d("gross_cagr", "Gross CAGR", F.PERCENT, True),
        _d("gross_total_return", "Gross total return", F.PERCENT, True),
    )

    def compute(self, ctx: MetricContext) -> dict[str, MetricOutput]:
        """Compute cost totals."""
        if ctx.is_benchmark:
            return {d.key: None for d in self.descriptors}
        res = ctx.result

        def total(*cats: str) -> float:
            return float(sum(float(res.costs[c].sum()) for c in cats if c in res.costs))

        ppy = ctx.periods_per_year
        gross_cagr = S.cagr(res.gross_returns, ppy)
        nn = S.nan_to_none
        return {
            "total_commission": total(COST_COMMISSION, COST_FEES),
            "total_slippage": total(COST_SLIPPAGE),
            "total_financing": total(COST_BORROW, COST_FINANCING),
            "cost_drag": nn(gross_cagr - S.cagr(res.returns, ppy)),
            "gross_cagr": nn(gross_cagr),
            "gross_total_return": nn(S.total_return(res.gross_returns)),
        }

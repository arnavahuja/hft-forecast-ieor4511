"""Portfolio exposure and turnover metrics."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from hft_forecast.analytics import stats as S
from hft_forecast.core.interfaces import Metric, MetricContext, MetricOutput
from hft_forecast.core.registry import register
from hft_forecast.core.specs import MetricDescriptor
from hft_forecast.core.specs import MetricFormat as F

GROUP = "Portfolio"


def _d(key: str, label: str, fmt: F, hib: bool | None = None) -> MetricDescriptor:
    return MetricDescriptor(
        key=key, label=label, group=GROUP, format=fmt, higher_is_better=hib, benchmark=False
    )


@register("metric", name="portfolio", version="1.0.0", tags=["core"])
class PortfolioMetrics(Metric):
    """Turnover, gross/net exposure, leverage, concentration, holdings, long/short PnL."""

    descriptors: ClassVar[tuple[MetricDescriptor, ...]] = (
        _d("turnover_ann", "Turnover (ann., one-way)", F.PERCENT, False),
        _d("avg_gross", "Avg gross exposure", F.PERCENT),
        _d("avg_net", "Avg net exposure", F.PERCENT),
        _d("avg_leverage", "Avg leverage", F.RATIO),
        _d("max_leverage", "Max leverage", F.RATIO),
        _d("avg_hhi", "Avg concentration (HHI)", F.NUMBER),
        _d("avg_positions", "Avg positions", F.NUMBER),
        _d("long_contribution", "Long contribution", F.PERCENT, True),
        _d("short_contribution", "Short contribution", F.PERCENT, True),
    )

    def compute(self, ctx: MetricContext) -> dict[str, MetricOutput]:
        """Compute exposure metrics."""
        if ctx.is_benchmark:
            return {d.key: None for d in self.descriptors}
        res = ctx.result
        w = res.weights
        gross = res.gross_exposure
        with np.errstate(divide="ignore", invalid="ignore"):
            hhi = np.where(gross > 0, (w**2).sum(axis=1) / gross**2, np.nan)
            price_ret = np.nan_to_num(res.prices[1:] / res.prices[:-1] - 1.0)
        held = w[:-1]
        long_c = (np.clip(held, 0, None) * price_ret).sum(axis=1)
        short_c = (np.clip(held, None, 0) * price_ret).sum(axis=1)
        nn = S.nan_to_none
        invested = gross > 0
        return {
            "turnover_ann": nn(float(res.turnover.mean() * ctx.periods_per_year)),
            "avg_gross": nn(float(gross.mean())),
            "avg_net": nn(float(res.net_exposure.mean())),
            "avg_leverage": nn(float(gross[invested].mean())) if invested.any() else 0.0,
            "max_leverage": nn(float(gross.max())) if gross.size else None,
            "avg_hhi": nn(float(np.nanmean(hhi))) if invested.any() else None,
            "avg_positions": nn(float(res.n_positions.mean())),
            "long_contribution": nn(float(np.expm1(np.log1p(long_c).sum()))),
            "short_contribution": nn(float(np.expm1(np.log1p(short_c).sum()))),
        }

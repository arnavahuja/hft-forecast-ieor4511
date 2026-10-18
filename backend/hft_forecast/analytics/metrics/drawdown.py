"""Drawdown metrics."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from pydantic import Field

from hft_forecast.analytics import stats as S
from hft_forecast.core.interfaces import Metric, MetricContext, MetricOutput
from hft_forecast.core.params import MetricParams
from hft_forecast.core.registry import register
from hft_forecast.core.specs import MetricDescriptor, MetricKind
from hft_forecast.core.specs import MetricFormat as F

GROUP = "Drawdown"


class DrawdownParams(MetricParams):
    """Drawdown metric parameters."""

    top_n: int = Field(5, ge=1, le=50, description="Rows in the top drawdowns table")


@register("metric", name="drawdown", version="1.0.0", tags=["core"])
class DrawdownMetrics(Metric):
    """Max and average drawdown, durations, recovery and the top-N drawdowns table."""

    Params = DrawdownParams
    params: DrawdownParams
    descriptors: ClassVar[tuple[MetricDescriptor, ...]] = (
        MetricDescriptor(
            key="max_drawdown",
            label="Max drawdown",
            group=GROUP,
            format=F.PERCENT,
            higher_is_better=True,
        ),
        MetricDescriptor(
            key="avg_drawdown",
            label="Average drawdown",
            group=GROUP,
            format=F.PERCENT,
            higher_is_better=True,
        ),
        MetricDescriptor(
            key="max_dd_duration",
            label="Max drawdown duration",
            group=GROUP,
            format=F.BARS,
            higher_is_better=False,
        ),
        MetricDescriptor(
            key="max_recovery",
            label="Longest recovery",
            group=GROUP,
            format=F.BARS,
            higher_is_better=False,
        ),
        MetricDescriptor(
            key="top_drawdowns",
            label="Top drawdowns",
            group=GROUP,
            kind=MetricKind.TABLE,
            benchmark=False,
        ),
    )

    def compute(self, ctx: MetricContext) -> dict[str, MetricOutput]:
        """Compute drawdown metrics."""
        r = ctx.returns
        periods = S.drawdown_periods(r)
        ts = ctx.result.timestamps
        depths = np.array([p.depth for p in periods])
        recoveries = [p.recovery for p in periods if p.recovery is not None]
        table = [
            {
                "rank": k + 1,
                "start": str(ts[p.start])[:10],
                "trough": str(ts[p.trough])[:10],
                "end": str(ts[p.end])[:10] if p.end is not None else None,
                "depth": p.depth,
                "length": p.length,
                "recovery": p.recovery,
            }
            for k, p in enumerate(periods[: self.params.top_n])
        ]
        nn = S.nan_to_none
        return {
            "max_drawdown": nn(S.max_drawdown(r)),
            "avg_drawdown": nn(float(depths.mean())) if depths.size else 0.0,
            "max_dd_duration": float(max((p.length for p in periods), default=0)),
            "max_recovery": float(max(recoveries)) if recoveries else None,
            "top_drawdowns": table,
        }

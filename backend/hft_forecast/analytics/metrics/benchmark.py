"""Benchmark-relative metrics."""

from __future__ import annotations

from typing import ClassVar

from hft_forecast.analytics import stats as S
from hft_forecast.core.interfaces import Metric, MetricContext, MetricOutput
from hft_forecast.core.registry import register
from hft_forecast.core.specs import MetricDescriptor
from hft_forecast.core.specs import MetricFormat as F

GROUP = "Benchmark-relative"


def _d(key: str, label: str, fmt: F, hib: bool | None) -> MetricDescriptor:
    return MetricDescriptor(
        key=key, label=label, group=GROUP, format=fmt, higher_is_better=hib, benchmark=False
    )


@register("metric", name="benchmark_relative", version="1.0.0", tags=["core", "benchmark"])
class BenchmarkMetrics(Metric):
    """Alpha, beta, correlation, tracking error, information ratio, Treynor, captures."""

    descriptors: ClassVar[tuple[MetricDescriptor, ...]] = (
        _d("alpha", "Alpha (ann.)", F.PERCENT, True),
        _d("beta", "Beta", F.NUMBER, None),
        _d("correlation", "Correlation", F.NUMBER, None),
        _d("tracking_error", "Tracking error", F.PERCENT, None),
        _d("information_ratio", "Information ratio", F.RATIO, True),
        _d("treynor", "Treynor", F.RATIO, True),
        _d("up_capture", "Up capture", F.PERCENT, True),
        _d("down_capture", "Down capture", F.PERCENT, False),
        _d("active_return", "Active return (CAGR)", F.PERCENT, True),
    )

    def compute(self, ctx: MetricContext) -> dict[str, MetricOutput]:
        """Compute metrics versus the benchmark (``None`` without one)."""
        b = ctx.benchmark_returns
        if b is None or ctx.is_benchmark:
            return {d.key: None for d in self.descriptors}
        r, ppy, rf = ctx.returns, ctx.periods_per_year, ctx.risk_free_rate
        rel = S.relative_stats(r, b, ppy, rf)
        excess = S.annualized_mean(r, ppy) - rf
        treynor = excess / rel.beta if rel.beta and abs(rel.beta) > 1e-12 else float("nan")
        nn = S.nan_to_none
        return {
            "alpha": nn(rel.alpha),
            "beta": nn(rel.beta),
            "correlation": nn(rel.correlation),
            "tracking_error": nn(rel.tracking_error),
            "information_ratio": nn(rel.information_ratio),
            "treynor": nn(treynor),
            "up_capture": nn(rel.up_capture),
            "down_capture": nn(rel.down_capture),
            "active_return": nn(S.cagr(r, ppy) - S.cagr(b, ppy)),
        }

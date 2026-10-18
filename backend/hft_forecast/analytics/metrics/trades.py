"""Trade-level metrics."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from hft_forecast.analytics import stats as S
from hft_forecast.core.interfaces import Metric, MetricContext, MetricOutput
from hft_forecast.core.registry import register
from hft_forecast.core.specs import MetricDescriptor
from hft_forecast.core.specs import MetricFormat as F
from hft_forecast.core.types import BoolArray, FloatArray

GROUP = "Trades"


def _d(key: str, label: str, fmt: F, hib: bool | None) -> MetricDescriptor:
    return MetricDescriptor(
        key=key, label=label, group=GROUP, format=fmt, higher_is_better=hib, benchmark=False
    )


def max_streak(flags: BoolArray) -> int:
    """Longest run of True values."""
    if not flags.size:
        return 0
    padded = np.concatenate(([0], flags.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    return int((edges[1::2] - edges[0::2]).max()) if edges.size else 0


def _mean(values: FloatArray) -> float | None:
    return S.nan_to_none(float(np.mean(values))) if values.size else None


@register("metric", name="trades", version="1.0.0", tags=["core", "trades"])
class TradeMetrics(Metric):
    """Win rate, profit factor, payoff, expectancy, holding period, streaks, MAE/MFE."""

    descriptors: ClassVar[tuple[MetricDescriptor, ...]] = (
        _d("n_trades", "Number of trades", F.INTEGER, None),
        _d("win_rate", "Win rate", F.PERCENT, True),
        _d("profit_factor", "Profit factor", F.RATIO, True),
        _d("avg_win", "Average win", F.CURRENCY, True),
        _d("avg_loss", "Average loss", F.CURRENCY, True),
        _d("payoff_ratio", "Payoff ratio", F.RATIO, True),
        _d("expectancy", "Expectancy", F.CURRENCY, True),
        _d("avg_trade_return", "Avg trade return", F.PERCENT, True),
        _d("avg_holding", "Avg holding period", F.BARS, None),
        _d("max_consec_wins", "Max consecutive wins", F.INTEGER, True),
        _d("max_consec_losses", "Max consecutive losses", F.INTEGER, False),
        _d("avg_mae", "Avg MAE", F.PERCENT, True),
        _d("avg_mfe", "Avg MFE", F.PERCENT, True),
    )

    def compute(self, ctx: MetricContext) -> dict[str, MetricOutput]:
        """Compute trade statistics from closed trades."""
        trades = ctx.result.trades
        if ctx.is_benchmark or trades.height == 0:
            empty: dict[str, MetricOutput] = {d.key: None for d in self.descriptors}
            return empty | {"n_trades": 0.0}
        closed = trades.filter(~trades.get_column("is_open"))
        pnl = closed.get_column("pnl").to_numpy()
        wins, losses = pnl[pnl > 0], pnl[pnl < 0]
        nn = S.nan_to_none
        avg_win = float(wins.mean()) if wins.size else None
        avg_loss = float(losses.mean()) if losses.size else None
        return {
            "n_trades": float(trades.height),
            "win_rate": float((pnl > 0).mean()) if pnl.size else None,
            "profit_factor": nn(float(wins.sum() / -losses.sum())) if losses.size else None,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "payoff_ratio": nn(avg_win / -avg_loss) if avg_win and avg_loss else None,
            "expectancy": float(pnl.mean()) if pnl.size else None,
            "avg_trade_return": _mean(closed.get_column("return").to_numpy()),
            "avg_holding": _mean(trades.get_column("bars_held").to_numpy()),
            "max_consec_wins": float(max_streak(pnl > 0)),
            "max_consec_losses": float(max_streak(pnl < 0)),
            "avg_mae": _mean(trades.get_column("mae").to_numpy()),
            "avg_mfe": _mean(trades.get_column("mfe").to_numpy()),
        }

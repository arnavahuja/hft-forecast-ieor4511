"""Lookahead check: a strategy must make identical decisions on truncated data.

Runs ``generate_targets`` on the full history and on histories truncated at several cut
points, and asserts that the decisions up to each cut point are identical. Any difference
means the strategy used information from after the decision time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np

from hft_forecast.core.interfaces import PortfolioConstructor, Strategy
from hft_forecast.core.types import BoolArray, FloatArray, MarketData, TargetFrame
from hft_forecast.engine.base import align_targets

DEFAULT_CUTS: Final = 5
TOLERANCE: Final = 1e-9
MAX_EXAMPLES: Final = 5


@dataclass
class LookaheadReport:
    """Outcome of a lookahead check."""

    passed: bool
    cut_points: list[str] = field(default_factory=list)
    mismatches: list[dict[str, object]] = field(default_factory=list)
    message: str = ""


def _equal(a: FloatArray, b: FloatArray) -> BoolArray:
    both_nan = np.isnan(a) & np.isnan(b)
    return both_nan | (np.abs(np.nan_to_num(a) - np.nan_to_num(b)) <= TOLERANCE) & ~(
        np.isnan(a) ^ np.isnan(b)
    )


def cut_indices(n_bars: int, warmup: int, n_cuts: int = DEFAULT_CUTS) -> list[int]:
    """Evenly spaced cut points after the warm-up period."""
    start = min(max(warmup + 1, 1), n_bars - 1)
    if n_bars - 1 <= start:
        return [n_bars - 1] if n_bars > 1 else []
    return sorted({int(x) for x in np.linspace(start, n_bars - 2, n_cuts)})


def _targets(
    strategy: Strategy, data: MarketData, constructor: PortfolioConstructor | None
) -> TargetFrame:
    out = align_targets(strategy.generate_targets(data), data)
    if constructor is not None:
        from hft_forecast.core.calendar import periods_per_year_for
        from hft_forecast.core.interfaces import ConstructionContext

        out = constructor.construct(out, ConstructionContext(data, periods_per_year_for(data)))
    return out


def check_lookahead(
    strategy: Strategy,
    data: MarketData,
    n_cuts: int = DEFAULT_CUTS,
    constructor: PortfolioConstructor | None = None,
) -> LookaheadReport:
    """Compare full-history decisions with decisions made on truncated histories."""
    if not strategy.implements_vectorized():
        return LookaheadReport(True, message="Event-only strategy: Context enforces no lookahead")
    full = _targets(strategy, data, constructor)
    ts = data.timestamps
    cuts = cut_indices(len(ts), strategy.warmup(), n_cuts)
    report = LookaheadReport(True, [str(ts[c])[:19] for c in cuts])
    for cut in cuts:
        partial = _targets(strategy, data.truncate(ts[cut]), constructor)
        partial = partial.reindex(ts[: cut + 1], full.instruments)
        same = _equal(full.values[: cut + 1], partial.values)
        if not same.all():
            report.passed = False
            rows, cols = np.nonzero(~same)
            for r, c in list(zip(rows, cols, strict=True))[:MAX_EXAMPLES]:
                report.mismatches.append(
                    {
                        "cut": str(ts[cut])[:19],
                        "timestamp": str(ts[r])[:19],
                        "instrument": full.instruments[c],
                        "full": float(full.values[r, c]),
                        "truncated": float(partial.values[r, c]),
                    }
                )
    report.message = (
        "No lookahead detected"
        if report.passed
        else "Decisions changed when future data was removed: the strategy uses future data"
    )
    return report

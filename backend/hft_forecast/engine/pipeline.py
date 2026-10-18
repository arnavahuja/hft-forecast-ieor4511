"""Portfolio construction and overlay pipeline.

``Strategy output -> PortfolioConstructor -> Overlay 1 -> ... -> Engine``. Each overlay's
effect is recorded generically (no per-overlay code) in an :class:`OverlayReport`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from hft_forecast.core.errors import ConfigError
from hft_forecast.core.interfaces import (
    ConstructionContext,
    Overlay,
    OverlayContext,
    PortfolioConstructor,
)
from hft_forecast.core.results import OverlayReport
from hft_forecast.core.types import FloatArray, MarketData, TargetFrame, TargetKind

CHANGE_EPS: Final = 1e-12


@dataclass(frozen=True)
class NamedOverlay:
    """An overlay instance with its plugin name."""

    name: str
    overlay: Overlay


@dataclass(frozen=True)
class Pipeline:
    """Ordered constructor + overlays.

    Attributes:
        constructor: Turns signals into weights (required when the strategy emits signals).
        overlays: Overlays applied in order.
    """

    constructor: PortfolioConstructor | None = None
    overlays: tuple[NamedOverlay, ...] = field(default=())

    def construct(self, targets: TargetFrame, ctx: ConstructionContext) -> TargetFrame:
        """Apply the constructor to signals (weights pass through unchanged)."""
        if targets.kind is TargetKind.WEIGHTS and self.constructor is None:
            return targets
        if self.constructor is None:
            raise ConfigError("The strategy emits signals; choose a portfolio constructor")
        out = self.constructor.construct(targets, ctx)
        return out.replace(kind=TargetKind.WEIGHTS)

    def apply_overlays(
        self,
        targets: TargetFrame,
        data: MarketData,
        periods_per_year: float,
        benchmark: str | None,
        lag_bars: int,
        simulate: Callable[[TargetFrame], FloatArray],
        initial_capital: float = 1_000_000.0,
    ) -> tuple[TargetFrame, tuple[OverlayReport, ...]]:
        """Apply overlays in order and record what each changed."""
        reports: list[OverlayReport] = []
        current = targets
        for pos, named in enumerate(self.overlays):
            ctx = OverlayContext(
                data, periods_per_year, benchmark, lag_bars, simulate, [], initial_capital
            )
            before = current
            after = named.overlay.apply(before, ctx)
            if after.instruments != before.instruments:
                before = before.reindex(after.timestamps, after.instruments)
            diff = np.abs(after.filled() - before.filled())
            reports.append(
                OverlayReport(
                    name=named.name,
                    position=pos,
                    gross_before=np.abs(before.filled()).sum(axis=1),
                    gross_after=np.abs(after.filled()).sum(axis=1),
                    returns_before=simulate(before),
                    returns_after=simulate(after),
                    mean_abs_change=float(diff.sum(axis=1).mean()) if diff.size else 0.0,
                    cells_changed=int((diff > CHANGE_EPS).sum()),
                    notes=tuple(ctx.notes),
                )
            )
            current = after
        return current, tuple(reports)

"""Level-2 features. Stub: the employer's book format is not known yet (§3.8, §9.8)."""

from __future__ import annotations

from typing import ClassVar, Final

import polars as pl
from pydantic import Field

from hft_forecast.core import columns as C
from hft_forecast.core.errors import ConfigError
from hft_forecast.core.params import PluginParams
from hft_forecast.ml.base import FeatureSet, require_columns
from hft_forecast.ml.features.registry import register_feature_set

MAX_LEVELS: Final = 10
EPS: Final = 1e-12


class BookParams(PluginParams):
    """Depth of the book to use."""

    levels: int = Field(5, ge=1, le=MAX_LEVELS, description="Book levels to read")


@register_feature_set(name="book", version="0.1.0")
class Book(FeatureSet):
    """Level-N book features. Implemented only for depth-weighted imbalance so far.

    Raises at ``compute`` time when the book columns are absent, which is the normal case
    today: TAQ NBBO carries level 1 only. Fill this in when level-2 data arrives.
    """

    Params = BookParams
    params: BookParams
    lookback_bars: ClassVar[int] = 0

    @property
    def requires(self) -> tuple[str, ...]:  # type: ignore[override]  # depth is a parameter
        """Book columns for the configured depth."""
        return C.book_columns(self.params.levels)

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Depth-weighted imbalance across ``levels``.

        Raises:
            ConfigError: If the frame carries no order book.
        """
        columns = C.book_columns(self.params.levels)
        require_columns(frame, columns, f"book features ({self.params.levels} levels)")
        bids = [pl.col(f"bid_sz_{i}") for i in range(1, self.params.levels + 1)]
        asks = [pl.col(f"ask_sz_{i}") for i in range(1, self.params.levels + 1)]
        bid_depth, ask_depth = sum(bids[1:], bids[0]), sum(asks[1:], asks[0])
        return frame.with_columns(
            ((bid_depth - ask_depth) / (bid_depth + ask_depth + EPS)).alias("f_depth_imbalance")
        )

    def not_implemented(self) -> None:
        """Marker for the parts a group still has to write (queue sizes, order flow)."""
        raise ConfigError("Level-2 features beyond depth imbalance are not implemented")

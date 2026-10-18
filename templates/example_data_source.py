"""Example data source: a user data source."""

from __future__ import annotations

import polars as pl

from hft_forecast.core import DataSource, Instrument, MarketData, columns as C, register
from hft_forecast.core.types import DataRequest, DatasetInfo, Frequency
from hft_forecast.data.normalize import normalize_frame


@register("data_source", name="example_data_source", version="0.1.0", tags=["user"])
class ExampleDataSource(DataSource):
    """Reads CSV files named ``<symbol>.csv`` (date,close) from ``data/example_data_source/``."""

    def describe(self) -> list[DatasetInfo]:
        """Datasets offered."""
        return [DatasetInfo(source="example_data_source", dataset="daily", description="CSV prices",
                            frequencies=(Frequency.D1,))]

    def search_instruments(self, query: str) -> list[Instrument]:
        """Instruments matching the query."""
        folder = self.env.data_dir / "example_data_source"
        return [Instrument(id=p.stem, symbol=p.stem) for p in sorted(folder.glob("*.csv"))
                if query.upper() in p.stem.upper()]

    def fetch(self, request: DataRequest) -> MarketData:
        """Map your data to the canonical long schema: timestamp, instrument_id, fields."""
        frames = []
        for symbol in request.instruments:
            raw = pl.read_csv(self.env.data_dir / "example_data_source" / f"{symbol}.csv",
                              try_parse_dates=True)
            frames.append(raw.select(pl.col("date").alias(C.TIMESTAMP),
                                     pl.lit(symbol).alias(C.INSTRUMENT),
                                     pl.col("close").alias(C.CLOSE)))
        metadata = {"source": "example_data_source", "adjustment": request.adjustment.value}
        data = MarketData(normalize_frame(pl.concat(frames)), request.frequency,
                          metadata=metadata)
        return data.between(request.start, request.end)

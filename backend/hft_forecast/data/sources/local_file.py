"""Local file data source: datasets imported through the import wizard.

Each import lives in ``imports/<dataset_id>.parquet`` in canonical form with its mapping in
``imports/<dataset_id>.yaml``. Extra columns (custom signals, alternative data) are kept and
available to strategies as fields.
"""

from __future__ import annotations

from typing import Any, ClassVar

import polars as pl
import yaml

from hft_forecast.core import columns as C
from hft_forecast.core.errors import DataError
from hft_forecast.core.interfaces import DataSource
from hft_forecast.core.registry import register
from hft_forecast.core.types import (
    Adjustment,
    DataRequest,
    DatasetInfo,
    Frequency,
    Instrument,
    MarketData,
)
from hft_forecast.data.adjustments import convert
from hft_forecast.data.importer import ImportMapping, quote_metadata


@register(
    "data_source",
    name="local_file",
    version="1.0.0",
    tags=["local", "csv", "parquet", "excel"],
    capabilities={
        "asset:equity",
        "asset:future",
        "freq:daily",
        "freq:intraday",
        "freq:subminute",
    },
)
class LocalFileSource(DataSource):
    """Datasets imported from CSV, Parquet, Excel or Feather files."""

    source_version: ClassVar[str] = "1"
    cacheable: ClassVar[bool] = False

    def _mapping(self, dataset: str) -> dict[str, Any]:
        path = self.env.imports_dir / f"{dataset}.yaml"
        if not path.exists():
            return {}
        loaded: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        return loaded

    def describe(self) -> list[DatasetInfo]:
        """One dataset per imported file."""
        out = []
        for path in sorted(self.env.imports_dir.glob("*.parquet")):
            mapping = self._mapping(path.stem)
            schema = pl.read_parquet_schema(path)
            out.append(
                DatasetInfo(
                    source="local_file",
                    dataset=path.stem,
                    description=str(mapping.get("name", path.stem)),
                    frequencies=(Frequency(mapping.get("frequency", "1d")),),
                    fields=tuple(c for c in schema if c not in C.KEY_COLUMNS),
                    quality="user",
                    notes=str(mapping.get("source_file", "")),
                )
            )
        return out

    def search_instruments(self, query: str) -> list[Instrument]:
        """Instruments across all imported files matching the query."""
        q = query.strip().upper()
        found: set[str] = set()
        for path in self.env.imports_dir.glob("*.parquet"):
            ids = pl.scan_parquet(path).select(C.INSTRUMENT).unique().collect()
            found.update(i for i in ids.get_column(C.INSTRUMENT).to_list() if q in i.upper())
        return [Instrument(id=i, symbol=i) for i in sorted(found)]

    def fetch(self, request: DataRequest) -> MarketData:
        """Read an imported dataset, filtered by instruments and dates."""
        path = self.env.imports_dir / f"{request.dataset}.parquet"
        if not path.exists():
            raise DataError(f"Imported dataset '{request.dataset}' not found")
        mapping = self._mapping(request.dataset)
        lazy = pl.scan_parquet(path)
        if request.instruments:
            lazy = lazy.filter(pl.col(C.INSTRUMENT).is_in(list(request.instruments)))
        frame = lazy.collect()
        stored = Adjustment(mapping.get("adjustment", Adjustment.SPLIT.value))
        frame = convert(frame, stored, request.adjustment)
        data = MarketData(
            frame,
            Frequency(mapping.get("frequency", request.frequency.value)),
            metadata={
                "source": "local_file",
                "dataset": request.dataset,
                "adjustment": request.adjustment.value,
                **_quote_metadata(mapping, frame.columns),
            },
        )
        return data.between(request.start, request.end)


def _quote_metadata(mapping: dict[str, Any], columns: list[str]) -> dict[str, str]:
    """Bar labelling and price basis of an imported dataset (nothing for a legacy import)."""
    if not mapping:
        return {}
    return quote_metadata(ImportMapping.model_validate(mapping), columns)

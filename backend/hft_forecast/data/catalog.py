"""Dataset catalog queried with DuckDB.

Each dataset (a cache entry or an imported file) has a small JSON record in
``<data_dir>/catalog/``. DuckDB queries those records and the underlying Parquet files. Using
files instead of one DuckDB database file avoids cross-process write locks between the API
process and background job workers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
from pydantic import BaseModel, ConfigDict, Field

from hft_forecast.core.errors import NotFoundError
from hft_forecast.data.validation import QualityReport

RECORD_SUFFIX = ".json"
QUALITY_SUFFIX = ".quality.json"


class DatasetRecord(BaseModel):
    """Catalog entry for one dataset.

    Attributes:
        id: Dataset id (cache series key or import id).
        kind: ``cache`` (pulled from a source) or ``import`` (local file).
        source: Source plugin name.
        dataset: Dataset name within the source.
        name: Display name.
        frequency: Bar frequency.
        start: First timestamp (ISO).
        end: Last timestamp (ISO).
        instruments: Instrument ids.
        fields: Data fields.
        rows: Row count.
        last_refresh: ISO time of the last fetch or import.
        path: Parquet file with the current data.
        request: Original request (for refresh), without secrets.
        quality: Issue counts per severity.
        survivorship_bias_free: Whether the dataset includes delisted instruments.
        quality_label: Source quality label (e.g. ``lower`` for Yahoo).
        adjustment: Price adjustment convention.
        notes: Free-form notes.
    """

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    id: str
    kind: str
    source: str
    dataset: str
    name: str
    frequency: str
    start: str | None = None
    end: str | None = None
    instruments: list[str] = Field(default_factory=list)
    fields: list[str] = Field(default_factory=list)
    rows: int = 0
    last_refresh: str = ""
    path: str = ""
    request: dict[str, Any] = Field(default_factory=dict)
    quality: dict[str, int] = Field(default_factory=dict)
    survivorship_bias_free: bool = False
    quality_label: str = "standard"
    adjustment: str = ""
    notes: str = ""


class Catalog:
    """DuckDB-backed dataset catalog.

    Args:
        root: Directory holding the JSON records.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, dataset_id: str, suffix: str = RECORD_SUFFIX) -> Path:
        return self.root / f"{dataset_id}{suffix}"

    def upsert(self, record: DatasetRecord, quality: QualityReport | None = None) -> None:
        """Insert or replace a record (and its quality report)."""
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self._path(record.id, ".tmp")
        tmp.write_text(record.model_dump_json(indent=2))
        tmp.replace(self._path(record.id))
        if quality is not None:
            self._path(record.id, QUALITY_SUFFIX).write_text(quality.model_dump_json(indent=2))

    def get(self, dataset_id: str) -> DatasetRecord:
        """Fetch one record."""
        path = self._path(dataset_id)
        if not path.exists():
            raise NotFoundError(f"Dataset '{dataset_id}' not found")
        return DatasetRecord.model_validate_json(path.read_text())

    def quality(self, dataset_id: str) -> QualityReport:
        """Quality report of a dataset."""
        path = self._path(dataset_id, QUALITY_SUFFIX)
        if not path.exists():
            raise NotFoundError(f"No quality report for dataset '{dataset_id}'")
        return QualityReport.model_validate_json(path.read_text())

    def delete(self, dataset_id: str) -> None:
        """Remove a record (data files are left to the owner of the dataset)."""
        for suffix in (RECORD_SUFFIX, QUALITY_SUFFIX):
            self._path(dataset_id, suffix).unlink(missing_ok=True)

    def records(self, source: str | None = None, kind: str | None = None) -> list[DatasetRecord]:
        """All records, newest refresh first, optionally filtered, via DuckDB."""
        files = [p for p in self.root.glob(f"*{RECORD_SUFFIX}") if QUALITY_SUFFIX not in p.name]
        if not files:
            return []
        clauses, params = [], []
        if source:
            clauses.append("source = ?")
            params.append(source)
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = (
            "SELECT to_json(t) AS rec FROM read_json_auto(?, union_by_name=true, "
            f"format='auto') t {where} ORDER BY last_refresh DESC"
        )
        with duckdb.connect() as con:
            rows = con.execute(query, [[str(p) for p in files], *params]).fetchall()
        return [DatasetRecord.model_validate(json.loads(r[0])) for r in rows]

    def preview(self, dataset_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """First rows of a dataset's Parquet file (via DuckDB)."""
        record = self.get(dataset_id)
        with duckdb.connect() as con:
            cur = con.execute(
                "SELECT * FROM read_parquet(?) ORDER BY 1, 2 LIMIT ?", [record.path, limit]
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]

    def stats(self, dataset_id: str) -> dict[str, Any]:
        """Per-instrument row counts and date ranges (via DuckDB)."""
        record = self.get(dataset_id)
        with duckdb.connect() as con:
            cur = con.execute(
                "SELECT instrument_id, count(*) AS rows, min(timestamp) AS start, "
                "max(timestamp) AS end FROM read_parquet(?) GROUP BY 1 ORDER BY 1",
                [record.path],
            )
            cols = [d[0] for d in cur.description]
            return {"instruments": [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]}

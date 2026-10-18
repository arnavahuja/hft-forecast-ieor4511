"""Versioned Parquet cache keyed by a hash of the normalized request.

Layout::

    cache/<series_key>/meta.json
    cache/<series_key>/v0001.parquet
    cache/<series_key>/v0002.parquet   # after a refresh or an extension

``series_key`` is the request hash *without* dates, so requests that differ only in their
date range share an entry; a partial overlap only fetches the missing range. Every write
creates a new immutable version (a snapshot) so a past run can be reproduced exactly from the
version it recorded, even if Yahoo later restates its history.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from hft_forecast.core.types import DataRequest, Frequency, Instrument, MarketData

META_FILE = "meta.json"


@dataclass(frozen=True)
class CacheVersion:
    """One immutable snapshot of a cache entry."""

    version: int
    file: str
    start: date
    end: date
    fetched_at: str
    source_version: str
    rows: int

    def to_json(self) -> dict[str, Any]:
        """JSON representation."""
        return {
            "version": self.version,
            "file": self.file,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "fetched_at": self.fetched_at,
            "source_version": self.source_version,
            "rows": self.rows,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> CacheVersion:
        """Parse from JSON."""
        return cls(
            version=int(raw["version"]),
            file=str(raw["file"]),
            start=date.fromisoformat(raw["start"]),
            end=date.fromisoformat(raw["end"]),
            fetched_at=str(raw["fetched_at"]),
            source_version=str(raw["source_version"]),
            rows=int(raw["rows"]),
        )

    @property
    def ref(self) -> str:
        """Reference string ``<version>@<fetched_at>`` recorded by runs."""
        return f"v{self.version}@{self.fetched_at}"


@dataclass(frozen=True)
class CacheHit:
    """Result of a cache lookup."""

    key: str
    version: CacheVersion
    data: MarketData


@dataclass(frozen=True)
class MissingRange:
    """A date range the cache does not cover."""

    start: date
    end: date


class ParquetCache:
    """Parquet cache with versioned snapshots.

    Args:
        root: Cache directory.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    # ---- metadata

    def _dir(self, key: str) -> Path:
        return self.root / key

    def _read_meta(self, key: str) -> dict[str, Any] | None:
        path = self._dir(key) / META_FILE
        if not path.exists():
            return None
        meta: dict[str, Any] = json.loads(path.read_text())
        return meta

    def _write_meta(self, key: str, meta: dict[str, Any]) -> None:
        path = self._dir(key) / META_FILE
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, indent=2, default=str))
        tmp.replace(path)

    def versions(self, key: str) -> list[CacheVersion]:
        """All versions of an entry, oldest first."""
        meta = self._read_meta(key)
        if meta is None:
            return []
        return [CacheVersion.from_json(v) for v in meta.get("versions", [])]

    def latest(self, key: str) -> CacheVersion | None:
        """Latest version of an entry."""
        versions = self.versions(key)
        return versions[-1] if versions else None

    def entries(self) -> list[dict[str, Any]]:
        """Metadata of all cache entries."""
        out: list[dict[str, Any]] = []
        if not self.root.is_dir():
            return out
        for meta_path in sorted(self.root.glob(f"*/{META_FILE}")):
            meta = json.loads(meta_path.read_text())
            meta["key"] = meta_path.parent.name
            out.append(meta)
        return out

    # ---- lookup

    def missing_ranges(self, request: DataRequest) -> list[MissingRange]:
        """Date ranges of ``request`` not covered by the latest version."""
        latest = self.latest(request.key_without_dates())
        if latest is None:
            return [MissingRange(request.start, request.end)]
        gaps: list[MissingRange] = []
        if request.start < latest.start:
            gaps.append(MissingRange(request.start, latest.start - timedelta(days=1)))
        if request.end > latest.end:
            gaps.append(MissingRange(latest.end + timedelta(days=1), request.end))
        return gaps

    def load(self, request: DataRequest, version: int | None = None) -> CacheHit | None:
        """Load the cached data for a request (sliced to its dates) or ``None``."""
        key = request.key_without_dates()
        versions = self.versions(key)
        if not versions:
            return None
        matches = [v for v in versions if version is None or v.version == version]
        chosen = matches[-1] if matches else None
        if chosen is None:
            return None
        meta = self._read_meta(key) or {}
        frame = pl.read_parquet(self._dir(key) / chosen.file)
        instruments = {
            k: Instrument.model_validate(v) for k, v in meta.get("instruments", {}).items()
        }
        data = MarketData(
            frame,
            Frequency(meta.get("frequency", request.frequency.value)),
            instruments,
            metadata={
                **meta.get("data_metadata", {}),
                "cache_key": key,
                "cache_version": chosen.ref,
            },
        ).between(request.start, request.end)
        return CacheHit(key, chosen, data)

    # ---- store

    def store(
        self,
        request: DataRequest,
        data: MarketData,
        source_version: str,
        covered: tuple[date, date] | None = None,
    ) -> CacheVersion:
        """Write a new version for the request's entry.

        Args:
            request: The request (its date-free key identifies the entry).
            data: Full data for the covered range.
            source_version: Version string of the source plugin.
            covered: Date range the data covers (defaults to the request range).

        Returns:
            The new version.
        """
        key = request.key_without_dates()
        directory = self._dir(key)
        directory.mkdir(parents=True, exist_ok=True)
        meta = self._read_meta(key) or {
            "request": request.model_dump(mode="json", exclude={"start", "end"}),
            "frequency": request.frequency.value,
            "versions": [],
        }
        number = len(meta["versions"]) + 1
        file = f"v{number:04d}.parquet"
        data.frame.write_parquet(directory / file)
        start, end = covered or (request.start, request.end)
        version = CacheVersion(
            version=number,
            file=file,
            start=start,
            end=end,
            fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
            source_version=source_version,
            rows=data.frame.height,
        )
        meta["versions"].append(version.to_json())
        meta["data_metadata"] = {
            k: v for k, v in data.metadata.items() if isinstance(v, str | int | float | bool)
        }
        known = meta.get("instruments", {})
        known.update({k: v.model_dump(mode="json") for k, v in data.instrument_meta.items()})
        meta["instruments"] = known
        self._write_meta(key, meta)
        return version

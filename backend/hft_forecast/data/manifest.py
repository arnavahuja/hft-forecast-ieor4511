"""Named collections of cached partitions.

``hftf data pull-taq`` writes one manifest per pulled dataset; the ``ml`` layer scans the
listed Parquet files instead of re-requesting data. A partial pull leaves a usable manifest,
so an interrupted multi-hour pull can be resumed by re-running the same command.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from hft_forecast.core.errors import DataError


class ManifestEntry(BaseModel):
    """One cached partition: a single instrument over a single month."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    instrument: str
    dataset: str
    start: str
    end: str
    rows: int
    path: str


class Manifest(BaseModel):
    """Everything that makes up a named dataset (e.g. ``sp50_2024h1_5s``)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    source: str
    frequency: str
    created: str = ""
    entries: list[ManifestEntry] = []

    @property
    def instruments(self) -> list[str]:
        """Instruments with at least one partition."""
        return sorted({e.instrument for e in self.entries})

    @property
    def files(self) -> list[Path]:
        """Parquet files to scan, in manifest order."""
        return [Path(e.path) for e in self.entries]

    def merged(self, entries: list[ManifestEntry]) -> Manifest:
        """Copy with ``entries`` added, replacing any with the same key (a resumed pull)."""
        keyed = {(e.instrument, e.dataset, e.start): e for e in [*self.entries, *entries]}
        return self.model_copy(
            update={
                "entries": sorted(keyed.values(), key=lambda e: (e.dataset, e.instrument, e.start)),
                "created": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )

    def save(self, directory: Path) -> Path:
        """Write ``<directory>/<name>.yaml``."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.name}.yaml"
        path.write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False))
        return path

    @classmethod
    def load(cls, directory: Path, name: str) -> Manifest:
        """Read a manifest by name.

        Raises:
            DataError: If no such manifest exists.
        """
        path = directory / f"{name}.yaml"
        if not path.exists():
            raise DataError(f"No manifest '{name}' in {directory}")
        return cls.model_validate(yaml.safe_load(path.read_text()))

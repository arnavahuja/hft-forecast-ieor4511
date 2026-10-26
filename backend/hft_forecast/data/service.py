"""``DataService``: the single entry point for data.

Every fetch goes through the Parquet cache; every stored dataset is validated and registered
in the catalog.
"""

from __future__ import annotations

import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import structlog
import yaml

from hft_forecast.core import columns as C
from hft_forecast.core.calendar import TradingCalendar
from hft_forecast.core.errors import DataError
from hft_forecast.core.interfaces import DataSource, SourceEnvironment
from hft_forecast.core.registry import PluginKind, registry
from hft_forecast.core.types import DataRequest, MarketData
from hft_forecast.data.cache import ParquetCache
from hft_forecast.data.catalog import Catalog, DatasetRecord
from hft_forecast.data.importer import (
    ImportMapping,
    apply_mapping,
    quote_metadata,
    read_any,
)
from hft_forecast.data.normalize import normalize_frame
from hft_forecast.data.validation import QualityReport, validate

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class DataPaths:
    """Directories used by the data layer."""

    data_dir: Path
    cache_dir: Path
    imports_dir: Path
    catalog_dir: Path
    manifests_dir: Path

    @classmethod
    def under(cls, data_dir: Path) -> DataPaths:
        """Standard layout below ``data_dir``."""
        return cls(
            data_dir,
            data_dir / "cache",
            data_dir / "imports",
            data_dir / "catalog",
            data_dir / "manifests",
        )

    def ensure(self) -> None:
        """Create the directories."""
        for p in (self.cache_dir, self.imports_dir, self.catalog_dir, self.manifests_dir):
            p.mkdir(parents=True, exist_ok=True)


def _periods(start: date, end: date, by: str) -> list[tuple[date, date]]:
    """Inclusive [start, end] cut into calendar months or years."""
    out = []
    cursor = start
    while cursor <= end:
        if by == "month":
            nxt = date(cursor.year + cursor.month // 12, cursor.month % 12 + 1, 1)
        else:
            nxt = date(cursor.year + 1, 1, 1)
        out.append((cursor, min(end, nxt - timedelta(days=1))))
        cursor = nxt
    return out


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:40] or "dataset"


class DataService:
    """Fetch, cache, validate, import and catalog data.

    Args:
        paths: Data directories.
        calendar: Trading calendar for validation.
        wrds_username: WRDS user name handed to sources (password comes from ``~/.pgpass``).
    """

    def __init__(
        self,
        paths: DataPaths,
        calendar: TradingCalendar | None = None,
        wrds_username: str | None = None,
    ) -> None:
        paths.ensure()
        self.paths = paths
        self.calendar = calendar or TradingCalendar()
        self.cache = ParquetCache(paths.cache_dir)
        self.catalog = Catalog(paths.catalog_dir)
        self.env = SourceEnvironment(paths.data_dir, paths.imports_dir, wrds_username)

    # ---- sources

    def source(self, name: str, params: dict[str, Any] | None = None) -> DataSource:
        """Instantiate a data source plugin."""
        spec = registry(PluginKind.DATA_SOURCE).get(name)
        source: DataSource = spec.create(params, env=self.env)
        return source

    def sources(self) -> list[dict[str, Any]]:
        """All sources with their datasets and availability."""
        out = []
        for spec in registry(PluginKind.DATA_SOURCE).specs():
            try:
                src = self.source(spec.name)
                ok, reason = src.available()
                datasets = [d.model_dump(mode="json") for d in src.describe()] if ok else []
            except Exception as exc:
                ok, reason, datasets = False, str(exc), []
            out.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "available": ok,
                    "reason": reason,
                    "datasets": datasets,
                }
            )
        return out

    def search_instruments(self, source: str, query: str) -> list[dict[str, Any]]:
        """Search instruments in a source."""
        return [i.model_dump(mode="json") for i in self.source(source).search_instruments(query)]

    # ---- fetching

    def fetch(
        self,
        request: DataRequest,
        *,
        refresh: bool = False,
        cache_version: int | None = None,
        source_params: dict[str, Any] | None = None,
    ) -> MarketData:
        """Fetch data through the cache.

        Args:
            request: The request.
            refresh: Force a re-pull of the full range (creates a new cache version).
            cache_version: Load an exact cached snapshot (for reproducing a past run).
            source_params: Parameters for the source plugin.

        Returns:
            Market data sliced to the request's date range.
        """
        src = self.source(request.source, source_params)
        if not getattr(src, "cacheable", True):
            return src.fetch(request)
        if cache_version is not None:
            hit = self.cache.load(request, version=cache_version)
            if hit is None:
                raise DataError(f"Cache version {cache_version} not found")
            return hit.data
        gaps = [] if refresh else self.cache.missing_ranges(request)
        if not refresh and not gaps:
            hit = self.cache.load(request)
            if hit is not None:
                log.debug("cache_hit", key=hit.key)
                return hit.data
        latest = self.cache.latest(request.key_without_dates())
        if refresh or latest is None:
            full = src.fetch(request)
            covered = (request.start, request.end)
        else:
            full_req = request.model_copy(update={"start": latest.start, "end": latest.end})
            existing = self.cache.load(full_req)
            full = (
                existing.data
                if existing
                else MarketData(
                    pl.DataFrame({C.TIMESTAMP: [], C.INSTRUMENT: []}), request.frequency
                )
            )
            for gap in gaps:
                part = src.fetch(request.model_copy(update={"start": gap.start, "end": gap.end}))
                full = part if full.is_empty() else full.concat(part)
            covered = (min(request.start, latest.start), max(request.end, latest.end))
            log.info(
                "cache_extended",
                key=request.key_without_dates(),
                gaps=[(g.start.isoformat(), g.end.isoformat()) for g in gaps],
            )
        full = MarketData(
            normalize_frame(full.frame), full.frequency, full.instrument_meta, full.metadata
        )
        version = self.cache.store(request, full, src.source_version, covered)
        report = validate(full, self.calendar)
        self._register_cache(request, full, covered, version.ref, report)
        hit = self.cache.load(request)
        if hit is None:  # pragma: no cover - just stored
            raise DataError("Cache write failed")
        return hit.data

    def partitions(self, request: DataRequest, by: str = "month") -> list[DataRequest]:
        """Split a request into one request per instrument and per month (or per year).

        Quote data is pulled and cached in slices small enough to hold in memory: roughly
        100 k rows (1 min) to 500 k rows (5 s) per Parquet file.

        Raises:
            DataError: If ``by`` is not ``"month"`` or ``"year"``.
        """
        if by not in ("month", "year"):
            raise DataError(f"Unknown partition unit '{by}'")
        out = []
        for instrument in request.instruments or ("",):
            for start, end in _periods(request.start, request.end, by):
                update: dict[str, Any] = {"start": start, "end": end}
                if instrument:
                    update["instruments"] = (instrument,)
                out.append(request.model_copy(update=update))
        return out

    def fetch_partitioned(
        self, request: DataRequest, by: str = "month", **kwargs: Any
    ) -> MarketData:
        """Fetch a long request as per-instrument, per-month partitions and concatenate.

        Partitions that return no data are skipped, so a name that was not yet listed in part
        of the range does not fail the whole pull.
        """
        parts = [
            part
            for request_part in self.partitions(request, by)
            if (part := self._fetch_optional(request_part)) is not None
        ]
        if not parts:
            raise DataError("No data for any partition of the request")
        merged = parts[0]
        for part in parts[1:]:
            merged = merged.concat(part)
        return merged

    def _fetch_optional(self, request: DataRequest) -> MarketData | None:
        try:
            return self.fetch(request)
        except DataError as exc:
            log.warning("partition_empty", instruments=request.instruments, error=str(exc))
            return None

    def cached_file(self, request: DataRequest) -> Path | None:
        """Path of the Parquet file holding a cached request, if it is cached."""
        latest = self.cache.latest(request.key_without_dates())
        if latest is None:
            return None
        return self.paths.cache_dir / request.key_without_dates() / latest.file

    def _register_cache(
        self,
        request: DataRequest,
        data: MarketData,
        covered: tuple[date, date],
        version_ref: str,
        report: QualityReport,
    ) -> None:
        key = request.key_without_dates()
        meta = data.metadata
        latest = self.cache.latest(key)
        record = DatasetRecord(
            id=key,
            kind="cache",
            source=request.source,
            dataset=request.dataset,
            name=f"{request.source}:{request.dataset} {','.join(data.instruments[:5])}"
            + ("…" if len(data.instruments) > 5 else ""),
            frequency=request.frequency.value,
            start=covered[0].isoformat(),
            end=covered[1].isoformat(),
            instruments=list(data.instruments),
            fields=list(data.fields),
            rows=len(data),
            last_refresh=datetime.now(UTC).isoformat(timespec="seconds"),
            path=str(self.paths.cache_dir / key / latest.file) if latest else "",
            request=request.model_dump(mode="json", exclude={"start", "end"}),
            quality=report.summary,
            survivorship_bias_free=bool(meta.get("survivorship_bias_free", False)),
            quality_label=str(meta.get("quality", "standard")),
            adjustment=request.adjustment.value,
            notes=version_ref,
        )
        self.catalog.upsert(record, report)

    def refresh(self, dataset_id: str) -> DatasetRecord:
        """Force a re-pull of a cached dataset's full range."""
        record = self.catalog.get(dataset_id)
        if record.kind != "cache":
            raise DataError("Only pulled datasets can be refreshed; re-import local files")
        assert record.start is not None and record.end is not None
        request = DataRequest.model_validate(
            {**record.request, "start": record.start[:10], "end": record.end[:10]}
        )
        self.fetch(request, refresh=True)
        return self.catalog.get(dataset_id)

    # ---- local import

    def import_file(
        self, path: Path, mapping: ImportMapping, dataset_id: str | None = None
    ) -> DatasetRecord:
        """Import a local file into ``imports/`` and register it in the catalog.

        Args:
            path: Source file.
            mapping: Column mapping (from the wizard or a saved YAML).
            dataset_id: Reuse an id to re-import (replaces the data).

        Returns:
            The catalog record.
        """
        raw = read_any(path)
        canonical = apply_mapping(raw, mapping)
        dataset_id = dataset_id or f"{_slug(mapping.name or path.stem)}-{uuid.uuid4().hex[:6]}"
        meta = quote_metadata(mapping, canonical.columns)
        data = MarketData(canonical, mapping.frequency, metadata=meta)
        report = validate(data, self.calendar, raw=canonical)
        deduped = MarketData(normalize_frame(canonical), mapping.frequency, metadata=meta)
        target = self.paths.imports_dir / f"{dataset_id}.parquet"
        deduped.frame.write_parquet(target)
        saved = mapping.model_dump(mode="json")
        saved["source_file"] = str(path.name)
        (self.paths.imports_dir / f"{dataset_id}.yaml").write_text(yaml.safe_dump(saved))
        ts = deduped.timestamps
        record = DatasetRecord(
            id=dataset_id,
            kind="import",
            source="local_file",
            dataset=dataset_id,
            name=mapping.name or path.stem,
            frequency=mapping.frequency.value,
            start=str(ts[0]) if len(ts) else None,
            end=str(ts[-1]) if len(ts) else None,
            instruments=list(deduped.instruments),
            fields=list(deduped.fields),
            rows=len(deduped),
            last_refresh=datetime.now(UTC).isoformat(timespec="seconds"),
            path=str(target),
            quality=report.summary,
            survivorship_bias_free=False,
            quality_label="user",
            adjustment=mapping.adjustment.value,
            notes=f"imported from {path.name}",
        )
        self.catalog.upsert(record, report)
        log.info("dataset_imported", dataset=dataset_id, rows=len(deduped))
        return record

    def stage_upload(self, filename: str, content: bytes) -> Path:
        """Save an uploaded file to ``imports/uploads`` and return its path."""
        suffix = Path(filename).suffix.lower()
        uploads = self.paths.imports_dir / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        target = uploads / f"{_slug(Path(filename).stem)}-{uuid.uuid4().hex[:8]}{suffix}"
        target.write_bytes(content)
        return target

    def resolve_upload(self, token: str) -> Path:
        """Path of a staged upload by its file name (no directory traversal)."""
        path = (self.paths.imports_dir / "uploads" / Path(token).name).resolve()
        if not path.exists():
            raise DataError(f"Upload '{token}' not found")
        return path

    def saved_mapping(self, dataset_id: str) -> ImportMapping:
        """Mapping saved with an imported dataset (for one-click re-import)."""
        path = self.paths.imports_dir / f"{dataset_id}.yaml"
        if not path.exists():
            raise DataError(f"No saved mapping for '{dataset_id}'")
        raw = yaml.safe_load(path.read_text()) or {}
        raw.pop("source_file", None)
        return ImportMapping.model_validate(raw)

    def delete_dataset(self, dataset_id: str) -> None:
        """Delete a dataset's catalog record and data."""
        record = self.catalog.get(dataset_id)
        if record.kind == "import":
            for suffix in (".parquet", ".yaml"):
                (self.paths.imports_dir / f"{dataset_id}{suffix}").unlink(missing_ok=True)
        else:
            shutil.rmtree(self.paths.cache_dir / dataset_id, ignore_errors=True)
        self.catalog.delete(dataset_id)

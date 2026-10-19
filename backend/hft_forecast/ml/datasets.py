"""Bars in, one feature/label Parquet per instrument out (§3.9, §4.4).

Memory strategy: one instrument at a time, long frames only. The 5 s panel for 50 names is
29.5 M rows; nothing here ever pivots it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import structlog

from hft_forecast.core import columns as C
from hft_forecast.core.calendar import TradingCalendar
from hft_forecast.core.errors import ConfigError, DataError
from hft_forecast.core.types import BAR_LABEL_KEY, FloatArray, IntArray, MarketData
from hft_forecast.data.manifest import Manifest
from hft_forecast.data.service import DataService
from hft_forecast.ml.base import (
    LABEL_VALID,
    SECONDS_SINCE_OPEN,
    SECONDS_TO_CLOSE,
    SESSION_ID,
    require_columns,
)
from hft_forecast.ml.config import ExperimentConfig
from hft_forecast.ml.features.registry import FEATURE_SETS, feature_columns
from hft_forecast.ml.labels import drop_reason_columns, label_report, make_labels

log = structlog.get_logger(__name__)

FEATURES_DIR: Final = "features"
DATASET_META: Final = "dataset.json"


def load_bars(manifest: Manifest, instrument: str, dataset: str = "taq_nbbo_bars") -> pl.LazyFrame:
    """Scan one instrument's partitions of a manifest.

    Raises:
        DataError: If the manifest holds nothing for that instrument and dataset.
    """
    files = [
        e.path for e in manifest.entries if e.instrument == instrument and e.dataset == dataset
    ]
    if not files:
        raise DataError(f"Manifest '{manifest.name}' has no {dataset} for '{instrument}'")
    return pl.scan_parquet(files).sort(C.TIMESTAMP)


def with_session_columns(frame: pl.DataFrame, calendar: TradingCalendar) -> pl.DataFrame:
    """Add ``session_id`` and the seconds since the open / to the close of that session.

    Session identity comes from the exchange calendar in local time, never from truncating
    the UTC timestamp: a 16:00 New York close is 20:00 or 21:00 UTC depending on the season.
    """
    stamps = frame.get_column(C.TIMESTAMP).to_numpy().astype("datetime64[us]")
    sessions = calendar.session_ids(stamps)
    opens = np.full(len(stamps), np.datetime64("NaT", "us"), dtype="datetime64[us]")
    closes = opens.copy()
    for day in np.unique(sessions[sessions >= 0]):
        where = sessions == day
        start, end = calendar.session_bounds(day.astype("datetime64[D]").astype(object))
        opens[where], closes[where] = start, end
    since = (stamps - opens) / np.timedelta64(1, "s")
    until = (closes - stamps) / np.timedelta64(1, "s")
    return frame.with_columns(
        pl.Series(SESSION_ID, sessions),
        pl.Series(SECONDS_SINCE_OPEN, since),
        pl.Series(SECONDS_TO_CLOSE, until),
    ).filter(pl.col(SESSION_ID) >= 0)


def fetch_bars(config: ExperimentConfig, data: DataService, instrument: str) -> pl.DataFrame:
    """One instrument's bars, from the manifest when there is one, else through the cache.

    Raises:
        ConfigError: If the bars are not labelled at the bar end.
    """
    if config.manifest:
        manifest = Manifest.load(data.paths.manifests_dir, config.manifest)
        frame = load_bars(manifest, instrument).collect()
    else:
        fetched = data.fetch(config.data.to_request([instrument]))
        _check_bar_label(fetched)
        frame = fetched.frame
    return frame.sort(C.TIMESTAMP)


def _check_bar_label(data: MarketData) -> None:
    if data.metadata.get(BAR_LABEL_KEY) != "end":
        raise ConfigError(
            "ml needs bars labelled at the bar end",
            details={BAR_LABEL_KEY: data.metadata.get(BAR_LABEL_KEY)},
        )


def build_dataset(config: ExperimentConfig, data: DataService) -> Path:
    """Write ``<data_dir>/features/<hash>/<instrument>.parquet`` and return the directory.

    Already-built directories are reused: the hash covers the data, horizon, labels and
    feature sets, so a rebuild can only produce the same files.
    """
    directory = data.paths.data_dir / FEATURES_DIR / config.dataset_hash()
    if directory.exists() and any(directory.glob("*.parquet")):
        log.info("dataset_cached", path=str(directory))
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    sets = [FEATURE_SETS.get(ref.name).create(ref.params) for ref in config.features]
    calendar = TradingCalendar()
    total = 0
    for instrument in config.data.instruments:
        frame = with_session_columns(fetch_bars(config, data, instrument), calendar)
        for feature_set in sets:
            require_columns(frame, feature_set.requires, f"feature set '{feature_set}'")
            frame = feature_set.compute(frame)
        frame = make_labels(frame, config.horizon, config.labels)
        report = label_report(frame)
        frame = drop_reason_columns(frame)
        total += frame.height
        if total > config.max_rows_in_memory:
            raise ConfigError(
                f"dataset exceeds max_rows_in_memory ({config.max_rows_in_memory}); "
                "narrow instruments or the date range"
            )
        frame.write_parquet(directory / f"{instrument}.parquet")
        log.info("instrument_built", instrument=instrument, **report)
    return directory


def load_dataset(directory: Path, instruments: list[str] | None = None) -> pl.DataFrame:
    """Read a built dataset back, sorted by instrument then time (deterministic order)."""
    files = sorted(directory.glob("*.parquet"))
    if instruments is not None:
        wanted = set(instruments)
        files = [f for f in files if f.stem in wanted]
    if not files:
        raise DataError(f"No feature files in {directory}")
    return pl.concat([pl.read_parquet(f) for f in files]).sort([C.INSTRUMENT, C.TIMESTAMP])


def design_matrix(frame: pl.DataFrame) -> tuple[FloatArray, list[str]]:
    """The ``f_*`` columns as a float matrix, with NaN for nulls.

    float64 rather than the float32 of the design: sklearn upcasts anyway, and a fold's rows
    are a subset. A group whose model needs float32 casts inside its own ``fit``.
    """
    columns = feature_columns(frame)
    matrix: FloatArray = frame.select(columns).to_numpy().astype(np.float64)
    return matrix, columns


def usable_rows(frame: pl.DataFrame) -> pl.DataFrame:
    """Rows with a valid label and no missing feature."""
    columns = feature_columns(frame)
    finite = pl.all_horizontal([pl.col(c).is_finite() & pl.col(c).is_not_null() for c in columns])
    return frame.filter(pl.col(LABEL_VALID) & finite)


def windowed(x: FloatArray, lookback: int, session_id: IntArray) -> FloatArray:
    """``(n, lookback, f)`` sequence windows that never span a session boundary.

    Rows with fewer than ``lookback`` bars since the session open are filled with NaN, so a
    sequence model can mask them.
    """
    n, f = x.shape
    out = np.full((n, lookback, f), np.nan, dtype=x.dtype)
    starts = np.flatnonzero(np.diff(session_id, prepend=session_id[0] - 1) != 0)
    for begin, end in zip(starts, [*starts[1:], n], strict=True):
        block = x[begin:end]
        for k in range(lookback):
            rows = np.arange(lookback - 1, len(block))
            out[begin + rows, k] = block[rows - (lookback - 1 - k)]
    return out

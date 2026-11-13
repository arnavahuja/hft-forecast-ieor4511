"""Bars in, one feature/label Parquet per instrument out (§3.9, §4.4).

Memory strategy: one instrument at a time, long frames only. The 5 s panel for 50 names is
29.5 M rows; nothing here ever pivots it.
"""

from __future__ import annotations

import shutil
from datetime import date
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
    HorizonSpec,
    require_columns,
)
from hft_forecast.ml.config import ExperimentConfig
from hft_forecast.ml.features.registry import FEATURE_SETS, feature_columns
from hft_forecast.ml.labels import drop_reason_columns, label_report, make_labels

log = structlog.get_logger(__name__)

FEATURES_DIR: Final = "features"
DATASET_META: Final = "dataset.json"


def load_bars(manifest: Manifest, instrument: str, dataset: str = "taq_nbbo_bars") -> pl.LazyFrame:
    """Scan one instrument's partitions of a manifest, each sliced to the month it declares.

    The slice is what makes a manifest mean what it says. Partitions are cached by request
    without its dates, so every month of one instrument resolves to the *same* cache file,
    and a later, wider pull extends that file in place. Scanning the paths as they are would
    read an instrument once per partition and ignore the range the manifest was pulled for.

    Raises:
        DataError: If the manifest holds nothing for that instrument and dataset.
    """
    entries = [e for e in manifest.entries if e.instrument == instrument and e.dataset == dataset]
    if not entries:
        raise DataError(f"Manifest '{manifest.name}' has no {dataset} for '{instrument}'")
    day = pl.col(C.TIMESTAMP).dt.date()
    spans = [
        pl.scan_parquet(e.path).filter(
            day.is_between(date.fromisoformat(e.start), date.fromisoformat(e.end))
        )
        for e in entries
    ]
    # partitions pulled at different times hold the same columns in a different order, which
    # concat rejects; a select re-orders them and still raises if one is genuinely missing
    columns = spans[0].collect_schema().names()
    spans = [s.select(columns) for s in spans]
    return pl.concat(spans).unique(subset=[C.TIMESTAMP, C.INSTRUMENT]).sort(C.TIMESTAMP)


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
        ConfigError: If the bars are not labelled at the bar end, or are not on the
            horizon's grid.
    """
    if config.manifest:
        manifest = Manifest.load(data.paths.manifests_dir, config.manifest)
        # the manifest is a pull, not an experiment: it can cover more than this run asked
        # for, so the config's own window still decides what the dataset contains
        window = pl.col(C.TIMESTAMP).dt.date().is_between(config.data.start, config.data.end)
        frame = load_bars(manifest, instrument).filter(window).collect()
    else:
        fetched = data.fetch(config.data.to_request([instrument]))
        _check_bars(fetched, config.horizon)
        frame = fetched.frame
    return frame.sort(C.TIMESTAMP)


def _check_bars(data: MarketData, horizon: HorizonSpec) -> None:
    """The two things a dataset has to agree with before labels mean anything.

    The grid check matters for imported files, whose bar size is whatever the mapping
    declared: labelling 1 min bars as a 5 s grid would silently drop every row on the
    ``bar_gap`` rule instead of saying the frequency is wrong.
    """
    if data.metadata.get(BAR_LABEL_KEY) != "end":
        raise ConfigError(
            "ml needs bars labelled at the bar end",
            details={BAR_LABEL_KEY: data.metadata.get(BAR_LABEL_KEY)},
        )
    if data.frequency is not horizon.grid:
        raise ConfigError(
            f"data is {data.frequency} but horizon '{horizon.name}' is on a {horizon.grid} grid",
            details={"data": data.frequency.value, "grid": horizon.grid.value},
        )


def build_dataset(config: ExperimentConfig, data: DataService) -> Path:
    """Write ``<data_dir>/features/<hash>/<instrument>.parquet`` and return the directory.

    Already-built directories are reused: the hash covers the data, horizon, labels and
    feature sets, so a rebuild can only produce the same files.
    """
    final = data.paths.data_dir / FEATURES_DIR / config.dataset_hash()
    if final.exists():
        log.info("dataset_cached", path=str(final))
        return final
    # built under a temporary name and renamed at the end: a directory that exists is a
    # directory that is complete, so an interrupted build is never served as a cached one
    directory = final.with_name(f"{final.name}.partial")
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    sets = [FEATURE_SETS.get(ref.name).create(ref.params) for ref in config.features]
    calendar = TradingCalendar()
    total = 0
    for instrument in config.data.instruments:
        frame = with_session_columns(fetch_bars(config, data, instrument), calendar)
        for feature_set in sets:
            require_columns(frame, feature_set.requires, f"feature set '{feature_set}'")
            frame = feature_set.compute(frame)
        frame = _labelled(frame, config)
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
    add_cross_sectional(directory)
    directory.rename(final)
    return final


def _labelled(frame: pl.DataFrame, config: ExperimentConfig) -> pl.DataFrame:
    """Labels, measured on ``config.label_price`` instead of the snapshot mid when asked.

    The snapshot mid jumps half a spread whenever one side's quote flickers, so in a
    wide-spread name a large part of the forward return is bid-ask bounce that the
    imbalance features predict mechanically and no strategy can collect. ``mid_twap`` is
    duration-weighted across the bar and averages that out.

    ``make_labels`` is frozen shared code and always reads ``mid``, so the column is
    swapped for the call and swapped back. The feature sets have already run, so only the
    label columns see the substitute.
    """
    if config.label_price == C.MID:
        return make_labels(frame, config.horizon, config.labels)
    require_columns(frame, (config.label_price,), "label_price")
    snapshot_mid = frame.get_column(C.MID)
    # make_labels keeps every row, so the saved column still lines up afterwards
    labelled = make_labels(
        frame.with_columns(pl.col(config.label_price).alias(C.MID)), config.horizon, config.labels
    )
    return labelled.with_columns(snapshot_mid)


# The 10-bar return and the 1 min order flow: the two the market factor is worth having for.
# Columns a feature set does not produce are skipped, so this stays valid if a group retunes
# the `basic` lags or the `ofi` windows.
CROSS_SOURCES: Final = ("f_logret_10", "f_ofi_12")


def add_cross_sectional(directory: Path) -> None:
    """Add the cross-stock columns to every file in a built dataset, in place.

    For each source column the equal-weighted mean over the instruments trading that bar
    (the market factor) and the instrument's deviation from it. Same-bar only, so nothing
    here looks ahead. A single-instrument dataset gets nothing: the mean would be the
    column itself and the deviation identically zero.
    """
    files = sorted(directory.glob("*.parquet"))
    if len(files) < 2:
        return
    heads = pl.read_parquet_schema(files[0])
    sources = [c for c in CROSS_SOURCES if c in heads]
    if not sources:
        return
    market = (
        pl.concat([pl.scan_parquet(f).select(C.TIMESTAMP, *sources) for f in files])
        .group_by(C.TIMESTAMP)
        .agg([pl.col(c).mean().alias(f"f_mkt_{c[2:]}") for c in sources])
        .collect()
    )
    for file in files:
        frame = pl.read_parquet(file).join(market, on=C.TIMESTAMP, how="left")
        frame = frame.with_columns(
            (pl.col(c) - pl.col(f"f_mkt_{c[2:]}")).alias(f"f_rel_{c[2:]}") for c in sources
        )
        frame.write_parquet(file)
    log.info("cross_sectional_added", columns=sources, instruments=len(files))


def load_dataset(directory: Path, instruments: list[str] | None = None) -> pl.DataFrame:
    """Read a built dataset back, sorted by instrument then time (deterministic order)."""
    files = sorted(directory.glob("*.parquet"))
    if instruments is not None:
        wanted = set(instruments)
        files = [f for f in files if f.stem in wanted]
    if not files:
        raise DataError(f"No feature files in {directory}")
    frames = [pl.read_parquet(f) for f in files]
    # instruments can reach the dataset by different routes (a manifest partition, the
    # cache, an imported file) and those write the same columns in a different order; a
    # select re-orders them, and still raises if one is genuinely missing
    columns = frames[0].columns
    return pl.concat([f.select(columns) for f in frames]).sort([C.INSTRUMENT, C.TIMESTAMP])


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


def null_report(frame: pl.DataFrame) -> dict[str, float]:
    """Share of rows each feature is missing for, plus the share of invalid labels.

    What you want printed when a dataset turns out to be empty: it names the column that
    emptied it instead of leaving you to bisect the feature set.
    """
    out = {
        column: _missing_share(frame.get_column(column).is_finite().fill_null(value=False))
        for column in feature_columns(frame)
    }
    out["label_invalid"] = _missing_share(frame.get_column(LABEL_VALID))
    return out


def _missing_share(flags: pl.Series) -> float:
    """Share of ``False`` in a boolean column."""
    return 1.0 - float(flags.to_numpy().mean()) if flags.len() else 1.0


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

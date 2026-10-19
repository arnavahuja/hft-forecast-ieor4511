"""Purged, embargoed walk-forward cross-validation in session units (§5).

Overlapping labels are the problem this file exists for: at a 30 s horizon on a 5 s grid,
six consecutive rows share five of their six bars. Training on a row whose label interval
reaches into the test window leaks the answer, so those rows are purged; the remaining
overlap is handled by uniqueness weights rather than pretending the rows are independent.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl

from hft_forecast.analytics.robustness.core import walk_forward_windows
from hft_forecast.core import columns as C
from hft_forecast.core.errors import ConfigError
from hft_forecast.core.types import BoolArray, FloatArray, IntArray, TimeArray
from hft_forecast.ml.base import LABEL_END, SESSION_ID, HorizonSpec
from hft_forecast.ml.config import CVSpec

MIN_TEST_ROWS: Final = 1


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold as boolean masks over the dataset rows (nothing is copied)."""

    index: int
    train: BoolArray
    test: BoolArray
    train_sessions: tuple[int, ...]
    test_sessions: tuple[int, ...]

    @property
    def n_train(self) -> int:
        """Training rows."""
        return int(self.train.sum())

    @property
    def n_test(self) -> int:
        """Test rows."""
        return int(self.test.sum())


def split(
    frame: pl.DataFrame, spec: CVSpec, horizon: HorizonSpec, embargo_bars: int | None = None
) -> Iterator[Fold]:
    """Yield purged, embargoed walk-forward folds.

    Windows are cut on the sorted unique sessions, never on bars, so a test window always
    starts at a session open.

    Args:
        frame: The dataset; needs ``session_id``, ``timestamp`` and ``label_end_timestamp``.
        spec: Window sizes and the purge/embargo switches.
        horizon: Used for the default embargo and the bar length.
        embargo_bars: Overrides ``spec.embargo_bars``.

    Yields:
        One :class:`Fold` per window with enough training rows.

    Raises:
        ConfigError: If the dataset holds fewer sessions than one window needs.
    """
    sessions = np.unique(frame.get_column(SESSION_ID).to_numpy())
    windows = walk_forward_windows(
        len(sessions), spec.train_sessions, spec.test_sessions, spec.anchored
    )
    if not windows:
        raise ConfigError(
            f"{len(sessions)} sessions are not enough for "
            f"{spec.train_sessions} train + {spec.test_sessions} test",
            details={"sessions": len(sessions)},
        )
    session_of = frame.get_column(SESSION_ID).to_numpy()
    starts = frame.get_column(C.TIMESTAMP).to_numpy().astype("datetime64[us]")
    ends = frame.get_column(LABEL_END).to_numpy().astype("datetime64[us]")
    embargo = horizon.steps if embargo_bars is None else embargo_bars
    gap = np.timedelta64(embargo * int(horizon.delta / np.timedelta64(1, "s")), "s")
    index = 0
    for window in windows:
        test_sessions = sessions[window.train_end : window.test_end]
        train_sessions = sessions[window.train_start : window.train_end]
        test = np.isin(session_of, test_sessions)
        train = np.isin(session_of, train_sessions)
        if test.sum() < MIN_TEST_ROWS:
            continue
        if spec.purge:
            train &= ~_overlaps(starts, ends, starts[test].min(), ends[test].max())
        train &= ~_embargoed(starts, ends[test].max(), gap)
        if spec.stride:
            train &= _strided(session_of, spec.stride)
        if train.sum() < spec.min_train_rows:
            continue
        yield Fold(
            index,
            train,
            test,
            tuple(int(s) for s in train_sessions),
            tuple(int(s) for s in test_sessions),
        )
        index += 1


def _overlaps(
    starts: TimeArray, ends: TimeArray, test_start: np.datetime64, test_end: np.datetime64
) -> BoolArray:
    """Rows whose label interval ``[start, end]`` touches the test window."""
    overlap: BoolArray = (ends >= test_start) & (starts <= test_end)
    return overlap


def _embargoed(starts: TimeArray, test_end: np.datetime64, gap: np.timedelta64) -> BoolArray:
    """Rows starting inside the embargo that follows the test window."""
    banned: BoolArray = (starts > test_end) & (starts <= test_end + gap)
    return banned


def _strided(session_of: IntArray, stride: int) -> BoolArray:
    """Every ``stride``-th row of each session: a sanity run on non-overlapping labels."""
    position = np.arange(len(session_of)) - np.maximum.accumulate(
        np.where(np.diff(session_of, prepend=session_of[0] - 1) != 0, np.arange(len(session_of)), 0)
    )
    keep: BoolArray = position % stride == 0
    return keep


def label_uniqueness(starts: TimeArray, ends: TimeArray) -> FloatArray:
    """Average ``1 / concurrent labels`` over each row's label interval.

    Lopez de Prado, *Advances in Financial Machine Learning*, ch. 4. Computed with a
    difference-array sweep over the sorted bar timestamps, so it is O(n log n) rather than
    the O(n^2) pairwise form. Concurrency is ``steps`` almost everywhere, which makes the
    weights about ``1/steps`` except at the session edges.
    """
    n = len(starts)
    if n == 0:
        return np.zeros(0)
    grid = np.unique(np.concatenate([starts, ends]))
    begin = np.searchsorted(grid, starts, side="left")
    finish = np.searchsorted(grid, ends, side="right")
    counts = np.zeros(len(grid) + 1)
    np.add.at(counts, begin, 1.0)
    np.add.at(counts, finish, -1.0)
    concurrent = np.cumsum(counts)[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse = np.where(concurrent > 0, 1.0 / concurrent, 0.0)
    cumulative = np.concatenate([[0.0], np.cumsum(inverse)])
    width = np.maximum(finish - begin, 1)
    weights: FloatArray = (cumulative[finish] - cumulative[begin]) / width
    return weights


def uniqueness_weights(frame: pl.DataFrame) -> FloatArray:
    """Uniqueness weights for a dataset, computed per instrument."""
    out = np.zeros(frame.height)
    instruments = frame.get_column(C.INSTRUMENT).to_numpy()
    starts = frame.get_column(C.TIMESTAMP).to_numpy().astype("datetime64[us]")
    ends = frame.get_column(LABEL_END).to_numpy().astype("datetime64[us]")
    for instrument in np.unique(instruments):
        where = instruments == instrument
        out[where] = label_uniqueness(starts[where], ends[where])
    return out


def effective_sample_size(weights: FloatArray) -> float:
    """Sum of the uniqueness weights: about ``n / steps`` under a pure overlap of ``steps``.

    Not the Kish formula ``(sum w)^2 / sum w^2`` the design suggests: that measures how
    unequal the weights are, so perfectly overlapping labels (all weights ``1/steps``) would
    come out as ``n``, which is the opposite of the intent. Summing the average uniqueness is
    Lopez de Prado's count and gives ``n / steps``.

    An upper bound, not an estimate: microstructure noise and volatility clustering make the
    true number smaller. The block-bootstrap intervals in ``ml.evaluate`` are the honest
    uncertainty statement.
    """
    return float(weights.sum())

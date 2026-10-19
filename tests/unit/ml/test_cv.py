"""CV invariants: no leakage, disjoint ordered test windows, sane uniqueness weights."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from hft_forecast.core import columns as C
from hft_forecast.core.errors import ConfigError
from hft_forecast.ml.base import H30S, LABEL_END, SESSION_ID
from hft_forecast.ml.config import CVSpec
from hft_forecast.ml.cv import (
    _embargoed,
    effective_sample_size,
    label_uniqueness,
    split,
    uniqueness_weights,
)

STEPS = H30S.steps


def _frame(n_sessions: int = 6, bars: int = 40, steps: int = STEPS) -> pl.DataFrame:
    """One instrument, contiguous 5 s bars, labels spanning ``steps`` bars."""
    rows = []
    for session in range(n_sessions):
        base = np.datetime64("2024-07-01T13:30", "us") + np.timedelta64(session, "D")
        step = np.timedelta64(5, "s").astype("timedelta64[us]")
        stamps = base + np.arange(bars) * step
        ends = base + (np.arange(bars) + steps) * step
        rows.append(
            pl.DataFrame(
                {
                    C.TIMESTAMP: stamps,
                    C.INSTRUMENT: ["AAA"] * bars,
                    SESSION_ID: [session] * bars,
                    LABEL_END: ends,
                }
            )
        )
    return pl.concat(rows)


SPEC = CVSpec(train_sessions=2, test_sessions=1, min_train_rows=0)


def test_no_training_label_overlaps_a_test_row():
    frame = _frame()
    starts = frame[C.TIMESTAMP].to_numpy()
    ends = frame[LABEL_END].to_numpy()
    folds = list(split(frame, SPEC, H30S))
    assert folds
    for fold in folds:
        t0, t1 = starts[fold.test].min(), ends[fold.test].max()
        assert not ((ends[fold.train] >= t0) & (starts[fold.train] <= t1)).any()


def test_test_windows_are_disjoint_ordered_and_cover_each_row_once():
    folds = list(split(_frame(), SPEC, H30S))
    covered = np.zeros(len(folds[0].test), dtype=int)
    last = -1
    for fold in folds:
        covered += fold.test.astype(int)
        first = int(np.flatnonzero(fold.test)[0])
        assert first > last
        last = int(np.flatnonzero(fold.test)[-1])
    assert covered.max() == 1


def test_windows_start_at_a_session_open():
    frame = _frame()
    for fold in split(frame, SPEC, H30S):
        first = frame.filter(pl.Series(fold.test)).head(1)
        session = first[SESSION_ID][0]
        opens = frame.filter(pl.col(SESSION_ID) == session)[C.TIMESTAMP].min()
        assert first[C.TIMESTAMP][0] == opens


def test_purging_bites_when_a_label_reaches_into_the_test_window():
    # forward windows train on earlier sessions only, so the purge is a no-op unless a label
    # outlives its session; stretch the last session's labels to make one that does
    frame = _frame()
    last_train = frame[SESSION_ID].max() - 1
    stretched = frame.with_columns(
        pl.when(pl.col(SESSION_ID) == last_train)
        .then(pl.col(LABEL_END) + pl.duration(days=1))
        .otherwise(pl.col(LABEL_END))
        .alias(LABEL_END)
    )
    purged = list(split(stretched, SPEC, H30S))
    raw = list(split(stretched, SPEC.model_copy(update={"purge": False}), H30S))
    assert sum(f.n_train for f in raw) > sum(f.n_train for f in purged)


def test_the_embargo_bans_the_rows_just_after_a_test_window():
    starts = np.array(
        ["2024-07-01T15:00", "2024-07-01T15:01", "2024-07-01T16:00"], dtype="datetime64[us]"
    )
    end = np.datetime64("2024-07-01T15:00", "us")
    banned = _embargoed(starts, end, np.timedelta64(120, "s"))
    assert banned.tolist() == [False, True, False]


def test_too_few_sessions_is_a_config_error():
    with pytest.raises(ConfigError, match="not enough"):
        list(split(_frame(n_sessions=2), SPEC, H30S))


def test_stride_keeps_every_nth_row():
    folds = list(split(_frame(), SPEC.model_copy(update={"stride": STEPS}), H30S))
    plain = list(split(_frame(), SPEC, H30S))
    assert sum(f.n_train for f in folds) < sum(f.n_train for f in plain) / 2


def test_uniqueness_is_about_one_over_steps_in_the_middle():
    frame = _frame(n_sessions=1, bars=200)
    weights = uniqueness_weights(frame)
    middle = weights[20:-20]
    assert middle.mean() == pytest.approx(1 / STEPS, rel=0.25)
    assert weights[-1] > middle.mean()  # the last label overlaps nothing


def test_non_overlapping_labels_are_fully_unique():
    starts = np.array(["2024-07-01T10:00", "2024-07-01T11:00"], dtype="datetime64[us]")
    ends = np.array(["2024-07-01T10:30", "2024-07-01T11:30"], dtype="datetime64[us]")
    assert label_uniqueness(starts, ends).tolist() == [1.0, 1.0]


def test_effective_sample_size_falls_with_overlap():
    frame = _frame(n_sessions=1, bars=600)
    weights = uniqueness_weights(frame)
    n_eff = effective_sample_size(weights)
    assert n_eff < frame.height
    assert n_eff == pytest.approx(frame.height / STEPS, rel=0.3)
    assert effective_sample_size(np.ones(100)) == pytest.approx(100.0)

"""The price impact regression: it recovers a slope that was put there on purpose."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from hft_forecast.core import columns as C
from hft_forecast.core.errors import ConfigError
from hft_forecast.ml.base import SESSION_ID
from hft_forecast.ml.features.ofi import OFI
from hft_forecast.ml.impact import (
    depth_scaling,
    impact_summary,
    interval_table,
    price_impact,
)

BETA = 2e-4
"""Dollars of mid move per share of imbalance, the slope the fixture is built with."""


MEAN_REVERSION = 0.9
"""Displayed size reverts rather than wandering, so average depth means something."""


def _sizes(rng: np.random.Generator, n: int, depth: float) -> np.ndarray:
    """Displayed size on one side: mean-reverting around ``depth``."""
    shocks = rng.normal(0.0, depth / 10.0, n)
    level = np.zeros(n)
    for k in range(1, n):
        level[k] = MEAN_REVERSION * level[k - 1] + shocks[k]
    return depth + level


def simulated_book(
    n_bars: int = 2000, sessions: int = 4, depth: float = 500.0, beta: float = BETA, seed: int = 0
) -> pl.DataFrame:
    """Bars whose mid moves by ``beta`` times the order flow imbalance, plus noise.

    The quote *prices* are held fixed and only the displayed sizes move, so the imbalance is
    exactly the change in depth -- the third branch of the paper's formula -- and the frame
    has an answer known in advance. The mid is then generated from that imbalance, which is
    the relationship the regression is supposed to find.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for session in range(sessions):
        bid_size = _sizes(rng, n_bars, depth)
        ask_size = _sizes(rng, n_bars, depth)
        flow = np.diff(bid_size, prepend=bid_size[0]) - np.diff(ask_size, prepend=ask_size[0])
        noise = rng.normal(0.0, beta * depth / 100.0, n_bars)
        mid = 100.0 + np.cumsum(beta * flow + noise)
        frames.append(
            pl.DataFrame(
                {
                    C.TIMESTAMP: np.arange(n_bars) + session * n_bars,
                    C.INSTRUMENT: ["AAA"] * n_bars,
                    SESSION_ID: [session] * n_bars,
                    C.MID: mid,
                    C.BID: np.full(n_bars, 99.99),
                    C.ASK: np.full(n_bars, 100.01),
                    C.BID_SIZE: bid_size,
                    C.ASK_SIZE: ask_size,
                }
            )
        )
    return pl.concat(frames)


def test_the_regression_recovers_the_slope_it_was_given():
    fits = price_impact(simulated_book(), interval_bars=10)
    assert len(fits) == 4  # one per session
    assert np.median([f.beta for f in fits]) == pytest.approx(BETA, rel=0.1)
    assert np.median([f.r2 for f in fits]) > 0.9
    assert all(f.t_stat > 10 for f in fits)


def test_intervals_never_span_a_session_and_drop_the_opening_block():
    table = interval_table(simulated_book(n_bars=100, sessions=3), interval_bars=10)
    per_session = table.group_by(SESSION_ID).len().sort(SESSION_ID)
    assert per_session["len"].to_list() == [9, 9, 9]  # 10 blocks, the first one dropped
    assert table[OFI].null_count() == 0


def test_a_short_tail_block_is_dropped():
    """A block with fewer bars than the interval would understate the imbalance."""
    table = interval_table(simulated_book(n_bars=95, sessions=1), interval_bars=10)
    assert table.height == 8  # 9 full blocks, minus the opening one


def test_the_slope_falls_as_depth_rises():
    """The paper's second claim: beta is roughly one over the displayed depth."""
    fits = []
    for k, depth in enumerate((200.0, 800.0, 3200.0)):
        # a book twice as deep absorbs the same flow with half the price move
        fits += price_impact(
            simulated_book(sessions=2, depth=depth, beta=BETA * 500.0 / depth, seed=k),
            interval_bars=10,
        )
    assert depth_scaling(fits) == pytest.approx(-1.0, abs=0.2)


def test_sessions_with_too_few_intervals_are_skipped():
    assert price_impact(simulated_book(n_bars=40, sessions=1), interval_bars=10) == []


def test_the_summary_reports_the_two_claims():
    summary = impact_summary(price_impact(simulated_book(), interval_bars=10))
    assert summary["n_fits"] == 4
    assert summary["share_positive_beta"] == 1.0
    assert summary["share_significant"] == 1.0
    assert summary["median_beta_depth"] == pytest.approx(BETA * 500.0, rel=0.2)
    assert impact_summary([]) == {"n_fits": 0}


def test_a_frame_without_quotes_is_refused():
    frame = simulated_book(n_bars=50, sessions=1).drop(C.BID_SIZE)
    with pytest.raises(ConfigError, match="price impact"):
        interval_table(frame, interval_bars=10)

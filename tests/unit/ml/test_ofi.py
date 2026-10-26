"""Order flow imbalance: the paper's definition, the session guard and the feature set."""

from __future__ import annotations

import polars as pl
import pytest

from hft_forecast.core import columns as C
from hft_forecast.ml.base import SESSION_ID
from hft_forecast.ml.features.ofi import DEPTH, OFI, Ofi, OfiParams, add_order_flow
from hft_forecast.ml.features.registry import check_feature_lookahead, feature_columns


def book(rows: list[tuple[float, float, float, float]], sessions: list[int] | None = None):
    """A frame of best quotes: ``(bid, bid_size, ask, ask_size)`` per bar."""
    return pl.DataFrame(
        {
            C.TIMESTAMP: list(range(len(rows))),
            C.INSTRUMENT: ["AAA"] * len(rows),
            SESSION_ID: sessions or [1] * len(rows),
            C.BID: [r[0] for r in rows],
            C.BID_SIZE: [r[1] for r in rows],
            C.ASK: [r[2] for r in rows],
            C.ASK_SIZE: [r[3] for r in rows],
        }
    )


BASE = (10.00, 100.0, 10.02, 200.0)


@pytest.mark.parametrize(
    ("update", "expected", "what"),
    [
        ((10.01, 50.0, 10.02, 200.0), 50.0, "a better bid adds its whole size"),
        ((9.99, 80.0, 10.02, 200.0), -100.0, "a worse bid removes the size that was there"),
        ((10.00, 150.0, 10.02, 200.0), 50.0, "an unchanged bid contributes the depth change"),
        ((10.00, 100.0, 10.01, 40.0), -40.0, "a better ask subtracts its whole size"),
        ((10.00, 100.0, 10.03, 70.0), 200.0, "a worse ask adds back the size that was pulled"),
        ((10.00, 100.0, 10.02, 200.0), 0.0, "nothing moved"),
    ],
)
def test_one_quote_update(update, expected, what):
    out = add_order_flow(book([BASE, update]))
    assert out[OFI].to_list()[1] == pytest.approx(expected), what


def test_the_first_bar_of_a_session_has_no_imbalance():
    """The overnight gap is not an order book event."""
    out = add_order_flow(book([BASE, BASE, BASE, BASE], sessions=[1, 1, 2, 2]))
    assert out[OFI].to_list() == [None, 0.0, None, 0.0]


def test_both_quotes_moving_up_is_bullish_on_both_sides():
    """A higher bid and a higher ask are the same signal, so the two contributions add."""
    out = add_order_flow(book([BASE, (10.01, 100.0, 10.03, 100.0)]))
    bid_side = 100.0  # the better bid adds its size
    ask_side = 200.0  # the worse ask gives back the size that was pulled
    assert out[OFI].to_list()[1] == pytest.approx(bid_side + ask_side)


def test_depth_is_the_average_of_the_two_sides():
    out = add_order_flow(book([BASE]))
    assert out[DEPTH].to_list() == [150.0]


def test_the_feature_set_scales_by_depth_and_sums_over_windows():
    rows = [BASE] + [(10.00, 100.0 + 10 * k, 10.02, 200.0) for k in range(1, 11)]
    out = Ofi(OfiParams(windows=[1, 5], depth_window=5)).compute(book(rows))
    assert feature_columns(out) == ["f_ofi_1", "f_ofi_5", "f_log_depth"]
    # each bar adds 10 shares to the bid, so a 5-bar sum is 50 shares over average depth
    depth = out["f_ofi_5"].to_list()
    assert depth[-1] == pytest.approx(50.0 / ((100.0 + 10 * 8 + 200.0) / 2.0), rel=0.05)
    assert OFI not in out.columns and DEPTH not in out.columns  # raw columns are dropped


def test_the_feature_set_never_looks_ahead(quote_bars):
    check_feature_lookahead(Ofi(), quote_bars)


def test_it_needs_only_the_best_quotes(quote_bars):
    """Level 1 is the whole point: no n_quotes, no is_stale, no book depth."""
    assert set(Ofi.requires) == {C.BID, C.ASK, C.BID_SIZE, C.ASK_SIZE}
    bare = quote_bars.select([*C.KEY_COLUMNS, SESSION_ID, *Ofi.requires])
    assert Ofi().compute(bare).height == bare.height

"""The quote-dynamics feature set: each column means what its name says."""

from __future__ import annotations

import polars as pl
import pytest

from hft_forecast.core import columns as C
from hft_forecast.ml.base import SESSION_ID
from hft_forecast.ml.features.quote_dynamics import QuoteDynamics, QuoteDynamicsParams
from hft_forecast.ml.features.registry import check_feature_lookahead, feature_columns


def book(rows, sessions=None):
    """A frame of best quotes: ``(bid, bid_size, ask, ask_size)`` per bar."""
    bids = [r[0] for r in rows]
    asks = [r[2] for r in rows]
    return pl.DataFrame(
        {
            C.TIMESTAMP: list(range(len(rows))),
            C.INSTRUMENT: ["AAA"] * len(rows),
            SESSION_ID: sessions or [1] * len(rows),
            C.BID: bids,
            C.BID_SIZE: [r[1] for r in rows],
            C.ASK: asks,
            C.ASK_SIZE: [r[3] for r in rows],
            C.SPREAD_BPS: [(a - b) / ((a + b) / 2) * 1e4 for b, a in zip(bids, asks, strict=True)],
        }
    )


BASE = (10.00, 100.0, 10.02, 200.0)


def compute(rows, sessions=None, **params):
    out = QuoteDynamics(QuoteDynamicsParams(**params)).compute(book(rows, sessions))
    return out


def test_first_bar_of_a_session_is_null():
    """Nothing is carried across the overnight gap."""
    out = compute([BASE, BASE, BASE], sessions=[1, 1, 2])
    for column in feature_columns(out):
        if column in {"f_size_log_ratio", "f_log_notional"}:
            continue  # level features, defined on every bar
        values = out[column].to_list()
        assert values[0] is None, column
        assert values[2] is None, f"{column} crossed into the next session"


def test_spread_change_is_the_difference_in_bps():
    out = compute([BASE, (10.00, 100.0, 10.04, 200.0)])
    widened = out["f_spread_change_bps"].to_list()[1]
    assert widened == pytest.approx(out[C.SPREAD_BPS].to_list()[1] - out[C.SPREAD_BPS][0])
    assert widened > 0, "a wider spread is a positive change"


def test_size_log_ratio_signs_with_the_heavier_side():
    out = compute([(10.0, 300.0, 10.02, 100.0), (10.0, 100.0, 10.02, 300.0)])
    assert out["f_size_log_ratio"][0] == pytest.approx(pl.Series([3.0]).log()[0])
    assert out["f_size_log_ratio"][1] < 0


def test_imbalance_delta_is_the_tilt_not_the_level():
    """Two bars with the same lopsided book have a level but no change."""
    lopsided = (10.0, 500.0, 10.02, 100.0)
    out = compute([lopsided, lopsided])
    assert out["f_imbalance_delta"][1] == pytest.approx(0.0)
    assert out["f_size_log_ratio"][1] > 0


def test_size_change_handles_a_quote_emptying_to_zero():
    """A ratio would be undefined; log1p of each side is not."""
    out = compute([BASE, (10.00, 0.0, 10.02, 200.0)])
    assert out["f_bid_size_change"][1] == pytest.approx(-pl.Series([101.0]).log()[0])


def test_replenishment_only_counts_size_added_at_an_unmoved_price():
    rows = [
        BASE,
        (10.00, 150.0, 10.02, 200.0),  # same bid price, 50 more shares
        (10.01, 150.0, 10.02, 200.0),  # price moved: a different queue
        (10.01, 50.0, 10.02, 200.0),  # size left, not added
    ]
    out = compute(rows)
    assert out["f_bid_replenishment"].to_list()[1] == pytest.approx(0.5)
    assert out["f_bid_replenishment"].to_list()[2] == pytest.approx(0.0)
    assert out["f_bid_replenishment"].to_list()[3] == pytest.approx(0.0)


def test_ask_replenishment_is_measured_on_the_ask():
    out = compute([BASE, (10.00, 100.0, 10.02, 300.0)])
    assert out["f_ask_replenishment"][1] == pytest.approx(0.5)
    assert out["f_bid_replenishment"][1] == pytest.approx(0.0)


def test_delta_bars_widens_the_lookback():
    rows = [BASE, (10.00, 110.0, 10.02, 200.0), (10.00, 130.0, 10.02, 200.0)]
    out = compute(rows, delta_bars=2)
    assert out["f_bid_replenishment"][1] is None, "one bar back is not available yet"
    assert out["f_bid_replenishment"][2] == pytest.approx(0.3), "measured against bar 0"


def test_no_lookahead():
    rows = [
        (10.00 + 0.01 * (i % 3), 100.0 + 10 * (i % 5), 10.02 + 0.01 * (i % 2), 200.0 - 5 * (i % 4))
        for i in range(40)
    ]
    check_feature_lookahead(QuoteDynamics(), book(rows))

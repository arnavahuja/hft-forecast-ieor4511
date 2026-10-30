"""Step 7 leakage tests (synthetic data) + brute-force checks of the window features."""

import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from hft.features import build_features
from hft.labels import build_labels

DATE = "2025-03-03"
CFG = {
    "windows_s": [1, 2, 5, 10, 30, 60, 120, 300, 600],
    "horizons_s": [5, 10, 30, 60],
    "download": {"session": ["09:30:00", "16:00:00"]},
    "sample": {"start": "09:35:00", "end": "15:55:00", "close": "16:00:00"},
    "tick_size": 0.01,
}
OPEN = datetime(2025, 3, 3, 9, 30)


def synthetic_day(seed=0, day=OPEN, n_q=4000, n_t=3000, symbol="X"):
    rng = np.random.default_rng(seed)
    span_ns = int(6.5 * 3600 * 1e9)
    qt = np.sort(rng.integers(0, span_ns, n_q))
    bid = 100 + np.cumsum(rng.choice([-1, 0, 0, 1], n_q)) * 0.01
    ask = bid + rng.choice([1, 2, 3], n_q) * 0.01
    nbbo = pl.DataFrame({
        "symbol": symbol,
        "ts": [day + timedelta(microseconds=int(x // 1000)) for x in qt],
        "bid": np.round(bid, 2), "ask": np.round(ask, 2),
        "bid_size": rng.integers(1, 50, n_q) * 100, "ask_size": rng.integers(1, 50, n_q) * 100,
    }).with_columns(pl.col("ts").cast(pl.Datetime("ns")), mid=(pl.col("bid") + pl.col("ask")) / 2)
    tt = np.sort(rng.integers(0, span_ns, n_t))
    trades = pl.DataFrame({
        "symbol": symbol,
        "ts": [day + timedelta(microseconds=int(x // 1000)) for x in tt],
        "price": 100.0, "size": rng.integers(1, 500, n_t), "sign": rng.choice([-1, 1], n_t).astype(np.int8),
    }).with_columns(pl.col("ts").cast(pl.Datetime("ns")))
    return nbbo, trades


def frames_equal(a: pl.DataFrame, b: pl.DataFrame) -> bool:
    return a.equals(b, null_equal=True) or all(
        np.allclose(a[c].to_numpy().astype(float), b[c].to_numpy().astype(float), equal_nan=True, rtol=0, atol=1e-12)
        if a[c].dtype.is_numeric() else a[c].equals(b[c]) for c in a.columns)


# --- 1. shift test -------------------------------------------------------------------------
def test_features_ignore_events_after_t():
    nbbo, trades = synthetic_day()
    cut = datetime(2025, 3, 3, 12, 0, 0)
    base = build_features(nbbo, trades, ["X"], DATE, CFG)

    rng = np.random.default_rng(1)
    after_q = pl.col("ts") > cut
    nbbo2 = nbbo.with_columns(
        bid=pl.when(after_q).then(pl.col("bid") + 5).otherwise(pl.col("bid")),
        ask=pl.when(after_q).then(pl.col("ask") + 7).otherwise(pl.col("ask")),
        bid_size=pl.when(after_q).then(pl.col("bid_size") * 3).otherwise(pl.col("bid_size")),
    ).with_columns(mid=(pl.col("bid") + pl.col("ask")) / 2)
    trades2 = trades.with_columns(
        size=pl.when(after_q).then(pl.col("size") * 10).otherwise(pl.col("size")),
        sign=pl.when(after_q).then(-pl.col("sign")).otherwise(pl.col("sign")),
    )
    extra = trades2.filter(after_q).head(50).with_columns((pl.col("ts") + pl.duration(milliseconds=1)).cast(pl.Datetime("ns")))
    pert = build_features(nbbo2, pl.concat([trades2, extra]), ["X"], DATE, CFG)

    before = pl.col("t") <= cut
    assert frames_equal(base.filter(before), pert.filter(before))
    assert not frames_equal(base.filter(~before), pert.filter(~before))  # sanity: the perturbation matters


# --- brute-force check of window features ---------------------------------------------------
def test_window_features_match_brute_force():
    nbbo, trades = synthetic_day(seed=3)
    f = build_features(nbbo, trades, ["X"], DATE, CFG)
    q = nbbo.sort("ts").to_dicts()
    tr = trades.sort("ts").to_dicts()
    rng = np.random.default_rng(0)
    for t in rng.choice(f["t"].to_list(), 15, replace=False):
        row = f.filter(pl.col("t") == t).to_dicts()[0]
        mid_at = lambda x: next((r["mid"] for r in reversed(q) if r["ts"] <= x), None)
        for w in CFG["windows_s"]:
            lo = t - timedelta(seconds=w)
            win_t = [r for r in tr if lo < r["ts"] <= t]
            win_q = [r for r in q if lo < r["ts"] <= t]
            vol = sum(r["size"] for r in win_t)
            assert row[f"n_trades_{w}"] == pytest.approx(math.log1p(len(win_t)))
            assert row[f"volume_{w}"] == pytest.approx(math.log1p(vol))
            assert row[f"trade_imb_{w}"] == pytest.approx(sum(r["size"] * r["sign"] for r in win_t) / vol if vol else 0)
            assert row[f"n_quote_updates_{w}"] == pytest.approx(math.log1p(len(win_q)))
            start_mid = mid_at(lo) if lo >= q[0]["ts"] else None
            if start_mid is not None:
                assert row[f"ret_{w}"] == pytest.approx(math.log(mid_at(t) / start_mid), abs=1e-12)
            # OFI: sum of CKS contributions of updates in the window / their average depth
            idx = [i for i, r in enumerate(q) if lo < r["ts"] <= t]
            e = 0.0
            for i in idx:
                if i == 0:
                    continue
                c, p = q[i], q[i - 1]
                e += (c["bid_size"] if c["bid"] >= p["bid"] else 0) - (p["bid_size"] if c["bid"] <= p["bid"] else 0)
                e += -(c["ask_size"] if c["ask"] <= p["ask"] else 0) + (p["ask_size"] if c["ask"] >= p["ask"] else 0)
            depth = np.mean([(q[i]["bid_size"] + q[i]["ask_size"]) / 2 for i in idx]) if idx else None
            assert row[f"ofi_{w}"] == pytest.approx(e / depth if idx else 0.0)


# --- 2. label alignment ---------------------------------------------------------------------
def test_labels_on_known_mid_path():
    ts = lambda h, m, s, us=0: datetime(2025, 3, 3, h, m, s, us)
    nbbo = pl.DataFrame({
        "symbol": "X",
        "ts": [ts(9, 30, 0, 500_000), ts(10, 0, 10) + timedelta(microseconds=1), ts(10, 0, 30)],
        "bid": [9.99, 10.04, 10.09], "ask": [10.01, 10.06, 10.11],
    }).with_columns(pl.col("ts").cast(pl.Datetime("ns")), mid=(pl.col("bid") + pl.col("ask")) / 2)
    lab = build_labels(nbbo, ["X"], DATE, CFG)
    at = lambda t: lab.filter(pl.col("t") == t).to_dicts()[0]

    r = at(ts(10, 0, 0))
    assert r["r_30"] == pytest.approx(10.10 / 10.00 - 1)      # quote stamped exactly at t+30 counts
    assert r["r_30_ticks"] == pytest.approx(10) and r["dir_30"] == 1 and r["sign_30"] == 1
    assert r["r_10"] == 0 and r["dir_10"] == 0                 # 10:00:10.000001 is after t+10
    assert at(ts(10, 0, 5))["r_5"] == 0                        # same: jump lands 1 microsecond after t+5
    assert at(ts(10, 0, 6))["r_5"] == pytest.approx(10.05 / 10.00 - 1)
    assert at(ts(10, 0, 30))["r_30"] == 0                      # flat afterwards

    late = dict(CFG, sample={"start": "15:59:00", "end": "15:59:50", "close": "16:00:00"})
    lab2 = build_labels(nbbo, ["X"], DATE, late)
    row = lab2.filter(pl.col("t") == ts(15, 59, 50)).to_dicts()[0]
    assert row["r_10"] == 0 and row["r_30"] is None and row["r_60"] is None  # t+h past 16:00 -> null


def test_flat_band_direction():
    ts = lambda m, s: datetime(2025, 3, 3, 10, m, s)
    nbbo = pl.DataFrame({  # spread 4 cents; mid moves +1 cent (< half spread) then +3 cents
        "symbol": "X", "ts": [ts(0, 0), ts(0, 20), ts(1, 20)],
        "bid": [9.98, 9.99, 10.01], "ask": [10.02, 10.03, 10.05],
    }).with_columns(pl.col("ts").cast(pl.Datetime("ns")), mid=(pl.col("bid") + pl.col("ask")) / 2)
    lab = build_labels(nbbo, ["X"], DATE, CFG)
    r1 = lab.filter(pl.col("t") == ts(0, 0)).to_dicts()[0]
    r2 = lab.filter(pl.col("t") == ts(0, 50)).to_dicts()[0]
    assert r1["sign_30"] == 1 and r1["dir_30"] == 0            # +1 tick inside the flat band
    assert r2["sign_30"] == 1 and r2["dir_30"] == 1            # +2 ticks = half spread -> not flat


# --- 3. day boundary ------------------------------------------------------------------------
def test_other_days_do_not_leak():
    nbbo, trades = synthetic_day(seed=5)
    f0 = build_features(nbbo, trades, ["X"], DATE, CFG)
    l0 = build_labels(nbbo, ["X"], DATE, CFG)
    prev_q, prev_t = synthetic_day(seed=6, day=OPEN - timedelta(days=3))   # Friday before
    next_q, next_t = synthetic_day(seed=7, day=OPEN + timedelta(days=1))   # Tuesday after
    late_q = nbbo.tail(5).with_columns((pl.col("ts") + pl.duration(hours=1)).cast(pl.Datetime("ns")))  # after 16:00 same day
    nbbo_all, trades_all = pl.concat([prev_q, nbbo, next_q, late_q]), pl.concat([prev_t, trades, next_t])
    assert frames_equal(f0, build_features(nbbo_all, trades_all, ["X"], DATE, CFG))
    assert frames_equal(l0, build_labels(nbbo_all, ["X"], DATE, CFG))

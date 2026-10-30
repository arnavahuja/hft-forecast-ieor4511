from datetime import datetime, timedelta

import polars as pl

from hft.signing import sign_trades

T0 = datetime(2025, 3, 3, 10, 0, 0)


def at(ms):
    return T0 + timedelta(milliseconds=ms)


def frames(quotes, trades, symbol="X"):
    q = pl.DataFrame({"symbol": symbol, "ts": [at(m) for m, _ in quotes], "mid": [mid for _, mid in quotes]})
    t = pl.DataFrame({"symbol": symbol, "ts": [at(m) for m, _ in trades], "price": [p for _, p in trades],
                      "tr_seqnum": list(range(len(trades)))})
    cast = [pl.col("ts").cast(pl.Datetime("ns"))]
    return q.with_columns(cast), t.with_columns(cast)


def test_all_lee_ready_branches():
    quotes = [(10, 10.01), (100, 10.02)]
    trades = [
        (5, 10.00),    # before any quote, first trade -> no prior price -> 0 ("none")
        (20, 10.02),   # above mid 10.01 -> +1
        (30, 10.00),   # below mid -> -1
        (40, 10.01),   # at mid; previous different price 10.00 -> uptick -> +1 via tick test
        (50, 10.01),   # at mid; zero tick -> inherits last non-zero change (+1, from 10.00 -> 10.01)
        (100, 10.015), # quote at the SAME ts (mid 10.02) must be ignored: prevailing mid is 10.01 -> +1
        (110, 10.02),  # now quote 10.02 prevails, at mid; last change up -> +1 via tick
        (120, 10.01),  # below mid 10.02 -> -1
    ]
    q, t = frames(quotes, trades)
    out = sign_trades(t, q)
    assert out["sign"].to_list() == [0, 1, -1, 1, 1, 1, 1, -1]
    assert out["sign_method"].to_list() == ["none", "quote", "quote", "tick", "tick", "quote", "tick", "quote"]
    assert out["pre_mid"].to_list()[:2] == [None, 10.01]
    assert out["pre_mid"][5] == 10.01  # strictly-before rule


def test_at_mid_uses_tick_and_downtick():
    q, t = frames([(0, 10.00)], [(1, 10.02), (2, 10.00)])  # 10.02 above mid; 10.00 at mid after a downtick
    out = sign_trades(t, q)
    assert out["sign"].to_list() == [1, -1]
    assert out["sign_method"].to_list() == ["quote", "tick"]


def test_symbols_do_not_mix():
    qa, ta = frames([(0, 10.00)], [(1, 10.00)], "A")    # at mid, no prior trade of A -> 0
    qb, tb = frames([(0, 50.00)], [(0, 49.00)], "B")    # B trade at quote time: no prior quote of B, no prior trade -> 0
    out = sign_trades(pl.concat([ta, tb]), pl.concat([qa, qb])).sort("symbol")
    assert out["sign"].to_list() == [0, 0]
    assert out["pre_mid"].to_list() == [10.00, None]

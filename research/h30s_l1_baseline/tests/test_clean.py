from datetime import datetime

import polars as pl

from hft.clean import clean_nbbo, clean_trades

CFG = {
    "quote_conditions_keep": ["A", "B", "H", "O", "R", "W"],
    "max_spread_bps": 1000,
    "trade_corr_keep": ["00"],
    "trade_scond_allowed_chars": ["@", "F", "I", " "],
}
UNI = pl.DataFrame({"symbol": ["BRK.B", "AAPL"], "sym_root": ["BRK", "AAPL"], "sym_suffix": ["B", ""]})


def ts(us):
    return datetime(2025, 3, 3, 10, 0, 0, us)


def test_clean_nbbo_reasons_duplicates_and_timestamp():
    rows = [  # (ts_us, nano, qu_cond, bid, ask, bsz, asz)
        (1, 500, "R", 10.00, 10.02, 100, 200),  # kept
        (2, 0, "R", 10.00, 10.02, 100, 200),    # duplicate state -> dropped
        (3, 0, "R", 10.00, 10.02, 300, 200),    # size-only update -> kept
        (4, 0, "R", 10.03, 10.02, 100, 100),    # crossed
        (5, 0, "R", 10.02, 10.02, 100, 100),    # locked
        (6, 0, "R", 0.0, 10.02, 100, 100),      # zero bid
        (7, 0, "D", 10.00, 10.02, 100, 100),    # bad quote condition
        (8, 0, "R", 5.00, 15.00, 100, 100),     # too wide
        (9, 0, "O", 10.01, 10.03, 100, 100),    # opening quote, kept
    ]
    raw = pl.DataFrame({
        "ts": [ts(r[0]) for r in rows], "time_m_nano": [r[1] for r in rows], "sym_root": "BRK", "sym_suffix": "B",
        "qu_cond": [r[2] for r in rows], "best_bid": [r[3] for r in rows], "best_ask": [r[4] for r in rows],
        "best_bidsizeshares": [r[5] for r in rows], "best_asksizeshares": [r[6] for r in rows],
    }).with_columns(pl.col("ts").cast(pl.Datetime("us")), pl.col("time_m_nano").cast(pl.Int16))

    clean, counts = clean_nbbo(raw.lazy(), UNI, CFG)
    assert clean["symbol"].to_list() == ["BRK.B"] * 3
    assert clean["bid_size"].to_list() == [100, 300, 100]
    assert clean["ts"][0] == datetime(2025, 3, 3, 10, 0, 0, 1)  # microsecond part
    assert clean["ts"].dt.nanosecond()[0] == 1_500                # + 500 ns
    assert abs(clean["mid"][0] - 10.01) < 1e-12 and clean["spread_ticks"][0] == 2.0
    got = dict(counts.select("reason", "n").iter_rows())
    assert got == {"kept": 3, "duplicate_state": 1, "crossed": 1, "locked": 1, "missing_or_zero_price": 1,
                   "quote_condition": 1, "wide_spread": 1}


def test_clean_trades_conditions_and_flags():
    conds = ["@", None, "@F I", "F", "I", "@  W", "@4 I", "T", "@ ZI", "O X", "@"]
    corr = ["00"] * 10 + ["12"]
    raw = pl.DataFrame({
        "ts": [ts(i) for i in range(len(conds))], "time_m_nano": 0, "ex": ["D", "Q"] * 5 + ["N"],
        "sym_root": "AAPL", "sym_suffix": "", "tr_scond": conds, "size": 100, "price": 200.0,
        "tr_corr": corr, "tr_seqnum": list(range(len(conds))),
    }).with_columns(pl.col("ts").cast(pl.Datetime("us")), pl.col("time_m_nano").cast(pl.Int16))

    clean, counts = clean_trades(raw.lazy(), UNI, CFG)
    assert clean["tr_seqnum"].to_list() == [0, 1, 2, 3, 4]
    assert clean["is_iso"].to_list() == [False, False, True, True, False]
    assert clean["is_oddlot"].to_list() == [False, False, True, False, True]
    assert clean["is_trf"].to_list() == [True, False, True, False, True]
    sc = counts.filter(pl.col("reason") == "sale_condition")
    assert sorted(sc["detail"].to_list()) == sorted(["@  W", "@4 I", "T", "@ ZI", "O X"])
    assert counts.filter(pl.col("reason") == "corrected")["n"].item() == 1

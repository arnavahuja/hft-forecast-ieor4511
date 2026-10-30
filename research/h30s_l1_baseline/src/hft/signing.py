"""Step 4: Lee-Ready (1991) trade signing.

For each trade, take the prevailing clean NBBO strictly before the trade timestamp
(backward as-of join on ts - 1 ns, so a quote with the same timestamp is never used):
    price > mid -> +1 (buy), price < mid -> -1 (sell),
    price == mid (or no prior quote) -> tick test: sign of the change from the previous
    different trade price (zero ticks inherit the last non-zero tick); 0 if none exists.
"""

import polars as pl

AT_MID_EPS = 1e-6  # $; prices and mids are on a 0.5-cent grid, so this only absorbs float error


def tick_test(trades: pl.DataFrame) -> pl.Series:
    """Sign of the last non-zero price change per symbol (trades sorted by symbol, ts, tr_seqnum)."""
    d = trades.select(
        tick=(pl.col("price") - pl.col("price").shift(1).over("symbol")).sign()
        .replace(0, None).forward_fill().over("symbol").fill_null(0)
    )["tick"]
    return d.cast(pl.Int8)


def sign_trades(trades: pl.DataFrame, nbbo: pl.DataFrame) -> pl.DataFrame:
    """Add pre_mid (prevailing mid strictly before the trade), sign (+1/-1/0) and sign_method."""
    t = trades.sort("symbol", "ts", "tr_seqnum")
    t = t.with_columns(tick=tick_test(t), _key=pl.col("ts") - pl.duration(nanoseconds=1), _row=pl.int_range(pl.len()))
    q = nbbo.select("symbol", quote_ts="ts", pre_mid="mid").sort("quote_ts")
    j = (t.sort("_key")
         # both sides are sorted by the key globally, so sortedness holds within each symbol
         .join_asof(q, left_on="_key", right_on="quote_ts", by="symbol", strategy="backward", check_sortedness=False)
         .sort("_row"))

    diff = pl.col("price") - pl.col("pre_mid")
    at_mid = pl.col("pre_mid").is_null() | (diff.abs() < AT_MID_EPS)
    return j.with_columns(
        sign=pl.when(at_mid).then(pl.col("tick"))
        .when(diff > 0).then(pl.lit(1, pl.Int8)).otherwise(pl.lit(-1, pl.Int8)).cast(pl.Int8),
        sign_method=pl.when(~at_mid).then(pl.lit("quote"))
        .when(pl.col("tick") != 0).then(pl.lit("tick")).otherwise(pl.lit("none")),
    ).drop("_key", "_row", "tick", "quote_ts")

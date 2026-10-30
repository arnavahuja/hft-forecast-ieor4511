"""Step 6: mid-price return labels on the 1-second grid.

mid_t = last clean mid with ts <= t; mid_{t+h} = last clean mid with ts <= t+h.
r_h = mid_{t+h}/mid_t - 1, plus bps, ticks, a flat-band direction and the pure sign.
Labels whose t+h is after the close are null; quotes from other days are never used.
"""

from datetime import timedelta

import polars as pl

from hft.features import _in_session, day_bounds, make_grid


def build_labels(nbbo: pl.DataFrame, symbols, date, cfg) -> pl.DataFrame:
    horizons = sorted(cfg["horizons_s"])
    tick = cfg["tick_size"]
    open_, close = (day_bounds(date, s) for s in (cfg["download"]["session"][0], cfg["sample"]["close"]))
    start, end = day_bounds(date, cfg["sample"]["start"]), day_bounds(date, cfg["sample"]["end"])

    q = _in_session(nbbo, open_, close).select("symbol", "ts", "mid", spread=pl.col("ask") - pl.col("bid")).sort("ts")
    g = (make_grid(symbols, start, end + timedelta(seconds=max(horizons))).sort("t")
         .join_asof(q, left_on="t", right_on="ts", by="symbol", strategy="backward", check_sortedness=False)
         .sort("symbol", "t"))

    cols = [pl.col("mid").alias("mid_t"), pl.col("spread").alias("spread_t")]
    for h in horizons:
        fwd = pl.col("mid").shift(-h).over("symbol")
        ticks = ((fwd - pl.col("mid")) / tick).round(4)                 # exact zero for unchanged mids
        valid = pl.col("t") + pl.duration(seconds=h) <= pl.lit(close)
        dmid = ticks * tick
        cols += [
            pl.when(valid).then(dmid / pl.col("mid")).alias(f"r_{h}"),
            pl.when(valid).then(dmid / pl.col("mid") * 1e4).alias(f"r_{h}_bps"),
            pl.when(valid).then(ticks).alias(f"r_{h}_ticks"),
            pl.when(valid).then(
                pl.when(dmid.abs() < pl.col("spread") / 2).then(0).otherwise(ticks.sign()).cast(pl.Int8)
            ).alias(f"dir_{h}"),
            pl.when(valid).then(ticks.sign().cast(pl.Int8)).alias(f"sign_{h}"),
        ]
    return (g.select("symbol", "t", *cols)
            .filter(pl.col("t").is_between(start, end, closed="both")))

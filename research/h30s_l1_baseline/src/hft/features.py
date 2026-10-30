"""Step 5: features on a 1-second grid.

Method: for each event stream (clean NBBO updates, signed trades) build running totals per
symbol, then attach to every grid second t the state of the last event with ts <= t
(backward as-of join, equality allowed). A window feature over (t - w, t] is then
cum[t] - cum[t - w], read off the same regular grid. The grid starts `max(windows)` seconds
before the sample start, at times with no events (before the 09:30 open) the running totals
are 0, so windows near the open are truncated at 09:30 (see DECISIONS.md).

Every value at t uses only events with ts <= t; the one backward fill (mid before the first
quote, used as the start point of `ret_w`) reads the first mid of the day, which is <= t for
every sample time.
"""

from datetime import datetime, time, timedelta

import pandas as pd
import polars as pl


def day_bounds(date, hhmmss: str) -> datetime:
    return datetime.combine(pd.Timestamp(date).date(), time.fromisoformat(hhmmss))


def make_grid(symbols, start: datetime, end: datetime) -> pl.DataFrame:
    t = pl.datetime_range(start, end, interval="1s", time_unit="ns", eager=True).alias("t")
    return pl.DataFrame({"symbol": list(symbols)}).join(t.to_frame(), how="cross").sort("symbol", "t")


def _in_session(df: pl.DataFrame, open_: datetime, close: datetime) -> pl.DataFrame:
    """Events of this day's session only (guards against other days leaking in)."""
    return df.filter(pl.col("ts").is_between(open_, close, closed="both"))


def quote_stream(nbbo: pl.DataFrame) -> pl.DataFrame:
    """Per-update state + running totals: OFI (Cont, Kukanov & Stoikov 2014), depth, update count."""
    b, a, qb, qa = pl.col("bid"), pl.col("ask"), pl.col("bid_size"), pl.col("ask_size")
    pb, pa = b.shift(1).over("symbol"), a.shift(1).over("symbol")
    qb1, qa1 = qb.shift(1).over("symbol"), qa.shift(1).over("symbol")
    e = (pl.when(b >= pb).then(qb).otherwise(0) - pl.when(b <= pb).then(qb1).otherwise(0)
         - pl.when(a <= pa).then(qa).otherwise(0) + pl.when(a >= pa).then(qa1).otherwise(0))
    mid_changed = (pl.col("mid") != pl.col("mid").shift(1).over("symbol")).fill_null(True)
    return (nbbo.sort("symbol", "ts")
            .with_columns(ofi_e=e.fill_null(0).cast(pl.Float64), mid_changed=mid_changed)
            .select("symbol", "ts", "bid", "ask", "bid_size", "ask_size", "mid",
                    cum_ofi=pl.col("ofi_e").cum_sum().over("symbol"),
                    cum_depth=((qb + qa) / 2).cum_sum().over("symbol"),
                    cum_nq=pl.int_range(1, pl.len() + 1, dtype=pl.Int64).over("symbol"),
                    last_mid_change=pl.when("mid_changed").then(pl.col("ts")).forward_fill().over("symbol"))
            .sort("ts"))


def trade_stream(trades: pl.DataFrame) -> pl.DataFrame:
    """Running totals of trade count, volume, signed volume and signed count."""
    size = pl.col("size").cast(pl.Int64)
    sign = pl.col("sign").cast(pl.Int64)
    return (trades.sort("symbol", "ts")
            .select("symbol", "ts",
                    cum_nt=pl.int_range(1, pl.len() + 1, dtype=pl.Int64).over("symbol"),
                    cum_vol=size.cum_sum().over("symbol"),
                    cum_svol=(size * sign).cum_sum().over("symbol"),
                    cum_scnt=sign.cum_sum().over("symbol"),
                    last_trade=pl.col("ts"))
            .sort("ts"))


def build_features(nbbo: pl.DataFrame, trades: pl.DataFrame, symbols, date, cfg) -> pl.DataFrame:
    """Feature panel [symbol, t, features...] for grid seconds sample.start..sample.end.

    nbbo: clean NBBO (symbol, ts, bid, ask, bid_size, ask_size, mid)
    trades: signed clean trades (symbol, ts, size, sign)
    """
    windows = sorted(cfg["windows_s"])
    open_, close = (day_bounds(date, s) for s in (cfg["download"]["session"][0], cfg["sample"]["close"]))
    start, end = day_bounds(date, cfg["sample"]["start"]), day_bounds(date, cfg["sample"]["end"])
    tick = cfg["tick_size"]

    q = quote_stream(_in_session(nbbo, open_, close))
    tr = trade_stream(_in_session(trades, open_, close))
    grid = make_grid(symbols, start - timedelta(seconds=max(windows)), end).sort("t")
    asof = dict(left_on="t", right_on="ts", by="symbol", strategy="backward", check_sortedness=False)
    g = (grid.join_asof(q, **asof).drop("ts")
         .join_asof(tr, **asof).drop("ts")
         .sort("symbol", "t")
         .with_columns(pl.col("^cum_.*$").fill_null(0)))

    over = lambda e: e.over("symbol")
    mid, bid, ask, bs, as_ = pl.col("mid"), pl.col("bid"), pl.col("ask"), pl.col("bid_size"), pl.col("ask_size")
    g = g.with_columns(
        logmid=over(mid.backward_fill()).log(),
        spread_bps=(ask - bid) / mid * 1e4,
    ).with_columns(
        cum_r2=over(pl.col("logmid").diff().fill_null(0) ** 2).cum_sum().over("symbol"),
        cum_spread=pl.col("spread_bps").fill_null(0).cum_sum().over("symbol"),
        cum_spread_n=pl.col("spread_bps").is_not_null().cast(pl.Int64).cum_sum().over("symbol"),
    )

    secs = lambda e: e.dt.total_nanoseconds() / 1e9
    open_ns = pl.lit(open_).cast(pl.Datetime("ns"))
    feats = [
        pl.col("spread_bps"),
        ((ask - bid) / tick).round(4).alias("spread_ticks"),
        ((bs - as_) / (bs + as_)).alias("quote_imbalance"),
        (((ask * bs + bid * as_) / (bs + as_) - mid) / mid).alias("microprice_dev"),
        secs(pl.col("t") - pl.col("last_trade").fill_null(open_ns)).alias("time_since_last_trade"),
        secs(pl.col("t") - pl.col("last_mid_change").fill_null(open_ns)).alias("time_since_last_mid_change"),
        secs(pl.col("t") - open_ns).alias("secs_since_open"),
    ]
    for w in windows:
        d = lambda c: over(pl.col(c) - pl.col(c).shift(w))
        n_q, n_t, vol = d("cum_nq"), d("cum_nt"), d("cum_vol")
        feats += [
            d("logmid").alias(f"ret_{w}"),
            pl.when(n_q > 0).then(d("cum_ofi") / (d("cum_depth") / n_q)).otherwise(0.0).alias(f"ofi_{w}"),
            pl.when(vol > 0).then(d("cum_svol") / vol).otherwise(0.0).alias(f"trade_imb_{w}"),
            pl.when(n_t > 0).then(d("cum_scnt") / n_t).otherwise(0.0).alias(f"trade_cnt_imb_{w}"),
            n_t.cast(pl.Float64).log1p().alias(f"n_trades_{w}"),
            vol.cast(pl.Float64).log1p().alias(f"volume_{w}"),
            n_q.cast(pl.Float64).log1p().alias(f"n_quote_updates_{w}"),
            (d("cum_spread") / d("cum_spread_n")).alias(f"avg_spread_{w}"),
        ]
        if w >= 10:
            feats.append(d("cum_r2").alias(f"rv_{w}"))

    return (g.select("symbol", "t", *feats)
            .filter(pl.col("t").is_between(start, end, closed="both")))

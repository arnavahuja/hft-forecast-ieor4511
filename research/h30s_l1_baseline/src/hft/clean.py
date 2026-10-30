"""Step 3: clean raw TAQ L1 quotes (complete NBBO) and trades (ctm).

Both functions take the raw LazyFrame and the day's universe, and return
(clean DataFrame, drop counts DataFrame[symbol, reason, detail, n]).
Timestamps become one nanosecond column `ts` = SIP time (date + time_m) + time_m_nano.
Filter choices are documented in reports/DECISIONS.md.
"""

import re

import polars as pl

KEY = ["sym_root", "sym_suffix"]


def _attach_symbol(raw: pl.LazyFrame, universe: pl.DataFrame) -> pl.LazyFrame:
    syms = universe.select("symbol", *KEY).lazy()
    return raw.join(syms, on=KEY, how="inner").with_columns(
        ts=pl.col("ts").cast(pl.Datetime("ns")) + pl.duration(nanoseconds=pl.col("time_m_nano").fill_null(0))
    )


def _counts(df: pl.DataFrame, detail: str | None = None) -> pl.DataFrame:
    by = ["symbol", "reason"] + ([detail] if detail else [])
    out = df.filter(pl.col("reason").is_not_null()).group_by(by).len("n")
    if detail:
        out = out.rename({detail: "detail"})
    else:
        out = out.with_columns(detail=pl.lit(None, pl.String))
    return out.select("symbol", "reason", "detail", pl.col("n").cast(pl.Int64))


def clean_nbbo(raw: pl.LazyFrame, universe: pl.DataFrame, cfg_clean: dict, tick: float = 0.01):
    """Drop bad/crossed/locked/too-wide NBBO rows and repeated identical states.

    Kept columns: symbol, ts, bid, ask, bid_size, ask_size, mid, spread, spread_ticks.
    Size-only updates (same prices, new sizes) are kept: they matter for imbalance / OFI.
    """
    bid, ask = pl.col("best_bid"), pl.col("best_ask")
    reason = (
        pl.when(~pl.col("qu_cond").is_in(cfg_clean["quote_conditions_keep"])).then(pl.lit("quote_condition"))
        .when(bid.is_null() | ask.is_null() | (bid <= 0) | (ask <= 0)).then(pl.lit("missing_or_zero_price"))
        .when(bid > ask).then(pl.lit("crossed"))
        .when(bid == ask).then(pl.lit("locked"))
        .when((ask - bid) / ((ask + bid) / 2) * 1e4 > cfg_clean["max_spread_bps"]).then(pl.lit("wide_spread"))
    )
    df = (_attach_symbol(raw, universe)
          .with_columns(reason=reason)
          .sort("symbol", "ts")
          .collect())
    counts = _counts(df)

    state = ["best_bid", "best_ask", "best_bidsizeshares", "best_asksizeshares"]
    kept = df.filter(pl.col("reason").is_null()).with_columns(
        dup=pl.all_horizontal([pl.col(c) == pl.col(c).shift(1).over("symbol") for c in state]).fill_null(False)
    )
    dup_counts = kept.filter("dup").group_by("symbol").len("n").select(
        "symbol", reason=pl.lit("duplicate_state"), detail=pl.lit(None, pl.String), n=pl.col("n").cast(pl.Int64))

    clean = kept.filter(~pl.col("dup")).select(
        "symbol", "ts",
        bid=bid, ask=ask,
        bid_size=pl.col("best_bidsizeshares"), ask_size=pl.col("best_asksizeshares"),
        mid=(ask + bid) / 2,
        spread=ask - bid,
        spread_ticks=((ask - bid) / tick).round(4),
    )
    kept_counts = clean.group_by("symbol").len("n").select(
        "symbol", reason=pl.lit("kept"), detail=pl.lit(None, pl.String), n=pl.col("n").cast(pl.Int64))
    return clean, pl.concat([kept_counts, counts, dup_counts]).sort("symbol", "reason")


def clean_trades(raw: pl.LazyFrame, universe: pl.DataFrame, cfg_clean: dict):
    """Keep uncorrected regular-way trades (incl. ISO and odd lots); flag ISO, odd lot and off-exchange (TRF).

    Kept columns: symbol, ts, price, size, ex, tr_seqnum, is_iso, is_oddlot, is_trf.
    """
    allowed = "".join(re.escape(c) for c in cfg_clean["trade_scond_allowed_chars"])
    cond = pl.col("tr_scond")
    reason = (
        pl.when(~pl.col("tr_corr").is_in(cfg_clean["trade_corr_keep"])).then(pl.lit("corrected"))
        .when(pl.col("price").is_null() | pl.col("size").is_null() | (pl.col("price") <= 0) | (pl.col("size") <= 0))
        .then(pl.lit("missing_or_zero_price_size"))
        .when(cond.is_not_null() & (cond.str.replace_all(f"[{allowed}]", "") != "")).then(pl.lit("sale_condition"))
    )
    df = (_attach_symbol(raw, universe)
          .with_columns(reason=reason)
          .sort("symbol", "ts", "tr_seqnum")
          .collect())
    counts = _counts(df.with_columns(detail=pl.when(pl.col("reason") == "sale_condition").then(cond)), "detail")

    clean = df.filter(pl.col("reason").is_null()).select(
        "symbol", "ts", "price", "size", "ex", "tr_seqnum",
        is_iso=cond.fill_null("").str.contains("F", literal=True),
        is_oddlot=cond.fill_null("").str.contains("I", literal=True),
        is_trf=pl.col("ex") == "D",
    )
    kept_counts = clean.group_by("symbol").len("n").select(
        "symbol", reason=pl.lit("kept"), detail=pl.lit(None, pl.String), n=pl.col("n").cast(pl.Int64))
    return clean, pl.concat([kept_counts, counts]).sort("symbol", "reason")

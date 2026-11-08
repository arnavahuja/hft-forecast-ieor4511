"""Where is the 5-minute signal? Cross-sectional IC by stock group and by time of day.

Not a trial (no fitting). For one run and its primary score (y_hat; the direction head is shown
too), on the tradeable target r = log mid(t+6) - log mid(t+1):

  tick regime   stocks whose median quoted spread over the test period is <= 1.5 ticks
                ("one-tick", large-tick names) vs the rest ("wide")
  price         three equal groups of stocks by median mid price
  time of day   30-minute windows of the New York session (09:30-10:00 ... 15:30-16:00)

Within a stock group the IC is computed across the stocks of that group at each minute, then
averaged per day; t over days. Groups are fixed from the stocks' own medians (no outcome used).

usage: hetero_h5m.py <label>=<config>:<run_id>:<model>     (same format as eval_h5m.py)
Writes <HFTF_DATA_DIR>/reports/hetero_<run_id>.md
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_h5m import load_trial, mean_t  # noqa: E402

TICK = 0.01
SCORES = (("s_ret", "y_hat"), ("s_prob", "p_up-p_down"))


def ic_by_day(d: pl.DataFrame, score: str) -> pl.Series:
    per_min = (
        d.group_by("timestamp")
        .agg(pl.corr(score, "r", method="spearman").alias("ic"), pl.col("session_id").first())
        .filter(pl.col("ic").is_finite())
    )
    return per_min.group_by("session_id").agg(pl.col("ic").mean())["ic"]


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    root = Path(os.environ["HFTF_DATA_DIR"])
    label, spec = sys.argv[1].split("=", 1)
    cfg_file, run_id, model = spec.split(":")
    df = load_trial(root, cfg_file, run_id, model).filter(pl.col("r").is_finite())

    stocks = df.group_by("instrument_id").agg(
        (pl.col("spr") * pl.col("mid") / 1e4 / TICK).median().alias("spread_ticks"),
        pl.col("mid").median().alias("price"),
    )
    stocks = stocks.with_columns(
        pl.when(pl.col("spread_ticks") <= 1.5).then(pl.lit("one-tick")).otherwise(pl.lit("wide"))
        .alias("tick_regime"),
        pl.col("price").qcut(3, labels=["low price", "mid price", "high price"]).cast(pl.Utf8)
        .alias("price_group"),
    )
    df = df.join(stocks.select("instrument_id", "tick_regime", "price_group"), on="instrument_id")
    et = pl.col("timestamp").dt.convert_time_zone("America/New_York")
    minute = et.dt.hour().cast(pl.Int32) * 60 + et.dt.minute().cast(pl.Int32)
    df = df.with_columns(((minute - 1 - 570) // 30).alias("tod"))  # bar end -> bar start

    rows = []
    for dim in ("tick_regime", "price_group"):
        for group in sorted(df[dim].unique().to_list()):
            sub = df.filter(pl.col(dim) == group)
            n_stocks = sub["instrument_id"].n_unique()
            for score, head in SCORES:
                m, t, days = mean_t(ic_by_day(sub, score))
                rows.append({"dimension": dim, "group": group, "stocks": n_stocks, "head": head,
                             "ic_cs": m, "t": t, "days": days})
    per_min = (
        df.group_by("timestamp")
        .agg(*[pl.corr(s, "r", method="spearman").alias(s) for s, _ in SCORES],
             pl.col("session_id").first(), pl.col("tod").first())
    )
    for tod in sorted(per_min["tod"].drop_nulls().unique().to_list()):
        sub = per_min.filter(pl.col("tod") == tod)
        start = 570 + 30 * tod
        name = f"{start // 60:02d}:{start % 60:02d}-{(start + 30) // 60:02d}:{(start + 30) % 60:02d}"
        for score, head in SCORES:
            daily = sub.filter(pl.col(score).is_finite()).group_by("session_id").agg(pl.col(score).mean())
            m, t, days = mean_t(daily[score])
            rows.append({"dimension": "time_of_day", "group": name, "stocks": 47, "head": head,
                         "ic_cs": m, "t": t, "days": days})
    out = pl.DataFrame(rows)

    def f(x: float) -> str:
        return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"

    lines = [f"# Where is the signal? {label} (run {run_id}, {model})", "",
             "Cross-sectional IC on log mid(t+6) - log mid(t+1); t over days.", "",
             "| dimension | group | stocks | head | ic_cs | t | days |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in out.iter_rows(named=True):
        lines.append(f"| {r['dimension']} | {r['group']} | {r['stocks']} | {r['head']} | "
                     f"{f(r['ic_cs'])} | {f(r['t'])} | {r['days']} |")
    lines += ["", "Stock groups:", ""]
    for r in stocks.sort("spread_ticks").iter_rows(named=True):
        lines.append(f"- {r['instrument_id']}: median spread {r['spread_ticks']:.2f} ticks, "
                     f"median price {r['price']:.2f}, {r['tick_regime']}, {r['price_group']}")
    dest = root / "reports" / f"hetero_{run_id}.md"
    dest.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:60]))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()

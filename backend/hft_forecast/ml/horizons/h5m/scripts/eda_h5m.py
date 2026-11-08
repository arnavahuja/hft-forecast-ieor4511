"""Exploratory analysis for the 5-minute horizon (no model fitting, so it is not a trial).

First build the dataset once (repo root, HFTF_DATA_DIR set to the pulled data):

    hftf ml build-dataset backend/hft_forecast/ml/horizons/h5m/config_taq.yaml

Then:

    python backend/hft_forecast/ml/horizons/h5m/scripts/eda_h5m.py \
        backend/hft_forecast/ml/horizons/h5m/config_taq.yaml

Outputs go to <HFTF_DATA_DIR>/reports/eda_h5m/:
    ic_by_feature.csv     IC of every f_* feature with the 1/2/5/10/30-minute mid return,
                          and with the 5-minute return that starts one bar later (tradeable)
    class_balance.csv     share of down / flat / up for dead zones of 1, 2, 3, 5, 10 bps
    by_stock.csv          price, spread, 5-minute volatility, break-even IC, key ICs per stock
    by_time_of_day.csv    volatility and imbalance IC per 30-minute bucket
    summary.md            the headline numbers in one page

IC = Spearman rank correlation, computed inside each (stock, session) and then averaged.
The t-statistic treats stock-sessions as independent, which overstates it because stocks
move together on the same day; read it as a ranking, not a p-value.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import polars as pl

from hft_forecast.ml.config import ExperimentConfig
from hft_forecast.ml.datasets import load_dataset, usable_rows

GROUP = ["instrument_id", "session_id"]
BPS = 1e4
TICK = 0.01


def ic_table(frame: pl.DataFrame, features: list[str], target: str, by: list[str]) -> pl.DataFrame:
    """Mean Spearman IC per feature across `by` groups, with t-stat and share positive."""
    sub = frame.filter(pl.col(target).is_not_null() & pl.col(target).is_finite())
    per_group = sub.group_by(by).agg(
        [pl.corr(f, target, method="spearman").alias(f) for f in features]
    )
    long = per_group.unpivot(index=by, on=features, variable_name="feature", value_name="ic")
    long = long.filter(pl.col("ic").is_not_null() & pl.col("ic").is_finite())
    out = long.group_by("feature").agg(
        pl.col("ic").mean().alias("ic_mean"),
        pl.col("ic").std().alias("ic_std"),
        pl.len().alias("n_groups"),
        (pl.col("ic") > 0).mean().alias("share_positive"),
    )
    return out.with_columns(
        (pl.col("ic_mean") / (pl.col("ic_std") / pl.col("n_groups").sqrt())).alias("t_stat"),
        pl.lit(target).alias("target"),
    )


def md_table(frame: pl.DataFrame, digits: int = 4) -> str:
    """A small markdown table without extra dependencies."""
    cols = frame.columns
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in frame.iter_rows():
        cells = [f"{v:.{digits}f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: eda_h5m.py <config.yaml>")
    config = ExperimentConfig.load(Path(sys.argv[1]))
    root = Path(os.environ.get("HFTF_DATA_DIR", "data")).resolve()
    directory = root / "features" / config.dataset_hash()
    if not directory.exists():
        sys.exit(f"{directory} not found: run `hftf ml build-dataset {sys.argv[1]}` first")
    out = root / "reports" / "eda_h5m"
    out.mkdir(parents=True, exist_ok=True)

    frame = load_dataset(directory)
    log_mid = pl.col("mid").log()
    targets = {f"r_{k}m": log_mid.shift(-k).over(GROUP) - log_mid for k in (1, 2, 5, 10, 30)}
    targets["r_5m_from_next_bar"] = log_mid.shift(-6).over(GROUP) - log_mid.shift(-1).over(GROUP)
    frame = frame.with_columns([expr.alias(name) for name, expr in targets.items()])
    rows = usable_rows(frame)
    features = [c for c in rows.columns if c.startswith("f_")]
    print(f"{rows.height:,} usable rows, {rows['instrument_id'].n_unique()} stocks, "
          f"{rows['session_id'].n_unique()} sessions, {len(features)} features")

    # 1. IC by feature and horizon
    ic = pl.concat([ic_table(rows, features, t, GROUP) for t in targets])
    ic = ic.sort(["target", "ic_mean"], descending=[False, True])
    ic.write_csv(out / "ic_by_feature.csv")

    # 2. class balance under different dead zones
    fwd_bps = (pl.col("fwd_logret") * BPS)
    balance = pl.DataFrame([
        rows.select(
            pl.lit(tau).alias("dead_zone_bps"),
            (fwd_bps < -tau).mean().alias("down"),
            (fwd_bps.abs() <= tau).mean().alias("flat"),
            (fwd_bps > tau).mean().alias("up"),
        ).row(0, named=True)
        for tau in (1.0, 2.0, 3.0, 5.0, 10.0)
    ])
    balance.write_csv(out / "class_balance.csv")

    # 3. per stock: costs vs volatility, and the two simplest signals
    stock = rows.group_by("instrument_id").agg(
        pl.col("mid").median().alias("median_price"),
        pl.col("spread_bps").median().alias("median_spread_bps"),
        ((pl.col("ask") - pl.col("bid")) / TICK).median().alias("median_spread_ticks"),
        (pl.col("fwd_logret") * BPS).std().alias("sd_5m_bps"),
        (pl.col("is_stale").cast(pl.Float64)).mean().alias("stale_share"),
        pl.corr("f_imbalance", "fwd_logret", method="spearman").alias("ic_imbalance_pooled"),
        pl.corr("f_logret_5", "fwd_logret", method="spearman").alias("ic_ret5_pooled"),
    )
    # trade every bar, pay the full quoted spread per round trip (Gaussian approximation)
    stock = stock.with_columns(
        (pl.col("median_spread_bps") / (pl.col("sd_5m_bps") * math.sqrt(2 / math.pi)))
        .alias("breakeven_ic_full_spread")
    ).sort("median_spread_ticks", descending=True)
    stock.write_csv(out / "by_stock.csv")

    # 4. time of day (30-minute buckets after the open)
    timed = rows.with_columns((pl.col("seconds_since_open") // 1800).cast(pl.Int32).alias("bucket_30m"))
    per_bucket = timed.group_by("bucket_30m").agg(
        (pl.col("fwd_logret").abs() * BPS).mean().alias("mean_abs_5m_bps"),
        pl.col("spread_bps").mean().alias("mean_spread_bps"),
        pl.len().alias("rows"),
    )
    by_ic = timed.filter(pl.col("fwd_logret").is_not_null()).group_by(GROUP + ["bucket_30m"]).agg(
        pl.corr("f_imbalance", "fwd_logret", method="spearman").alias("ic")
    ).filter(pl.col("ic").is_finite()).group_by("bucket_30m").agg(
        pl.col("ic").mean().alias("ic_imbalance"))
    tod = per_bucket.join(by_ic, on="bucket_30m", how="left").sort("bucket_30m")
    tod = tod.with_columns((pl.col("bucket_30m") * 30).alias("minutes_after_open"))
    tod.write_csv(out / "by_time_of_day.csv")

    # 5. one-page summary
    top5 = ic.filter(pl.col("target") == "r_5m").head(8)
    lagged = ic.filter(pl.col("target") == "r_5m_from_next_bar").head(8)
    md = [
        "# h5m EDA summary", "",
        f"- Usable rows: {rows.height:,}; stocks: {rows['instrument_id'].n_unique()}; "
        f"sessions: {rows['session_id'].n_unique()}",
        f"- 5-minute return s.d. (median across stocks): {stock['sd_5m_bps'].median():.1f} bps; "
        f"median quoted spread: {stock['median_spread_bps'].median():.2f} bps",
        f"- Median break-even IC if every bar is traded at the full spread: "
        f"{stock['breakeven_ic_full_spread'].median():.3f}",
        "", "## Class balance", "", md_table(balance, 3),
        "", "## Top features, IC with the 5-minute return", "",
        md_table(top5.select("feature", "ic_mean", "t_stat", "share_positive")),
        "", "## Same, but the return starts one bar later (what a trade can capture)", "",
        md_table(lagged.select("feature", "ic_mean", "t_stat", "share_positive")),
        "", "## IC decay of f_imbalance across horizons", "",
        md_table(ic.filter(pl.col("feature") == "f_imbalance").select("target", "ic_mean", "t_stat")),
        "", "## Time of day", "", md_table(tod.select("minutes_after_open", "mean_abs_5m_bps", "mean_spread_bps", "ic_imbalance"), 3),
        "", f"Per-stock table: {out / 'by_stock.csv'}",
    ]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

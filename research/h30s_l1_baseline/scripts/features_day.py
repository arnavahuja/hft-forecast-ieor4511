"""Steps 5-6: build the 1-second feature and label panels for one day.

    python scripts/features_day.py --date 2025-12-10
    python scripts/features_day.py --date 2025-12-10 --symbol AAPL      # one symbol, not saved

Reads  data/clean/nbbo/..., data/clean/trades_signed/..., data/universe/...
Writes data/features/date=YYYY-MM-DD/part.parquet  [date, t, symbol, role, features...]
       data/labels/date=YYYY-MM-DD/part.parquet    [date, t, symbol, mid_t, spread_t, r_h, ...]
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import polars as pl

from hft.config import load_config
from hft.features import build_features
from hft.labels import build_labels
from hft.universe import load_universe


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True)
    ap.add_argument("--symbol", action="append")
    ap.add_argument("--config", default="config/pilot.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = pd.Timestamp(args.date)
    part = f"date={d:%Y-%m-%d}"
    data = Path(cfg["data_dir"])
    uni = load_universe(data, d).select("symbol", "role")
    if args.symbol:
        uni = uni.filter(pl.col("symbol").is_in(args.symbol))
    symbols = uni["symbol"].to_list()

    t0 = time.time()
    sym = pl.col("symbol").is_in(symbols)
    nbbo = pl.scan_parquet(data / "clean" / "nbbo" / part / "part.parquet").filter(sym).collect()
    trades = pl.scan_parquet(data / "clean" / "trades_signed" / part / "part.parquet").filter(sym).collect()

    meta = [pl.lit(d.date()).alias("date"), pl.col("t"), pl.col("symbol")]
    feats = build_features(nbbo, trades, symbols, d, cfg).join(uni, on="symbol")
    feats = feats.select(*meta, "role", pl.exclude("t", "symbol", "role"))
    labels = build_labels(nbbo, symbols, d, cfg).select(*meta, pl.exclude("t", "symbol"))

    for name, df in [("features", feats), ("labels", labels)]:
        if not args.symbol:
            out = data / name / part / "part.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(out, compression="zstd")
        print(f"  {d:%Y-%m-%d} {name:8s} {df.height:>9,} rows x {df.width} cols")
    print(f"  done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

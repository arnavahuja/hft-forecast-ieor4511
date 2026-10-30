"""Step 3: clean one day of raw NBBO and trades for the day's universe (targets + benchmarks).

    python scripts/clean_day.py --date 2025-12-10
    python scripts/clean_day.py --date 2025-12-10 --symbol AAPL     # one symbol (debugging)

Reads  data/raw/{nbbo,trades}/date=YYYY-MM-DD/*.parquet and data/universe/date=.../universe.parquet
Writes data/clean/{nbbo,trades}/date=YYYY-MM-DD/part.parquet and data/clean/drops/date=.../drops.parquet
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import polars as pl

from hft.clean import clean_nbbo, clean_trades
from hft.config import load_config
from hft.universe import load_universe


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True)
    ap.add_argument("--symbol", action="append", help="restrict to these symbols (debugging); output is not saved")
    ap.add_argument("--config", default="config/pilot.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = pd.Timestamp(args.date)
    part = f"date={d:%Y-%m-%d}"
    data = Path(cfg["data_dir"])
    uni = load_universe(data, d)
    if args.symbol:
        uni = uni.filter(pl.col("symbol").is_in(args.symbol))

    drops = []
    for kind, fn, extra in [("nbbo", clean_nbbo, {"tick": cfg["tick_size"]}), ("trades", clean_trades, {})]:
        t0 = time.time()
        raw = pl.scan_parquet(data / "raw" / kind / part / "*.parquet")
        clean, counts = fn(raw, uni, cfg["clean"], **extra)
        drops.append(counts.with_columns(dataset=pl.lit(kind), date=pl.lit(d.date())))
        if not args.symbol:
            out = data / "clean" / kind / part / "part.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            clean.write_parquet(out, compression="zstd")
        print(f"  {d:%Y-%m-%d} {kind:6s} kept {len(clean):>11,} rows  ({time.time() - t0:.0f}s)")

    drops = pl.concat(drops)
    if not args.symbol:
        out = data / "clean" / "drops" / part / "drops.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        drops.write_parquet(out)
    summary = drops.group_by("dataset", "reason").agg(pl.col("n").sum()).sort("dataset", "n", descending=[False, True])
    with pl.Config(tbl_rows=50):
        print(summary)


if __name__ == "__main__":
    main()

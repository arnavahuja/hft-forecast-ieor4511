"""Step 4: Lee-Ready sign one day of clean trades.

    python scripts/sign_day.py --date 2025-12-10

Reads  data/clean/{nbbo,trades}/date=YYYY-MM-DD/part.parquet
Writes data/clean/trades_signed/date=YYYY-MM-DD/part.parquet (clean trades + pre_mid, sign, sign_method)
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import polars as pl

from hft.config import load_config
from hft.signing import sign_trades


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True)
    ap.add_argument("--config", default="config/pilot.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    part = f"date={pd.Timestamp(args.date):%Y-%m-%d}"
    clean = Path(cfg["data_dir"]) / "clean"
    t0 = time.time()
    signed = sign_trades(pl.read_parquet(clean / "trades" / part / "part.parquet"),
                         pl.read_parquet(clean / "nbbo" / part / "part.parquet", columns=["symbol", "ts", "mid"]))
    out = clean / "trades_signed" / part / "part.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    signed.write_parquet(out, compression="zstd")
    print(f"  {args.date}: signed {len(signed):,} trades ({time.time() - t0:.0f}s) -> {out}")

    method = signed.group_by("sign_method").len().with_columns(share=pl.col("len") / pl.col("len").sum())
    by_venue = signed.group_by("is_trf").agg(
        buy=(pl.col("sign") == 1).mean(), sell=(pl.col("sign") == -1).mean(), unsigned=(pl.col("sign") == 0).mean(),
        at_mid_tick=(pl.col("sign_method") != "quote").mean())
    print(method.sort("len", descending=True))
    print(by_venue.sort("is_trf"))


if __name__ == "__main__":
    main()

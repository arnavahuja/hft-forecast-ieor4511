"""Download raw TAQ L1 quotes and trades for one day (or the whole pilot range).

Examples (run from this folder with the project environment):
    python scripts/download_day.py --date 2025-03-03 --group pilot --group benchmarks
    python scripts/download_day.py --pilot-range --group pilot --group benchmarks

Groups:
    sp500       all S&P 500 members that day (large: ~550 MB/day; ask before running many days)
    pilot       configured pilot tickers that are S&P 500 members that day
    benchmarks  configured benchmark tickers (SPY)

Each run also (re)builds that day's universe files (Step 2) in data/universe/date=YYYY-MM-DD/.
"""

import argparse

import pandas as pd
from wrds_login import connect

from hft.config import candidate_dates, load_config
from hft.universe import build_universe, universe_dir
from hft.wrds_io import download_day, is_trading_day


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", action="append", default=[], help="YYYY-MM-DD (repeatable)")
    ap.add_argument("--pilot-range", action="store_true", help="all trading days in the config's pilot range")
    ap.add_argument("--group", action="append", choices=["sp500", "pilot", "benchmarks"], required=True)
    ap.add_argument("--config", default="config/pilot.yaml")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dates = [pd.Timestamp(x) for x in args.date] + (candidate_dates(cfg) if args.pilot_range else [])
    if not dates:
        ap.error("give --date or --pilot-range")

    db = connect()
    summaries = []
    for d in dates:
        if not is_trading_day(db, d):
            print(f"  {d:%Y-%m-%d} no TAQ tables (holiday?) - skipped")
            continue
        uni = build_universe(db, cfg, d)
        groups = {
            "sp500": lambda: pd.read_parquet(universe_dir(cfg["data_dir"], d) / "sp500_members.parquet"),
            "pilot": lambda: uni[uni["role"] == "target"],
            "benchmarks": lambda: uni[uni["role"] == "benchmark"],
        }
        for group in args.group:
            summaries.append(download_day(db, d, groups[group](), name=group, data_dir=cfg["data_dir"],
                                          datasets=cfg["download"]["datasets"],
                                          session=tuple(cfg["download"]["session"]), overwrite=args.overwrite))
    if summaries:
        print(pd.concat(summaries).to_string(index=False))


if __name__ == "__main__":
    main()

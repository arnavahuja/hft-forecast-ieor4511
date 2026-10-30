"""Step 2: build the point-in-time universe for one or more days.

    python scripts/build_universe.py --date 2025-12-10
    python scripts/build_universe.py --pilot-range

Writes data/universe/date=YYYY-MM-DD/{sp500_members,universe}.parquet.
"""

import argparse

import pandas as pd
from wrds_login import connect

from hft.config import candidate_dates, load_config
from hft.universe import build_universe
from hft.wrds_io import is_trading_day


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", action="append", default=[], help="YYYY-MM-DD (repeatable)")
    ap.add_argument("--pilot-range", action="store_true")
    ap.add_argument("--config", default="config/pilot.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dates = [pd.Timestamp(x) for x in args.date] + (candidate_dates(cfg) if args.pilot_range else [])
    if not dates:
        ap.error("give --date or --pilot-range")
    db = connect()
    for d in dates:
        if is_trading_day(db, d):
            build_universe(db, cfg, d)
        else:
            print(f"  {d:%Y-%m-%d}: no TAQ tables (holiday?) - skipped")


if __name__ == "__main__":
    main()

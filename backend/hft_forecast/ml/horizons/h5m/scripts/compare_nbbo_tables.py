"""One-day check: does `nbbom` (used by the shared pull) give the same NBBO as WRDS `complete_nbbo`?

Background. The shared `wrds_taq` source builds bars from `taqm_<year>.nbbom_<date>` (the
NBBO messages). The h30s group uses `complete_nbbo_<date>`, which WRDS builds by also adding the
exchange quotes that were themselves the NBBO (Holden & Jacobsen 2014 show the NBBO file alone
can miss those). If the two differ a lot, our 1-minute mids are stale and the whole team should
know. This script needs a WRDS login (the same pgpass the pull uses) and takes a few minutes.

    python backend/hft_forecast/ml/horizons/h5m/scripts/compare_nbbo_tables.py --date 2024-03-01 --symbol AAPL

Prints, for one stock-day: quote rows in each table, and on the 1-minute grid the share of
minutes where the last mid differs, the median absolute difference in bps, and the correlation
of the 5-minute mid returns.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import wrds

NBBOM = """
select time_m, best_bid as bid, best_ask as ask
from taqm_{y}.nbbom_{d}
where sym_root = %(s)s and sym_suffix is null
  and time_m >= '09:30:00' and time_m < '16:00:00'
  and best_bid > 0 and best_ask > best_bid
"""
COMPLETE = """
select time_m, best_bid as bid, best_ask as ask
from taqm_{y}.complete_nbbo_{d}
where sym_root = %(s)s and sym_suffix is null
  and time_m >= '09:30:00' and time_m < '16:00:00'
  and best_bid > 0 and best_ask > best_bid
"""


def minute_mids(frame: pd.DataFrame, day: str) -> pd.Series:
    stamp = pd.to_datetime(day + " " + frame["time_m"].astype(str))
    mid = pd.Series(((frame["bid"].astype(float) + frame["ask"].astype(float)) / 2).values, index=stamp)
    mid = mid.sort_index()
    # bar labelled at its end: the last mid in (t-1min, t]
    return mid.resample("1min", label="right", closed="right").last().ffill()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2024-03-01")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--username", default=None)
    args = parser.parse_args()
    y, d = args.date[:4], args.date.replace("-", "")
    db = wrds.Connection(wrds_username=args.username) if args.username else wrds.Connection()
    a = db.raw_sql(NBBOM.format(y=y, d=d), params={"s": args.symbol})
    b = db.raw_sql(COMPLETE.format(y=y, d=d), params={"s": args.symbol})
    db.close()
    print(f"{args.symbol} {args.date}: nbbom rows {len(a):,}; complete_nbbo rows {len(b):,}")
    ma, mb = minute_mids(a, args.date), minute_mids(b, args.date)
    both = pd.concat({"nbbom": ma, "complete": mb}, axis=1).dropna()
    diff_bps = (both["nbbom"] / both["complete"] - 1).abs() * 1e4
    r_a = np.log(both["nbbom"]).diff(5)
    r_b = np.log(both["complete"]).diff(5)
    print(f"minutes compared: {len(both)}")
    print(f"share of minutes with a different mid: {(diff_bps > 1e-6).mean():.1%}")
    print(f"median |difference| when different: {diff_bps[diff_bps > 1e-6].median():.3f} bps")
    print(f"corr of 5-minute mid returns: {r_a.corr(r_b):.4f}")
    print("Rule of thumb: < 5% different minutes and corr > 0.99 -> nbbom bars are fine for 5 minutes.")


if __name__ == "__main__":
    main()

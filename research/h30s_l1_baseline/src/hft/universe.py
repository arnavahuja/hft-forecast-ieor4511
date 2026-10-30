"""Step 2: point-in-time universe (pilot targets that are S&P 500 members that day, plus benchmarks)."""

from pathlib import Path

import pandas as pd
import polars as pl

from hft.wrds_io import sp500_members, taq_symbols


def universe_dir(data_dir, date) -> Path:
    return Path(data_dir) / "universe" / f"date={pd.Timestamp(date):%Y-%m-%d}"


def build_universe(db, cfg, date, verbose=True) -> pd.DataFrame:
    """Save sp500_members.parquet and universe.parquet for `date`; return the universe.

    Universe columns: date, symbol (CTA ticker), sym_root, sym_suffix, cusip, role ('target'|'benchmark').
    """
    d = pd.Timestamp(date)
    out = universe_dir(cfg["data_dir"], d)
    out.mkdir(parents=True, exist_ok=True)

    members = sp500_members(db, d)
    members.to_parquet(out / "sp500_members.parquet", index=False)

    targets = taq_symbols(db, d, cfg["universe"]["tickers"])
    targets["in_sp500"] = targets["cusip"].isin(members["cusip9"])
    missing = targets.loc[targets["sym_root"].isna() | ~targets["in_sp500"], "ticker"].tolist()
    targets = targets[targets["sym_root"].notna() & targets["in_sp500"]].assign(role="target")

    bench = taq_symbols(db, d, cfg["universe"]["benchmarks"])
    missing += bench.loc[bench["sym_root"].isna(), "ticker"].tolist()
    bench = bench[bench["sym_root"].notna()].assign(role="benchmark")

    uni = pd.concat([targets, bench]).rename(columns={"ticker": "symbol"})
    uni = uni[["symbol", "sym_root", "sym_suffix", "cusip", "role"]].assign(date=d.date())
    uni.to_parquet(out / "universe.parquet", index=False)
    if verbose:
        n_fb = (members["link"] == "ticker").sum()
        print(f"  {d:%Y-%m-%d}: {len(members)} S&P 500 members ({n_fb} ticker-fallback links); "
              f"universe {(uni.role == 'target').sum()} targets + {(uni.role == 'benchmark').sum()} benchmarks"
              + (f"; MISSING: {missing}" if missing else ""))
    return uni


def load_universe(data_dir, date) -> pl.DataFrame:
    path = universe_dir(data_dir, date) / "universe.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run scripts/build_universe.py --date {pd.Timestamp(date):%Y-%m-%d}")
    return pl.read_parquet(path)

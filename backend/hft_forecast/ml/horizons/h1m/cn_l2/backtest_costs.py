"""Cost-aware evaluation of the 60 s China L2 forecasts.

Reads the per-window predictions written by the notebook (May validation and the June
locked test) and the 5 s bar cache (for each stock's price, hence its tick size in bps).
Thresholds are chosen on May only and applied unchanged to June.

Two uses of the forecast are scored:

1. Intraday round trips with inventory ("T+0 on a base position"). A-shares settle T+1,
   so a desk that already holds the stock can sell first and buy back, or buy first and
   sell from inventory, keeping its net position unchanged. Each trade enters over the
   15 s after the signal and exits 60 s later, which is exactly the label, so the gross
   return of a trade in the predicted direction is sign(prediction) * actual_bps.
   Costs per round trip: stamp duty on the sell leg, commission and exchange fees on
   both legs, and one tick for crossing half the spread on each leg (the spread is at
   least one tick).

2. Execution timing. An order that must be filled now or 60 s later pays the same fees
   either way, so the forecast is worth what it saves on price: a buyer waits when a fall
   is predicted, a seller waits when a rise is predicted. Saving is in bps per order,
   against always trading now.

Usage:
    python backtest_costs.py --predictions <results dir> --bars <trade_bars_5s dir> --out <dir>
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

STAMP_DUTY_BPS = 5.0          # 0.05% on the sell leg (rate since 28 Aug 2023)
COMMISSION_BPS = 1.0          # per leg, an institutional rate; retail is often 1.5 to 2.5
EXCHANGE_FEES_BPS = 0.5       # per leg: handling, regulatory and transfer fees, rounded up
QUANTILES = [0.50, 0.80, 0.90, 0.95, 0.99]   # trade the top (1 - q) of |prediction|


def tick_bps(bars_dir: Path, months: list[str]) -> pd.DataFrame:
    bars = pl.concat([pl.read_parquet(bars_dir / f"bars_{m}.parquet") for m in months])
    daily = (bars.group_by(["ticker", "date"])
             .agg((pl.col("amount").sum() / pl.col("volume").sum()).alias("vwap"))
             .with_columns((1e4 * 0.01 / pl.col("vwap")).alias("tick_bps"),
                           pl.col("date").str.strptime(pl.Date, "%Y%m%d").alias("session_date")))
    out = daily.select("ticker", "session_date", "tick_bps").to_pandas()
    out["session_date"] = pd.to_datetime(out["session_date"])
    return out


def load(path: Path, ticks: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["ticker", "session_date", "timestamp", "actual_bps", "predicted_bps"])
    df["session_date"] = pd.to_datetime(df["session_date"])
    df = df.merge(ticks, on=["ticker", "session_date"], how="left")
    assert df.tick_bps.notna().all(), "missing tick size"
    return df


def sign_test_p(pos: int, n: int) -> float:
    if n == 0:
        return float("nan")
    k = max(pos, n - pos)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) * 0.5 ** n)


def round_trips(df: pd.DataFrame, threshold: float, spread_ticks: float) -> dict:
    t = df.loc[df.predicted_bps.abs() > threshold].copy()
    fees = STAMP_DUTY_BPS + 2 * (COMMISSION_BPS + EXCHANGE_FEES_BPS)
    t["gross"] = np.sign(t.predicted_bps) * t.actual_bps
    t["cost"] = fees + spread_ticks * t.tick_bps
    t["net"] = t.gross - t.cost
    daily = t.groupby("session_date").net.sum()
    n = len(t)
    return {
        "trades": n,
        "trades_per_stock_day": n / max(1, df.groupby(["ticker", "session_date"]).ngroups),
        "hit_rate": float((t.gross > 0).mean()) if n else np.nan,
        "gross_bps": float(t.gross.mean()) if n else np.nan,
        "gross_t": float(t.gross.mean() / (t.gross.std() / np.sqrt(n))) if n > 1 else np.nan,
        "cost_bps": float(t.cost.mean()) if n else np.nan,
        "net_bps": float(t.net.mean()) if n else np.nan,
        "positive_days": int((daily > 0).sum()),
        "days": int(len(daily)),
        "days_sign_p": sign_test_p(int((daily > 0).sum()), int(len(daily))),
    }


def timing(df: pd.DataFrame) -> dict:
    # Buyer waits when a fall is predicted; seller waits when a rise is predicted.
    a, p = df.actual_bps.to_numpy(), df.predicted_bps.to_numpy()
    buyer = np.where(p < 0, -a, 0.0)
    seller = np.where(p > 0, a, 0.0)
    both = np.concatenate([buyer, seller])
    oracle = np.concatenate([np.maximum(-a, 0), np.maximum(a, 0)])
    return {
        "orders": int(len(both)),
        "saving_bps_per_order": float(both.mean()),
        "saving_t": float(both.mean() / (both.std() / np.sqrt(len(both)))),
        "oracle_saving_bps": float(oracle.mean()),
        "share_of_oracle": float(both.mean() / oracle.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--bars", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    ticks = tick_bps(args.bars, ["202505", "202506"])
    may = load(args.predictions / "best_ensemble_predictions.csv", ticks)
    june = load(args.predictions / "FINAL_TEST_predictions.csv", ticks)

    rows = []
    for q in QUANTILES:
        thr = float(may.predicted_bps.abs().quantile(q))          # chosen on May only
        for label, spread_ticks in (("taker (1 tick spread)", 1.0), ("fees only (no spread)", 0.0)):
            for period, df in (("May (validation)", may), ("June (locked test)", june)):
                rows.append({"trade_top_share": round(1 - q, 2), "threshold_bps": round(thr, 3),
                             "costs": label, "period": period, **round_trips(df, thr, spread_ticks)})
    rt = pd.DataFrame(rows)
    tm = pd.DataFrame([{"period": "May (validation)", **timing(may)},
                       {"period": "June (locked test)", **timing(june)}])
    args.out.mkdir(parents=True, exist_ok=True)
    rt.round(4).to_csv(args.out / "backtest_round_trips.csv", index=False)
    tm.round(4).to_csv(args.out / "backtest_execution_timing.csv", index=False)
    meta = {"stamp_duty_bps_sell": STAMP_DUTY_BPS, "commission_bps_per_leg": COMMISSION_BPS,
            "exchange_fees_bps_per_leg": EXCHANGE_FEES_BPS,
            "fixed_round_trip_bps": STAMP_DUTY_BPS + 2 * (COMMISSION_BPS + EXCHANGE_FEES_BPS),
            "june_mean_tick_bps": float(june.tick_bps.mean()),
            "june_actual_abs_mean_bps": float(june.actual_bps.abs().mean()),
            "june_predicted_abs_mean_bps": float(june.predicted_bps.abs().mean())}
    (args.out / "backtest_assumptions.json").write_text(json.dumps(meta, indent=1))
    pd.set_option("display.width", 220)
    print(json.dumps(meta, indent=1))
    print(rt.round(3).to_string(index=False))
    print(tm.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

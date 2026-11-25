"""Execution backtests for the 60 s China L2 forecasts: a forecast-steered schedule and
forecast-switched passive orders.

A-shares settle T+1 and short selling is restricted, so the realistic use of a 60 s forecast
is to execute orders a desk has to do anyway more cheaply, not to trade it on its own. Both
tests compare policies that buy (or sell) the same quantity, so fees and stamp duty are the
same in every policy and cancel; only the price paid differs.

Prices come from the 5 s trade bars. At a decision time t:
  P(t)   15 s VWAP of (t, t+15 s], the label's start price (a midquote proxy: it mixes
         buyer and seller initiated trades); P(t+60) = P(t) * exp(actual_bps / 1e4)
  L(t)   last trade price at or before t
  taker  pays half a tick over P (a one tick spread), as in backtest_costs.py

1. Schedule. A parent order is spread over the day's decision points (one every 2 minutes).
   TWAP executes one slice at each. The steered schedule may fall behind TWAP by up to K
   slices when the forecast is adverse (a buyer when a fall is predicted) and run ahead by
   up to K when it is favourable, catching up to the TWAP path otherwise; whatever is left
   executes at the last decision point. Saving is in bps of the parent order against TWAP.

2. Passive orders. A child order that must be done within 60 s either crosses now or rests
   one tick behind the last trade (a buy at L - 1 tick, so at or below the best bid). With
   trades only and no queue data, a resting buy counts as filled only if a later trade in
   (t, t+60 s] prints strictly below it ("trade-through", conservative); "touch" (a print at
   the limit) is shown as an optimistic bound. Unfilled orders cross at t+60 s. The switched
   policy rests the order unless the forecast says waiting is costly (a buyer when a rise
   above the threshold is predicted), in which case it crosses now.

Both tests also run with the forecast shuffled within each stock and month ("random"), which
keeps how often each policy acts but removes the information; the gap to it is the value of
the forecast. Buy and sell parents are scored together, so a market drift cancels out.

Thresholds and K are chosen on the validation month of each run and applied unchanged to
every later month.

Usage (paths default to the local layout used for the PR):
    python backtest_execution.py --bars <bar cache dir> --out results
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

TICK = 0.01


def runs(h1: Path, h2: Path, oos: Path) -> list[dict]:
    """Each run: validation file (thresholds are chosen here) and later evaluation files."""
    months = [f"2025{m:02d}" for m in range(7, 13)]
    first = {"run": "first half model (Jan to Apr train)", "val": ("202505", h1 / "best_ensemble_predictions.csv"),
             "eval": [("202506", "all 50 (locked test)", h1 / "FINAL_TEST_predictions.csv")]}
    first["eval"] += [(m, "22 continuing", oos / f"predictions_{m}.csv") for m in months]
    first["eval"] += [(m, "28 new (cold start)", oos / f"predictions_new_{m}.csv") for m in months[1:]]
    second = {"run": "second half model (Jul to Oct train)", "val": ("202511", h2 / "best_ensemble_predictions.csv"),
              "eval": [("202512", "all 50 (locked test)", h2 / "FINAL_TEST_predictions.csv")]}
    return [first, second]


def load(path: Path, bars_dir: Path, month: str) -> pd.DataFrame:
    """Predictions joined with P(t), P(t+60), L(t) and the low and high over (t, t+60 s]."""
    p = pl.read_csv(path, columns=["ticker", "session_date", "timestamp", "actual_bps", "predicted_bps"])
    ts = pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
    p = p.with_columns(pl.col("session_date").str.replace_all("-", "").alias("date"),
                       (ts.dt.hour().cast(pl.Int32) * 3600 + ts.dt.minute().cast(pl.Int32) * 60
                        + ts.dt.second().cast(pl.Int32)).alias("t"))
    b = pl.read_parquet(bars_dir / f"bars_{month}.parquet").select(
        "ticker", "date", "bucket", "last", "high", "low", "volume", "amount")
    off = pl.DataFrame({"k": list(range(5, 65, 5))}, schema={"k": pl.Int32})
    w = (p.select("ticker", "date", "t").join(off, how="cross")
         .with_columns((pl.col("t") + pl.col("k")).alias("bucket"))
         .join(b, on=["ticker", "date", "bucket"], how="left")
         .group_by("ticker", "date", "t")
         .agg((pl.col("amount").filter(pl.col("k") <= 15).sum()
               / pl.col("volume").filter(pl.col("k") <= 15).sum()).alias("p0"),
              pl.col("low").min().alias("lo60"), pl.col("high").max().alias("hi60")))
    last = b.select("ticker", "date", pl.col("bucket").alias("t"), "last").sort("ticker", "date", "t")
    p = (p.sort("ticker", "date", "t").join_asof(last, on="t", by=["ticker", "date"], strategy="backward")
         .join(w, on=["ticker", "date", "t"], how="left"))
    df = p.to_pandas()
    df = df[np.isfinite(df.p0) & df["last"].notna() & df.lo60.notna()].copy()
    df["p60"] = df.p0 * np.exp(df.actual_bps / 1e4)
    return df.sort_values(["ticker", "date", "t"]).reset_index(drop=True)


# ---------------------------------------------------------------- 1. steered schedule
def schedule(df: pd.DataFrame, thr: float, k: int) -> pd.DataFrame:
    """Saving (bps) of the steered schedule against TWAP, per stock-day and side."""
    out = []
    for (tk, d), g in df.groupby(["ticker", "date"], sort=False):
        price, pred = g.p0.to_numpy(), g.predicted_bps.to_numpy()
        n = len(price)
        if n < 10:
            continue
        twap = price.mean()
        for side, s in (("buy", 1.0), ("sell", -1.0)):
            cum, cost = 0, 0.0
            for i in range(n):
                target = i + 1
                if i == n - 1:
                    q = n - cum
                elif s * pred[i] < -thr:               # adverse move expected later: lag
                    q = max(0, target - k - cum)
                elif s * pred[i] > thr:                # price expected to move against us: run ahead
                    q = min(n, target + k) - cum
                else:
                    q = max(0, target - cum)
                cum += q; cost += q * price[i]
            avg = cost / n
            out.append({"ticker": tk, "date": d, "side": side, "slices": n,
                        "saving_bps": 1e4 * s * (twap - avg) / twap})
    return pd.DataFrame(out)


def summarize_schedule(r: pd.DataFrame) -> dict:
    x = r.saving_bps
    daily = r.groupby("date").saving_bps.mean()
    return {"parent_orders": len(x), "saving_bps": x.mean(), "saving_t": x.mean() / (x.std() / np.sqrt(len(x))),
            "orders_positive": (x > 0).mean(), "days_positive": (daily > 0).mean(), "days": len(daily)}


# ---------------------------------------------------------------- 2. passive vs taker
def passive(df: pd.DataFrame, thr: float | None, rule: str) -> dict:
    """Saving (bps) per child order against crossing now; thr None means always rest."""
    res = {}
    for side, s in (("buy", 1.0), ("sell", -1.0)):
        p0, p60, last, pred = df.p0.to_numpy(), df.p60.to_numpy(), df["last"].to_numpy(), df.predicted_bps.to_numpy()
        limit = last - s * TICK
        if rule == "trade-through":
            filled = (df.lo60.to_numpy() < limit - 1e-9) if s > 0 else (df.hi60.to_numpy() > limit + 1e-9)
        else:
            filled = (df.lo60.to_numpy() <= limit + 1e-9) if s > 0 else (df.hi60.to_numpy() >= limit - 1e-9)
        take_now = p0 + s * TICK / 2
        rest = np.where(filled, limit, p60 + s * TICK / 2)
        cross = np.zeros(len(df), bool) if thr is None else (s * pred > thr)
        price = np.where(cross, take_now, rest)
        res[side] = (1e4 * s * (take_now - price) / p0, filled & ~cross, cross)
    sav = np.concatenate([res["buy"][0], res["sell"][0]])
    fill = np.concatenate([res["buy"][1], res["sell"][1]])
    cross = np.concatenate([res["buy"][2], res["sell"][2]])
    return {"orders": len(sav), "saving_bps": sav.mean(), "saving_t": sav.mean() / (sav.std() / np.sqrt(len(sav))),
            "share_crossed_now": cross.mean(), "share_filled_passive": fill.mean(),
            "share_crossed_late": 1 - cross.mean() - fill.mean()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", type=Path, required=True)
    ap.add_argument("--h1", type=Path, required=True, help="first half run results (May and June predictions)")
    ap.add_argument("--h2", type=Path, required=True, help="second half run results (November and December)")
    ap.add_argument("--oos", type=Path, required=True, help="oos_eval.py outputs (July to December)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sched_rows, pas_rows, chosen, sgrid = [], [], [], []
    rng = np.random.default_rng(0)
    shuffled = lambda df: df.assign(predicted_bps=df.groupby("ticker").predicted_bps.transform(lambda x: rng.permutation(x.to_numpy())))
    for run in runs(a.h1, a.h2, a.oos):
        vm, vpath = run["val"]
        val = load(vpath, a.bars, vm)
        qs = {q: float(val.predicted_bps.abs().quantile(q)) for q in (0.0, 0.5, 0.8, 0.9)}
        # Schedule: pick (threshold, K) on validation.
        grid = [(q, k) for q in qs for k in (1, 2, 4, 8)]
        vres = {(q, k): summarize_schedule(schedule(val, qs[q], k)) for q, k in grid}
        qb, kb = max(vres, key=lambda g: vres[g]["saving_bps"])
        sgrid += [{"run": run["run"], "validation": vm, "quantile": q, "threshold_bps": qs[q], "K": k, **v} for (q, k), v in vres.items()]
        # Passive: pick the crossing threshold on validation, separately for each fill rule.
        pb = {}
        for rule in ("trade-through", "touch"):
            pv = {q: passive(val, qs[q], rule)["saving_bps"] for q in qs}
            pb[rule] = max(pv, key=pv.get)
        chosen.append({"run": run["run"], "validation": vm, "schedule_quantile": qb, "schedule_threshold_bps": qs[qb],
                       "schedule_K": kb, **{f"passive_quantile_{r}": q for r, q in pb.items()},
                       **{f"passive_threshold_bps_{r}": qs[q] for r, q in pb.items()}})
        print(run["run"], "| chosen on", vm, chosen[-1], flush=True)
        periods = [(vm, "validation", val)] + [(m, grp, load(path, a.bars, m)) for m, grp, path in run["eval"] if path.exists()]
        for m, grp, df in periods:
            sh = shuffled(df)
            s = summarize_schedule(schedule(df, qs[qb], kb)); s0 = summarize_schedule(schedule(sh, qs[qb], kb))
            sched_rows.append({"run": run["run"], "month": m, "stocks": grp, **s,
                               "random_saving_bps": s0["saving_bps"], "random_saving_t": s0["saving_t"]})
            for rule in ("trade-through", "touch"):
                for label, d, thr in (("always rest", df, None), ("forecast-switched", df, qs[pb[rule]]),
                                      ("random-switched", sh, qs[pb[rule]])):
                    pas_rows.append({"run": run["run"], "month": m, "stocks": grp, "fill_rule": rule, "policy": label,
                                     **passive(d, thr, rule)})
            f = {(x["fill_rule"], x["policy"]): x["saving_bps"] for x in pas_rows[-6:]}
            print(f"  {m} {grp:22s} schedule {s['saving_bps']:+.2f} (random {s0['saving_bps']:+.2f}, t {s['saving_t']:.1f}) | passive vs cross now: "
                  + " ".join(f"{r}: rest {f[(r, 'always rest')]:+.2f} switched {f[(r, 'forecast-switched')]:+.2f} random {f[(r, 'random-switched')]:+.2f}"
                             for r in ("trade-through", "touch")), flush=True)
    pd.DataFrame(chosen).to_csv(a.out / "execution_choices.csv", index=False)
    pd.DataFrame(sgrid).round(4).to_csv(a.out / "execution_schedule_validation_grid.csv", index=False)
    pd.DataFrame(sched_rows).round(4).to_csv(a.out / "execution_schedule.csv", index=False)
    pd.DataFrame(pas_rows).round(4).to_csv(a.out / "execution_passive.csv", index=False)


if __name__ == "__main__":
    main()

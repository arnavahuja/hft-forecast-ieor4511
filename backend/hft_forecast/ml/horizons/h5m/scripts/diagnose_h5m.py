"""Is the 5-minute IC real or an artefact of how it is measured?  (No model fitting: not a trial.)

The EDA found f_spread_bps with a within-session IC of +0.08 at 5 minutes, rising to +0.17 at
30 minutes. A spread has no direction, so that is suspicious. One known cause: f_spread_bps =
spread / mid moves inversely with the price level, and ranking a level against its own future
change inside a short window is biased even for a random walk (the bias grows with the horizon).

This script separates the two stories, using the built dataset and, if given, a run's predictions:

  A. the same IC three ways for each variable:
       within_session   rank inside each (stock, session)        -> the EDA number
       within_stock     rank inside each stock over all 124 days -> window bias mostly gone
       cross_section    rank across the 47 stocks at each minute -> no time-series bias at all
  B. placebo variables: 1/mid (pure price level) and the spread in ticks / dollars (no price level)
  C. the tradeable version: target = 5-minute return starting one bar later
  D. for the model run: the same three ICs of y_hat, plus a cost-free long/short test:
     sign(y_hat) * next-bar 5-minute return, averaged per session, with its t-statistic

    python backend/hft_forecast/ml/horizons/h5m/scripts/diagnose_h5m.py ^
        backend/hft_forecast/ml/horizons/h5m/config_taq.yaml --run a569afa2eba8

Writes <HFTF_DATA_DIR>/reports/diagnose_h5m.md
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import polars as pl

from hft_forecast.ml.config import ExperimentConfig
from hft_forecast.ml.datasets import load_dataset, usable_rows

GROUP = ["instrument_id", "session_id"]
TICK = 0.01


def ic_three_ways(frame: pl.DataFrame, var: str, target: str) -> dict[str, float]:
    sub = frame.filter(pl.col(var).is_finite() & pl.col(target).is_finite())

    def mean_t(groups: list[str]) -> tuple[float, float, int]:
        ics = sub.group_by(groups).agg(pl.corr(var, target, method="spearman").alias("ic"))
        ics = ics.filter(pl.col("ic").is_finite())["ic"]
        n = len(ics)
        if n < 2:
            return float("nan"), float("nan"), n
        m, s = float(ics.mean()), float(ics.std())
        return m, m / (s / math.sqrt(n)) if s > 0 else float("nan"), n

    ws, ws_t, _ = mean_t(GROUP)
    wk, wk_t, _ = mean_t(["instrument_id"])
    cs, cs_t, _ = mean_t(["timestamp"])
    return {"within_session": ws, "t_ws": ws_t, "within_stock": wk, "t_wk": wk_t,
            "cross_section": cs, "t_cs": cs_t}


def table(rows: list[dict], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        out.append("| " + " | ".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])
                                     for c in cols) + " |")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--run", default=None, help="ml run id whose predictions to test")
    args = parser.parse_args()
    root = Path(os.environ.get("HFTF_DATA_DIR", "data")).resolve()
    config = ExperimentConfig.load(args.config)
    frame = load_dataset(root / "features" / config.dataset_hash())
    lm = pl.col("mid").log()
    frame = frame.with_columns(
        (lm.shift(-6).over(GROUP) - lm.shift(-1).over(GROUP)).alias("r5_next"),
        (1.0 / pl.col("mid")).alias("inv_mid"),
        ((pl.col("ask") - pl.col("bid")) / TICK).round(0).alias("spread_ticks"),
        (pl.col("ask") - pl.col("bid")).alias("spread_dollars"),
    )
    rows = usable_rows(frame)
    print(f"{rows.height:,} usable rows")

    variables = ["f_spread_bps", "inv_mid", "spread_ticks", "spread_dollars",
                 "f_imbalance", "f_microprice_bps", "f_logret_5", "f_logret_10", "f_vol_30"]
    part_a = []
    for target in ("fwd_logret", "r5_next"):
        for var in variables:
            if var in rows.columns:
                r = {"variable": var, "target": target}
                r.update(ic_three_ways(rows, var, target))
                part_a.append(r)
                print(r)

    md = ["# h5m IC diagnosis", "",
          "fwd_logret = 5-minute mid return from t; r5_next = 5-minute return from t+1 (tradeable).",
          "If an IC is large within_session but near 0 within_stock and cross_section, it is a",
          "measurement artefact, not a signal. inv_mid is the placebo: pure price level.", "",
          table(part_a, ["variable", "target", "within_session", "t_ws", "within_stock", "t_wk",
                         "cross_section", "t_cs"])]

    if args.run:
        pred = pl.read_parquet(root / "runs" / args.run / "predictions.parquet")
        keys = ["timestamp", "instrument_id"]
        extra = rows.select(keys + ["r5_next"])
        part_d = []
        for model in sorted(pred["model"].unique().to_list()):
            p = pred.filter(pl.col("model") == model).join(extra, on=keys, how="inner")
            if p["y_hat_logret"].std() in (None, 0.0):
                continue
            r = {"model": model, "target": "fwd_logret"}
            r.update(ic_three_ways(p, "y_hat_logret", "fwd_logret"))
            part_d.append(r)
            r2 = {"model": model, "target": "r5_next"}
            r2.update(ic_three_ways(p, "y_hat_logret", "r5_next"))
            part_d.append(r2)
            # cost-free long/short on the next-bar return, one number per session
            pnl = p.filter(pl.col("r5_next").is_finite()).with_columns(
                (pl.col("y_hat_logret").sign() * pl.col("r5_next") * 1e4).alias("pnl_bps"))
            daily = pnl.group_by("session_id").agg(pl.col("pnl_bps").mean())["pnl_bps"]
            m, s = float(daily.mean()), float(daily.std())
            r3 = {"model": model, "target": "cost-free sign(y_hat) x r5_next, bps per bar",
                  "within_session": m, "t_ws": m / (s / math.sqrt(len(daily))) if s else float("nan"),
                  "within_stock": float("nan"), "t_wk": float("nan"),
                  "cross_section": float("nan"), "t_cs": float("nan")}
            part_d.append(r3)
            print(r, r2, r3)
        md += ["", f"## Run {args.run}: model predictions", "",
               table(part_d, ["model", "target", "within_session", "t_ws", "within_stock", "t_wk",
                              "cross_section", "t_cs"]),
               "", "Compare the cost-free bps per bar with the median quoted spread (about 2.4 bps):",
               "a round trip costs roughly one spread, and a position is held for 5 bars."]
    out = root / "reports" / "diagnose_h5m.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

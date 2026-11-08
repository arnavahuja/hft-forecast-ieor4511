"""Signal decay with execution delay: cross-sectional IC of a predictor against the
return that starts L bars after the decision and lasts H bars. Not a trial (no fitting).

usage: latency_curve.py <config.yaml> <run_id>
"""
import math
import os
import sys
from pathlib import Path

import polars as pl

from hft_forecast.ml.config import ExperimentConfig
from hft_forecast.ml.datasets import load_dataset, usable_rows

G = ["instrument_id", "session_id"]
LAGS = range(0, 6)
HOLDS = (1, 5)

root = Path(os.environ["HFTF_DATA_DIR"])
cfg = ExperimentConfig.load(Path(sys.argv[1]))
run_id = sys.argv[2]

frame = load_dataset(root / "features" / cfg.dataset_hash())
lm = pl.col("mid").log()
targets = [(lm.shift(-(lag + hold)).over(G) - lm.shift(-lag).over(G)).alias(f"r_L{lag}_H{hold}")
           for lag in LAGS for hold in HOLDS]
frame = usable_rows(frame.with_columns(targets))

pred = pl.read_parquet(root / "runs" / run_id / "predictions.parquet")
pred = pred.filter(pl.col("model").str.starts_with("logit_ridge"))
frame = frame.join(pred.select(["timestamp", "instrument_id", "y_hat_logret"]),
                   on=["timestamp", "instrument_id"], how="inner")
print(f"{frame.height:,} rows with out-of-sample predictions")

rows = []
for var in ["f_imbalance", "f_microprice_bps", "f_logret_5", "y_hat_logret"]:
    for lag in LAGS:
        for hold in HOLDS:
            t = f"r_L{lag}_H{hold}"
            sub = frame.filter(pl.col(t).is_finite())
            ic = (sub.group_by("timestamp")
                     .agg(pl.corr(var, t, method="spearman").alias("ic"))
                     .filter(pl.col("ic").is_finite())["ic"])
            pnl = (sub.select((pl.col(var).sign() * pl.col(t) * 1e4).alias("p"), "session_id")
                      .group_by("session_id").agg(pl.col("p").mean())["p"])
            rows.append({"var": var, "lag": lag, "hold": hold,
                         "ic_cs": round(ic.mean(), 4),
                         "t_ic": round(ic.mean() / (ic.std() / math.sqrt(len(ic))), 1),
                         "gross_bps_per_trade": round(pnl.mean(), 3),
                         "t_pnl": round(pnl.mean() / (pnl.std() / math.sqrt(len(pnl))), 2)})
out = pl.DataFrame(rows)
pl.Config.set_tbl_rows(100)
print(out)
dest = root / "reports" / "latency_curve.csv"
out.write_csv(dest)
print(f"wrote {dest}")
print("median half-spread (bps):", round(frame["spread_bps"].median() / 2, 2))
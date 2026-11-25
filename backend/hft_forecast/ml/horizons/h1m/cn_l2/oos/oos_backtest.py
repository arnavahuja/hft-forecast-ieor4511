"""Monthly cost-aware backtest of the frozen model, June to December 2025, May thresholds."""
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import backtest_costs as bt  # noqa: E402

RUN = Path(os.environ.get("RESULTS_DIR", "cn_l2_60s_research/results_60s_multihorizon"))
MONTHS = [f"2025{m:02d}" for m in range(6, 13)]
ticks = bt.tick_bps(Path(os.environ.get("BARS_DIR", "bars")), ["202505"] + MONTHS)
may = bt.load(RUN / "best_ensemble_predictions.csv", ticks)
thresholds = {q: float(may.predicted_bps.abs().quantile(q)) for q in (0.90, 0.95, 0.99)}

rows = []
for m in MONTHS:
    df = bt.load(Path("out") / f"predictions_{m}.csv", ticks)
    row = {"month": m, "windows": len(df), "mean_tick_bps": df.tick_bps.mean(), **{
        f"timing_{k}": v for k, v in bt.timing(df).items() if k in ("saving_bps_per_order", "saving_t", "share_of_oracle")}}
    for q, thr in thresholds.items():
        top = f"top{round((1 - q) * 100)}"
        taker = bt.round_trips(df, thr, 1.0); fees = bt.round_trips(df, thr, 0.0)
        row.update({f"{top}_trades": taker["trades"], f"{top}_hit": taker["hit_rate"],
                    f"{top}_gross": taker["gross_bps"], f"{top}_net_taker": taker["net_bps"],
                    f"{top}_net_fees_only": fees["net_bps"]})
    rows.append(row)
res = pd.DataFrame(rows)
res.round(4).to_csv("out/monthly_backtest.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print("May thresholds (bps):", {k: round(v, 3) for k, v in thresholds.items()})
print(res.round(3).to_string(index=False))

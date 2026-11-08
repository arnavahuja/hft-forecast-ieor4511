"""One evaluation table for every h5m trial, with day-level statistics. Not a trial (no fitting).

Target (every trial, same definition): the tradeable 5-minute return
    r = log mid(t+6) - log mid(t+1), in bps         (decide at t, fill at t+1, hold 5 bars)

For each trial and for BOTH model heads
    s_ret  = y_hat_logret        the return head   (primary score, fixed in advance)
    s_prob = p_up - p_down       the direction head (secondary; what `hftf ml econ` trades on)
    s_cost = y_hat / spread      the return head per unit of quoted spread (cost-aware ranking,
                                 pre-registered on 2026-11-07 before T6-T8 were run)
it reports
    ic_cs      cross-sectional Spearman IC (same minute, across stocks), averaged per day,
               t-statistic over days (one number per day removes the overlap of 5-minute windows)
    ic_stock   within-stock IC over the whole test period, t-statistic over the 47 stocks
    ls_bps     long the top 10% / short the bottom 10% of stocks by score each minute, held
               5 minutes: gross return of the pair in bps, per day, t over days
    ls_cost    round-trip cost of that pair: quoted spread of the long leg + of the short leg
    ls_net     ls_bps - ls_cost
and, for the return head only, the selective rule fixed in advance:
    trade only when |y_hat| > half the quoted spread; share of rows that trade and the net
    bps per trade after paying the full quoted spread (round trip).

usage (from the repository root, environment active):
    python backend/hft_forecast/ml/horizons/h5m/scripts/eval_h5m.py --selftest
    python backend/hft_forecast/ml/horizons/h5m/scripts/eval_h5m.py ^
        T1=config_taq.yaml:a569afa2eba8:logit_ridge ^
        T3=config_taq_lag1.yaml:dac307569977:logit_ridge ...
Writes <HFTF_DATA_DIR>/reports/eval_h5m.md and eval_h5m.csv.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import polars as pl

H5M = Path(__file__).resolve().parents[1]
GROUP = ["instrument_id", "session_id"]
HOLD = 5
LAG = 1
TAIL = 0.1
NAN = float("nan")


def mean_t(values: pl.Series) -> tuple[float, float, int]:
    """Mean, t-statistic of the mean, and count, ignoring nulls and NaNs."""
    s = values.cast(pl.Float64).drop_nulls().drop_nans()
    n = len(s)
    if n < 2:
        return NAN, NAN, n
    m, sd = float(s.mean()), float(s.std())
    return m, (m / (sd / math.sqrt(n)) if sd > 0 else NAN), n


def metrics(df: pl.DataFrame, score: str, selective: bool) -> dict[str, float]:
    """df: timestamp, instrument_id, session_id, r (bps), spr (bps), and the score column."""
    d = df.filter(pl.col("r").is_finite() & pl.col(score).is_finite())
    out: dict[str, float] = {}

    # cross-sectional IC per minute -> per day -> t over days
    per_min = (
        d.group_by("timestamp")
        .agg(pl.corr(score, "r", method="spearman").alias("ic"), pl.col("session_id").first())
        .filter(pl.col("ic").is_finite())
    )
    per_day = per_min.group_by("session_id").agg(pl.col("ic").mean())["ic"]
    out["ic_cs"], out["t_ic_cs"], out["days"] = mean_t(per_day)

    # within-stock IC over the whole test period -> t over stocks
    per_stock = d.group_by("instrument_id").agg(pl.corr(score, "r", method="spearman").alias("ic"))
    out["ic_stock"], out["t_ic_stock"], out["stocks"] = mean_t(per_stock["ic"])

    # long top 10% / short bottom 10% each minute
    n = pl.col("timestamp").count().over("timestamp")
    k = (n * TAIL).floor().clip(lower_bound=1)
    rank = pl.col(score).rank("ordinal").over("timestamp")
    legs = d.with_columns((rank > n - k).alias("long"), (rank <= k).alias("short"))
    pair = (
        legs.group_by("timestamp")
        .agg(
            (pl.col("r").filter(pl.col("long")).mean() - pl.col("r").filter(pl.col("short")).mean())
            .alias("ls"),
            (pl.col("spr").filter(pl.col("long")).mean() + pl.col("spr").filter(pl.col("short")).mean())
            .alias("cost"),
            pl.col("session_id").first(),
        )
        .filter(pl.col("ls").is_finite())
    )
    daily = pair.group_by("session_id").agg(
        pl.col("ls").mean(), pl.col("cost").mean(), (pl.col("ls") - pl.col("cost")).mean().alias("net")
    )
    out["ls_bps"], out["t_ls"], _ = mean_t(daily["ls"])
    out["ls_cost"], _, _ = mean_t(daily["cost"])
    out["ls_net"], out["t_ls_net"], _ = mean_t(daily["net"])

    if selective:  # trade only when the predicted move exceeds the half spread
        trig = d.with_columns(
            ((pl.col(score) * 1e4).abs() > pl.col("spr") / 2).alias("trade"),
            (pl.col(score).sign() * pl.col("r") - pl.col("spr")).alias("net"),
        )
        out["sel_share"] = float(trig["trade"].mean())
        taken = trig.filter(pl.col("trade"))
        out["sel_trades"] = float(taken.height)
        if taken.height:
            by_day = taken.group_by("session_id").agg(pl.col("net").mean())["net"]
            out["sel_net_bps"], out["t_sel"], _ = mean_t(by_day)
        else:
            out["sel_net_bps"], out["t_sel"] = NAN, NAN
    return out


def load_trial(root: Path, cfg_file: str, run_id: str, model: str) -> pl.DataFrame:
    from hft_forecast.ml.config import ExperimentConfig
    from hft_forecast.ml.datasets import load_dataset, usable_rows

    cfg = ExperimentConfig.load(H5M / cfg_file)
    frame = load_dataset(root / "features" / cfg.dataset_hash())
    lm = pl.col("mid").log()
    # targets are computed on the full grid BEFORE rows are filtered, so shifts stay aligned
    frame = frame.with_columns(
        ((lm.shift(-(LAG + HOLD)).over(GROUP) - lm.shift(-LAG).over(GROUP)) * 1e4).alias("r"),
        ((pl.col("ask") - pl.col("bid")) / pl.col("mid") * 1e4).alias("spr"),
    )
    frame = usable_rows(frame).select(["timestamp", "instrument_id", "session_id", "r", "spr", "mid"])
    pred = pl.read_parquet(root / "runs" / run_id / "predictions.parquet")
    pred = pred.filter(pl.col("model").str.starts_with(model))
    names = pred["model"].unique().to_list()
    if len(names) != 1:
        sys.exit(f"{run_id}: expected one model starting with {model!r}, found {names}")
    pred = pred.select(
        "timestamp",
        "instrument_id",
        pl.col("y_hat_logret").alias("s_ret"),
        (pl.col("p_up") - pl.col("p_down")).alias("s_prob"),
    )
    df = frame.join(pred, on=["timestamp", "instrument_id"], how="inner")
    df = df.with_columns((pl.col("s_ret") * 1e4 / pl.col("spr")).alias("s_cost"))
    print(f"{run_id} {names[0]}: {df.height:,} rows with out-of-sample predictions")
    return df


def selftest() -> None:
    """Synthetic data with known answers: every metric must come out as designed."""
    import numpy as np

    rng = np.random.default_rng(0)
    days, minutes, stocks = 20, 60, 47
    rows = []
    for d in range(days):
        for m in range(minutes):
            for s in range(stocks):
                rows.append((d * 1000 + m, f"S{s:02d}", d))
    base = pl.DataFrame(rows, schema=["timestamp", "instrument_id", "session_id"], orient="row")
    r = rng.normal(0, 10, base.height)
    noise = rng.normal(0, 10, base.height)
    df = base.with_columns(
        pl.Series("r", r),
        pl.lit(2.0).alias("spr"),
        pl.Series("s_ret", r / 1e4),  # perfect return forecast, in log-return units
        pl.Series("s_prob", noise),  # pure noise
    )
    good = metrics(df, "s_ret", selective=True)
    bad = metrics(df, "s_prob", selective=False)
    checks = {
        "perfect score: IC = 1": abs(good["ic_cs"] - 1) < 1e-9,
        "perfect score: within-stock IC = 1": abs(good["ic_stock"] - 1) < 1e-9,
        "perfect score: long-short clearly positive": good["ls_bps"] > 20 and good["t_ls"] > 10,
        "cost of a pair = 2 spreads = 4 bps": abs(good["ls_cost"] - 4.0) < 1e-9,
        "net = gross - cost": abs(good["ls_net"] - (good["ls_bps"] - 4.0)) < 1e-6,
        "selective: trades only |forecast| > 1 bp": 0.85 < good["sel_share"] < 0.97,
        "noise score: |IC| small": abs(bad["ic_cs"]) < 0.02,
        "noise score: long-short not significant": abs(bad["t_ls"]) < 3,
        "20 days, 47 stocks": good["days"] == days and good["stocks"] == stocks,
    }
    for name, ok in checks.items():
        print(("PASS " if ok else "FAIL ") + name)
    if not all(checks.values()):
        sys.exit("selftest FAILED")
    print("selftest passed")


def fmt(x: float) -> str:
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"


def main() -> None:
    if sys.argv[1:] == ["--selftest"]:
        selftest()
        return
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    root = Path(os.environ["HFTF_DATA_DIR"])
    table = []
    for arg in sys.argv[1:]:
        label, spec = arg.split("=", 1)
        cfg_file, run_id, model = spec.split(":")
        df = load_trial(root, cfg_file, run_id, model)
        for score, head in (("s_ret", "y_hat"), ("s_prob", "p_up-p_down"),
                            ("s_cost", "y_hat/spread")):
            m = metrics(df, score, selective=(score == "s_ret"))
            table.append({"trial": label, "run": run_id, "model": model, "head": head, **m})
            print(label, head, {k: round(v, 4) for k, v in m.items() if isinstance(v, float)})
    out = pl.DataFrame(table)
    reports = root / "reports"
    out.write_csv(reports / "eval_h5m.csv")
    cols = ["trial", "head", "ic_cs", "t_ic_cs", "ic_stock", "t_ic_stock", "ls_bps", "t_ls",
            "ls_cost", "ls_net", "sel_share", "sel_net_bps", "t_sel"]
    lines = [
        "# h5m evaluation (tradeable target: log mid(t+6) - log mid(t+1))",
        "",
        "t-statistics: ic_cs and ls over test days; ic_stock over stocks. Returns in bps.",
        "ls = long top 10% / short bottom 10% each minute, held 5 minutes; ls_cost = spread of",
        "both legs (round trip). sel = trade only when |y_hat| > half spread (rule fixed in advance).",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in out.iter_rows(named=True):
        lines.append("| " + " | ".join(
            str(row.get(c, "")) if c in ("trial", "head") else fmt(row.get(c, NAN)) for c in cols
        ) + " |")
    (reports / "eval_h5m.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {reports / 'eval_h5m.md'} and eval_h5m.csv")


if __name__ == "__main__":
    main()

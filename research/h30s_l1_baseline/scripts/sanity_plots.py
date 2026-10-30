"""Step 3 sanity outputs: mid and spread through the day + drop counts, for a few symbol-days.

    python scripts/sanity_plots.py --date 2025-12-10                 # symbols from config clean.plot_symbols
    python scripts/sanity_plots.py --date 2025-12-10 --symbol NVDA

Writes reports/figures/clean_{SYMBOL}_{DATE}.png and reports/clean_sanity_{DATE}.md
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl

from hft.config import ROOT, load_config


def plot_symbol(nbbo: pl.DataFrame, trades: pl.DataFrame, symbol: str, day: str, out: Path):
    q = nbbo.filter(pl.col("symbol") == symbol)
    t = trades.filter(pl.col("symbol") == symbol)
    # 1-second last mid / spread for plotting only
    s = q.group_by_dynamic("ts", every="1s").agg(pl.col("mid").last(), pl.col("spread").last()).to_pandas()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, height_ratios=[2, 1])
    a1.plot(s["ts"], s["mid"], lw=0.6, color="tab:blue", label="mid (1s last)")
    tp = t.filter(~pl.col("is_trf")).sample(min(len(t), 3000), seed=0).sort("ts").to_pandas() if len(t) else None
    if tp is not None and len(tp):
        a1.scatter(tp["ts"], tp["price"], s=1, color="0.6", alpha=0.5, label="exchange trades (sample)")
    a1.set_ylabel("price ($)")
    a1.legend(loc="upper left", fontsize=8)
    a1.set_title(f"{symbol} {day}: {len(q):,} clean L1 updates, {len(t):,} clean trades")
    a2.plot(s["ts"], s["spread"] * 100, lw=0.6, color="tab:red")
    a2.set_ylabel("spread (cents)")
    a2.set_xlabel("time (ET)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True)
    ap.add_argument("--symbol", action="append")
    ap.add_argument("--config", default="config/pilot.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = pd.Timestamp(args.date)
    day, part = f"{d:%Y-%m-%d}", f"date={d:%Y-%m-%d}"
    data = Path(cfg["data_dir"])
    symbols = args.symbol or cfg["clean"]["plot_symbols"]

    nbbo = pl.scan_parquet(data / "clean" / "nbbo" / part / "part.parquet").filter(pl.col("symbol").is_in(symbols)).collect()
    trades = pl.scan_parquet(data / "clean" / "trades" / part / "part.parquet").filter(pl.col("symbol").is_in(symbols)).collect()
    drops = pl.read_parquet(data / "clean" / "drops" / part / "drops.parquet")

    fig_dir = ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# Cleaning sanity check — {day}", ""]
    for sym in symbols:
        png = fig_dir / f"clean_{sym.replace('.', '')}_{day}.png"
        plot_symbol(nbbo, trades, sym, day, png)
        lines += [f"## {sym}", "", f"![{sym}](figures/{png.name})", ""]
        for kind in ["nbbo", "trades"]:
            tab = (drops.filter((pl.col("symbol") == sym) & (pl.col("dataset") == kind))
                   .group_by("reason").agg(pl.col("n").sum()).sort("n", descending=True))
            total = tab["n"].sum()
            lines += [f"**{kind}** (raw rows: {total:,})", "", "| reason | rows | share |", "|---|---:|---:|"]
            lines += [f"| {r} | {n:,} | {n / total:.2%} |" for r, n in tab.iter_rows()] + [""]
        codes = (drops.filter((pl.col("symbol") == sym) & (pl.col("reason") == "sale_condition"))
                 .sort("n", descending=True).head(8))
        if len(codes):
            lines += ["Dropped sale-condition codes (top 8): " +
                      ", ".join(f"`{c}` {n:,}" for c, n in codes.select("detail", "n").iter_rows()), ""]
    report = ROOT / "reports" / f"clean_sanity_{day}.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {report} and {len(symbols)} figures")


if __name__ == "__main__":
    main()

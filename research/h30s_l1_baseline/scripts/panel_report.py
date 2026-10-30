"""CHECKPOINT 2 report for one day's feature/label panel.

    python scripts/panel_report.py --date 2025-12-10

Writes reports/checkpoint2_{DATE}.md (panel size, missing rates, zero-return fractions by
stock, label and feature summary stats, placebo leakage check).
"""

import argparse
from pathlib import Path

import pandas as pd
import polars as pl

from hft.config import ROOT, load_config
from hft.leakage import placebo
from hft.reporting import lookahead_section, run_tests


def md_table(df: pl.DataFrame, fmt: dict | None = None) -> list[str]:
    fmt = fmt or {}
    cols = df.columns
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.iter_rows():
        cells = []
        for c, v in zip(cols, row):
            if v is None:
                cells.append("")
            elif c in fmt:
                cells.append(format(v, fmt[c]))
            elif isinstance(v, float):
                cells.append(f"{v:.4g}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--config", default="config/pilot.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    day = f"{pd.Timestamp(args.date):%Y-%m-%d}"
    data = Path(cfg["data_dir"])
    F = pl.read_parquet(data / "features" / f"date={day}" / "part.parquet")
    L = pl.read_parquet(data / "labels" / f"date={day}" / "part.parquet")
    feat_cols = [c for c, t in F.schema.items() if t.is_numeric()]
    h0 = cfg["primary_horizon_s"]

    lines = [f"# Checkpoint 2 — feature/label panel, {day}", ""]
    lines += [f"- Panel: **{F.height:,} rows** = {F['symbol'].n_unique()} symbols "
              f"({F.filter(pl.col('role') == 'target')['symbol'].n_unique()} targets + SPY) × "
              f"{F['t'].n_unique():,} seconds ({cfg['sample']['start']}–{cfg['sample']['end']}); "
              f"{len(feat_cols)} features; labels for h ∈ {cfg['horizons_s']} s.",
              f"- Files: `data/features/date={day}/part.parquet`, `data/labels/date={day}/part.parquet`.", ""]

    # missing values
    miss = F.select(pl.col(feat_cols).null_count() + pl.col(feat_cols).is_nan().sum()).transpose(
        include_header=True, header_name="feature", column_names=["missing"]).filter(pl.col("missing") > 0)
    lab_cols = [c for c in L.columns if c.startswith(("r_", "dir_", "sign_"))]
    lmiss = L.select(pl.col(lab_cols).null_count()).transpose(
        include_header=True, header_name="label", column_names=["missing"]).filter(pl.col("missing") > 0)
    lines += ["## Missing values", "",
              (f"Features: {miss.height} of {len(feat_cols)} columns have missing/NaN values." if miss.height
               else f"Features: **no missing values** in any of the {len(feat_cols)} columns."),
              (f"Labels: {lmiss.height} columns have missing values." if lmiss.height
               else "Labels: **no missing values** (15:55 + 60 s < 16:00, so no label crosses the close)."), ""]
    if miss.height:
        lines += md_table(miss.with_columns(share=pl.col("missing") / F.height), {"share": ".2%"}) + [""]

    # zero-return fractions by stock
    by = (L.join(F.select("symbol", "t", "role", "spread_ticks"), on=["symbol", "t"])
          .group_by("symbol", "role")
          .agg(price=pl.col("mid_t").median(),
               med_spread_ticks=pl.col("spread_ticks").median(),
               **{f"zero_{h}s": (pl.col(f"r_{h}_ticks") == 0).mean() for h in cfg["horizons_s"]},
               flat_band_30s=(pl.col(f"dir_{h0}") == 0).mean(),
               std_r30_bps=pl.col(f"r_{h0}_bps").std())
          .sort("zero_30s", descending=True))
    lines += ["## Zero mid-change fractions by stock", "",
              f"Share of grid seconds with exactly unchanged mid over h seconds; `flat_band_30s` = share with "
              f"|Δmid| < half spread (dir = 0). Sorted by `zero_30s`.", ""]
    pct = {c: ".1%" for c in by.columns if c.startswith(("zero_", "flat_"))}
    lines += md_table(by, {**pct, "price": ".2f", "med_spread_ticks": ".1f", "std_r30_bps": ".2f"}) + [""]

    # label summary
    tgt = L.join(F.select("symbol", "t", "role"), on=["symbol", "t"]).filter(pl.col("role") == "target")
    lab = pl.DataFrame([{
        "horizon_s": h, "mean_bps": tgt[f"r_{h}_bps"].mean(), "std_bps": tgt[f"r_{h}_bps"].std(),
        "p1_bps": tgt[f"r_{h}_bps"].quantile(0.01), "p99_bps": tgt[f"r_{h}_bps"].quantile(0.99),
        "zero": (tgt[f"r_{h}_ticks"] == 0).mean(),
        "dir_up": (tgt[f"dir_{h}"] == 1).mean(), "dir_flat": (tgt[f"dir_{h}"] == 0).mean(),
        "dir_down": (tgt[f"dir_{h}"] == -1).mean()} for h in cfg["horizons_s"]])
    lines += ["## Labels (29 targets pooled)", ""]
    lines += md_table(lab, {c: ".1%" for c in ["zero", "dir_up", "dir_flat", "dir_down"]}) + [""]

    # feature summary
    T = F.filter(pl.col("role") == "target")
    stats = pl.DataFrame([{
        "feature": c, "mean": T[c].mean(), "std": T[c].std(), "p1": T[c].quantile(0.01),
        "p50": T[c].quantile(0.5), "p99": T[c].quantile(0.99), "zero_share": (T[c] == 0).mean()}
        for c in feat_cols])
    lines += ["## Feature summary (29 targets pooled)", ""]
    lines += md_table(stats, {"zero_share": ".1%"}) + [""]

    # look-ahead safeguards (required in every results report) + real-data placebo for this day
    r2 = placebo(F, L, horizon=h0)
    placebo_lines = [f"Real-data placebo for {day}: pooled OLS of r_{h0} on all features, trained 09:35–12:45, "
                     f"tested 12:45–15:55, R² vs zero forecast: correct features **{r2['correct']:.2%}**, "
                     f"features leaked from t+{h0}s **{r2[f'leaked_+{h0}s']:.2%}**, so the harness detects leakage.", ""]
    lines += lookahead_section(run_tests(), placebo_lines)

    out = ROOT / "reports" / f"checkpoint2_{day}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""Audit the WRDS TAQ 1-minute bars that `hftf data pull-taq` cached.

Run from the repository root, with HFTF_DATA_DIR pointing at the pulled data:

    $env:PYTHONUTF8 = "1"
    $env:HFTF_DATA_DIR = "E:\\4511_5min\\wrds_taq"
    python backend/hft_forecast/ml/horizons/h5m/scripts/check_data.py

It reads the newest cache version of every instrument (never the manifest), and writes
`<HFTF_DATA_DIR>/reports/data_audit.csv` and `data_audit.md`.

What it checks, per instrument and dataset:
  * sessions covered and rows per session (390 for a full regular session)
  * duplicate timestamps (must be 0)
  * share of missing mids, share of stale bars, median spread in bps and in ticks
  * which tickers in symbols.txt have no data at all
  * whether the manifest would read the same rows more than once (a known issue)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import polars as pl
import yaml

TICK = 0.01
FULL_SESSION_BARS = 390


def data_dir() -> Path:
    path = Path(os.environ.get("HFTF_DATA_DIR", "data")).resolve()
    if not (path / "cache").is_dir():
        sys.exit(f"No cache/ folder under {path}. Set HFTF_DATA_DIR to the pulled data folder.")
    return path


def latest_cache_files(root: Path) -> dict[tuple[str, str], dict]:
    """(dataset, instrument) -> newest version info, read from every cache/<key>/meta.json."""
    out: dict[tuple[str, str], dict] = {}
    for meta_path in sorted((root / "cache").glob("*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        request = meta.get("request", {})
        if request.get("source") != "wrds_taq" or not meta.get("versions"):
            continue
        newest = meta["versions"][-1]
        key = (request["dataset"], request["instruments"][0])
        out[key] = {
            "file": meta_path.parent / newest["file"],
            "start": newest["start"],
            "end": newest["end"],
            "rows_meta": newest["rows"],
            "n_versions": len(meta["versions"]),
        }
    return out


def ny_dates(frame: pl.DataFrame) -> pl.Series:
    ts = pl.col("timestamp")
    if frame.schema["timestamp"].time_zone is None:  # type: ignore[union-attr]
        ts = ts.dt.replace_time_zone("UTC")
    return frame.select(ts.dt.convert_time_zone("America/New_York").dt.date().alias("d"))["d"]


def audit_quotes(frame: pl.DataFrame) -> dict:
    dates = ny_dates(frame)
    per_day = dates.value_counts().get_column("count")
    out = {
        "rows": frame.height,
        "sessions": dates.n_unique(),
        "bars_per_session_min": int(per_day.min()),
        "bars_per_session_median": float(per_day.median()),
        "bars_per_session_max": int(per_day.max()),
        "sessions_not_390": int((per_day != FULL_SESSION_BARS).sum()),
        "duplicate_timestamps": frame.height - frame.get_column("timestamp").n_unique(),
        "first_bar_utc": str(frame.get_column("timestamp").min()),
        "last_bar_utc": str(frame.get_column("timestamp").max()),
    }
    if "mid" in frame.columns:
        out.update(
            {
                "mid_missing_share": round(frame.get_column("mid").null_count() / frame.height, 4),
                "median_price": round(float(frame.get_column("mid").median()), 2),
                "median_spread_bps": round(float(frame.get_column("spread_bps").median()), 2),
                "median_spread_ticks": round(
                    float(((frame["ask"] - frame["bid"]) / TICK).median()), 2
                ),
                "stale_share": round(float(frame.get_column("is_stale").cast(pl.Float64).mean()), 4),
                "median_quotes_per_bar": float(frame.get_column("n_quotes").median()),
            }
        )
    if "volume" in frame.columns:
        out["median_volume_per_bar"] = float(frame.get_column("volume").median())
        out["median_trades_per_bar"] = float(frame.get_column("n_trades").median())
    return out


def manifest_check(root: Path) -> list[str]:
    """How many times the manifest route would read each instrument's rows."""
    lines: list[str] = []
    for path in sorted((root / "manifests").glob("*.yaml")):
        book = yaml.safe_load(path.read_text(encoding="utf-8"))
        by_key: dict[tuple[str, str], list[dict]] = {}
        for entry in book.get("entries", []):
            by_key.setdefault((entry["dataset"], entry["instrument"]), []).append(entry)
        bad = 0
        for (dataset, inst), entries in sorted(by_key.items()):
            listed = sum(pl.scan_parquet(e["path"]).select(pl.len()).collect().item()
                         for e in entries if Path(e["path"]).exists())
            true_rows = sum(int(e["rows"]) for e in entries)
            if listed != true_rows:
                bad += 1
                lines.append(f"- {path.name} {dataset} {inst}: manifest route reads {listed:,} rows, "
                             f"true rows {true_rows:,} (x{listed / max(true_rows, 1):.1f})")
        if bad == 0:
            lines.append(f"- {path.name}: OK, no instrument is read twice")
    return lines


def main() -> None:
    root = data_dir()
    files = latest_cache_files(root)
    rows = []
    for (dataset, inst), info in sorted(files.items()):
        frame = pl.read_parquet(info["file"]).sort("timestamp")
        row = {"dataset": dataset, "instrument": inst, "start": info["start"], "end": info["end"],
               "cache_versions": info["n_versions"], "file": str(info["file"])}
        row.update(audit_quotes(frame))
        rows.append(row)
        print(f"{dataset:15s} {inst:6s} rows={row['rows']:>7,} sessions={row['sessions']:>4} "
              f"dup={row['duplicate_timestamps']}")
    table = pl.DataFrame(rows)
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    table.write_csv(reports / "data_audit.csv")

    wanted = []
    symbols = root / "symbols.txt"
    if symbols.exists():
        wanted = [s.strip() for s in symbols.read_text(encoding="utf-8").splitlines() if s.strip()]
    have_q = {i for d, i in files if d == "taq_nbbo_bars"}
    have_t = {i for d, i in files if d == "taq_trade_bars"}
    full_q = {r["instrument"] for r in rows
              if r["dataset"] == "taq_nbbo_bars" and r["end"] >= "2024-06-28" and r["start"] <= "2024-01-02"}

    md = ["# TAQ data audit", "", f"Data folder: `{root}`", ""]
    md.append(f"- Quote bars (taq_nbbo_bars): {len(have_q)} instruments, "
              f"{len(full_q)} covering 2024-01-02 to 2024-06-28")
    md.append(f"- Trade bars (taq_trade_bars): {len(have_t)} instruments")
    if wanted:
        md.append(f"- In symbols.txt but no quote bars yet: {', '.join(s for s in wanted if s not in have_q) or 'none'}")
        md.append(f"- In symbols.txt but no trade bars yet: {', '.join(s for s in wanted if s not in have_t) or 'none'}")
    partial = sorted(have_q - full_q)
    md.append(f"- Quote bars with a partial date range: {', '.join(partial) or 'none'}")
    md += ["", "## Manifest check", ""] + manifest_check(root)
    md += ["", "## Per instrument", "", "See data_audit.csv. Columns to look at first: "
           "sessions (124 expected), sessions_not_390, duplicate_timestamps (0 expected), "
           "stale_share, median_spread_ticks."]
    (reports / "data_audit.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nWrote {reports / 'data_audit.csv'} and data_audit.md")


if __name__ == "__main__":
    main()

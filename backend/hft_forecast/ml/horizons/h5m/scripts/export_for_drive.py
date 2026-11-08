"""Make a clean, shareable copy of the pulled TAQ bars for Google Drive.

The cache keeps every intermediate version (v0001 ... v0006) under hashed folder names.
This copies only the newest, complete version of each instrument to

    <out>/taq_nbbo_bars/<TICKER>.parquet
    <out>/taq_trade_bars/<TICKER>.parquet
    <out>/manifest_sp50_2024h1_1min.yaml   (relative paths, one entry per file)
    <out>/symbols.txt, sp500_top50_20240102_selection.csv (if present)

Run from the repository root:

    $env:HFTF_DATA_DIR = "E:\\4511_5min\\wrds_taq"
    python backend/hft_forecast/ml/horizons/h5m/scripts/export_for_drive.py --out E:\\4511_5min\\drive_upload

Then drag the two taq_* folders and the yaml into the Drive folder IEOR4511_h5m_TAQ_data.
Nothing in the cache is changed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--start", default="2024-01-02")
    parser.add_argument("--end", default="2024-06-28")
    args = parser.parse_args()
    root = Path(os.environ.get("HFTF_DATA_DIR", "data")).resolve()
    out: Path = args.out.resolve()
    entries, skipped = [], []
    for meta_path in sorted((root / "cache").glob("*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        request = meta.get("request", {})
        if request.get("source") != "wrds_taq" or not meta.get("versions"):
            continue
        newest = meta["versions"][-1]
        dataset, ticker = request["dataset"], request["instruments"][0]
        if newest["start"] > args.start or newest["end"] < args.end:
            skipped.append(f"{dataset} {ticker} ({newest['start']} to {newest['end']})")
            continue
        target = out / dataset / f"{ticker}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(meta_path.parent / newest["file"], target)
        entries.append({"instrument": ticker, "dataset": dataset, "start": newest["start"],
                        "end": newest["end"], "rows": newest["rows"],
                        "path": f"{dataset}/{ticker}.parquet"})
    book = {"name": "sp50_2024h1_1min", "source": "wrds_taq", "frequency": "1min",
            "note": "One file per instrument and dataset; paths are relative to this file.",
            "entries": entries}
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest_sp50_2024h1_1min.yaml").write_text(
        yaml.safe_dump(book, sort_keys=False), encoding="utf-8")
    for name in ("symbols.txt", "sp500_top50_20240102_selection.csv"):
        if (root / name).exists():
            shutil.copy2(root / name, out / name)
    n_q = sum(e["dataset"] == "taq_nbbo_bars" for e in entries)
    n_t = sum(e["dataset"] == "taq_trade_bars" for e in entries)
    print(f"Copied {n_q} quote-bar files and {n_t} trade-bar files to {out}")
    if skipped:
        print("Skipped (incomplete date range, finish the pull first):")
        print("\n".join(f"  {s}" for s in skipped))


if __name__ == "__main__":
    main()

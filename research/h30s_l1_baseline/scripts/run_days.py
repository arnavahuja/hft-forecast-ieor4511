"""Run the per-day pipeline for the pilot range (or given dates): download -> clean -> sign -> features.

    python scripts/run_days.py --pilot-range
    python scripts/run_days.py --date 2025-03-03 --date 2025-03-04

Each stage skips work that is already done (existing raw files are not re-downloaded), so the
run can be stopped and restarted. Holidays (no TAQ tables) are skipped by the download stage.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from hft.config import ROOT, candidate_dates, load_config

PY = sys.executable


def run(*args) -> bool:
    r = subprocess.run([PY, "-W", "ignore", *args], cwd=ROOT, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip().splitlines()
    for line in out[-3:]:
        print("    " + line, flush=True)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", action="append", default=[])
    ap.add_argument("--pilot-range", action="store_true")
    ap.add_argument("--config", default="config/pilot.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dates = [pd.Timestamp(x) for x in args.date] + (candidate_dates(cfg) if args.pilot_range else [])
    data = Path(cfg["data_dir"])

    t_all = time.time()
    for d in dates:
        day = f"{d:%Y-%m-%d}"
        t0 = time.time()
        print(f"=== {day}", flush=True)
        if not run("scripts/download_day.py", "--date", day, "--group", "pilot", "--group", "benchmarks",
                   "--config", args.config):
            print(f"    FAILED download {day}", flush=True)
            continue
        if not (data / "universe" / f"date={day}" / "universe.parquet").exists():
            print(f"    {day}: not a trading day - skipped", flush=True)
            continue
        for stage in ["clean_day.py", "sign_day.py", "features_day.py"]:
            if not run(f"scripts/{stage}", "--date", day, "--config", args.config):
                print(f"    FAILED {stage} {day}", flush=True)
                break
        else:
            print(f"    {day} done in {time.time() - t0:.0f}s", flush=True)
    print(f"ALL DONE in {(time.time() - t_all) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()

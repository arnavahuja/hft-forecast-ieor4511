"""Load the YAML pipeline config and expand the trading-day list."""

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path="config/pilot.yaml") -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    cfg["data_dir"] = str(ROOT / cfg.get("data_dir", "data"))
    return cfg


def candidate_dates(cfg) -> list[pd.Timestamp]:
    """Weekdays in the pilot range; holidays are dropped later (no TAQ table that day)."""
    return list(pd.bdate_range(cfg["dates"]["start"], cfg["dates"]["end"]))

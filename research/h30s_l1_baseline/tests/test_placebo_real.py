"""Step 7.4 real-data placebo on the dev day (skipped if the dev-day panel hasn't been built)."""

from pathlib import Path

import polars as pl
import pytest

from hft.config import load_config
from hft.leakage import placebo

CFG = load_config()
PART = f"date={CFG['dev_date']}"
FEAT = Path(CFG["data_dir"]) / "features" / PART / "part.parquet"
LAB = Path(CFG["data_dir"]) / "labels" / PART / "part.parquet"


@pytest.mark.skipif(not (FEAT.exists() and LAB.exists()), reason="dev-day panel not built")
def test_leaked_features_are_detected():
    r2 = placebo(pl.read_parquet(FEAT), pl.read_parquet(LAB), horizon=30)
    assert r2["leaked_+30s"] > 0.5, r2      # leaking the future must look absurdly good
    assert r2["correct"] < 0.2, r2          # honest features: small R²

"""Golden tests: frozen results for each reference strategy on fixture data.

Any change to these numbers needs an explicit update::

    HFTF_UPDATE_GOLDEN=1 pytest tests/golden
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hft_forecast.analytics.service import AnalyticsService
from hft_forecast.core.registry import registry
from hft_forecast.core.run_config import PluginRef
from hft_forecast.engine.base import RunOptions
from hft_forecast.engine.pipeline import NamedOverlay, Pipeline
from hft_forecast.engine.vectorized import VectorizedEngine
from tests.helpers import PPY, config, synthetic

GOLDEN_DIR = Path(__file__).parent / "snapshots"
UPDATE = os.environ.get("HFTF_UPDATE_GOLDEN") == "1"
RTOL = 1e-8
DATA = synthetic(instruments=("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "MKT"))
KEYS = (
    "total_return",
    "cagr",
    "ann_vol",
    "sharpe",
    "max_drawdown",
    "turnover_ann",
    "n_trades",
    "total_commission",
)

CASES: dict[str, dict[str, Any]] = {
    "buy_and_hold": {"strategy": PluginRef(name="buy_and_hold")},
}

ALL_CASES = CASES


def _run(case: dict[str, Any]) -> dict[str, float | None]:
    strat_ref: PluginRef = case["strategy"]
    strategy = registry("strategy").get(strat_ref.name).create(strat_ref.params)
    constructor = None
    if "constructor" in case:
        ref = case["constructor"]
        constructor = registry("portfolio_constructor").get(ref.name).create(ref.params)
    overlays = tuple(
        NamedOverlay(r.name, registry("overlay").get(r.name).create(r.params))
        for r in case.get("overlays", [])
    )
    cost = registry("cost_model").get("bps_notional").create({"bps": 5})
    engine = VectorizedEngine([cost])
    data = DATA.select_instruments([i for i in DATA.instruments if i != "MKT"])
    result = engine.run(
        config(strat_ref.name, strat_ref.params),
        DATA,
        strategy,
        Pipeline(constructor, overlays),
        RunOptions(periods_per_year=PPY, benchmark_id="MKT", strategy_instruments=data.instruments),
    )
    metrics = AnalyticsService().compute(result)
    return {k: metrics[k].value for k in KEYS}


@pytest.mark.parametrize("name", sorted(ALL_CASES))
def test_golden(name: str):
    actual = _run(ALL_CASES[name])
    path = GOLDEN_DIR / f"{name}.json"
    if UPDATE or not path.exists():
        if not UPDATE:
            pytest.fail(f"No golden snapshot for {name}; run with HFTF_UPDATE_GOLDEN=1")
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(actual, indent=2, sort_keys=True))
        return
    expected = json.loads(path.read_text())
    for key in KEYS:
        exp, act = expected[key], actual[key]
        if exp is None or act is None:
            assert exp == act, key
        else:
            assert np.isclose(act, exp, rtol=RTOL, atol=1e-10), (key, act, exp)

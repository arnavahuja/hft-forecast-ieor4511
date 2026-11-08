"""Each rung of the h5m model ladder changes exactly one thing from the rung below it."""

from pathlib import Path

import pytest

from hft_forecast.ml.config import ExperimentConfig

H5M = Path(__file__).resolve().parents[3] / "backend" / "hft_forecast" / "ml" / "horizons" / "h5m"
META = {"name", "notes", "tags"}

# (lower rung, upper rung, the one field allowed to differ)
LADDER = [
    ("config_taq_lag1.yaml", "config_taq_flow.yaml", "features"),  # T3 -> T4
    ("config_taq_flow.yaml", "config_taq_flow_lgbm.yaml", "model"),  # T4 -> T5
]


@pytest.mark.parametrize(("lower", "upper", "changed"), LADDER)
def test_one_change_per_rung(lower: str, upper: str, changed: str) -> None:
    a = ExperimentConfig.load(H5M / lower)
    b = ExperimentConfig.load(H5M / upper)
    skip = META | {changed}
    assert a.model_dump(mode="json", exclude=skip) == b.model_dump(mode="json", exclude=skip)
    assert getattr(a, changed) != getattr(b, changed)
    assert b.labels.start_lag_bars == 1


def test_t4_and_t5_share_one_dataset() -> None:
    t4 = ExperimentConfig.load(H5M / "config_taq_flow.yaml")
    t5 = ExperimentConfig.load(H5M / "config_taq_flow_lgbm.yaml")
    assert t4.dataset_hash() == t5.dataset_hash()
    assert [f.name for f in t4.features] == ["basic", "h30s_flow"]
    assert t4.features[1].params["ofi_windows"] == [1, 5, 15, 30]

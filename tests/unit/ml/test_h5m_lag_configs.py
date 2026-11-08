"""The lagged-label configs differ from their lag-0 twin only in the label start lag."""

from pathlib import Path

import pytest

from hft_forecast.ml.config import ExperimentConfig

H5M = Path(__file__).resolve().parents[3] / "backend" / "hft_forecast" / "ml" / "horizons" / "h5m"
IGNORED = {"name", "notes", "tags", "labels"}

PAIRS = [("config_taq.yaml", "config_taq_lag1.yaml", 1)]


@pytest.mark.parametrize(("base_file", "lag_file", "lag"), PAIRS)
def test_only_the_label_lag_differs(base_file: str, lag_file: str, lag: int) -> None:
    base = ExperimentConfig.load(H5M / base_file)
    lagged = ExperimentConfig.load(H5M / lag_file)

    assert base.model_dump(mode="json", exclude=IGNORED) == lagged.model_dump(
        mode="json", exclude=IGNORED
    )
    base_labels = base.labels.model_dump()
    lag_labels = lagged.labels.model_dump()
    assert base_labels.pop("start_lag_bars") == 0
    assert lag_labels.pop("start_lag_bars") == lag
    assert base_labels == lag_labels
    assert base.dataset_hash() != lagged.dataset_hash()  # a new dataset with the new label

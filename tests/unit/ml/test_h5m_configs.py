"""The h5m baseline configs stay aligned on one reproducible dataset."""

from pathlib import Path

from hft_forecast.ml.config import ExperimentConfig

H5M_CONFIG_DIR = (
    Path(__file__).resolve().parents[3] / "backend" / "hft_forecast" / "ml" / "horizons" / "h5m"
)


def test_h5m_baseline_configs_share_data_and_features() -> None:
    ridge = ExperimentConfig.load(H5M_CONFIG_DIR / "config.yaml")
    last_sign = ExperimentConfig.load(H5M_CONFIG_DIR / "config_last_sign5.yaml")

    assert ridge.experiment == last_sign.experiment == "h5m"
    assert ridge.horizon.seconds == last_sign.horizon.seconds == 300
    assert ridge.horizon.grid == last_sign.horizon.grid
    assert ridge.manifest is last_sign.manifest is None
    assert ridge.data.source == last_sign.data.source == "synthetic"
    assert ridge.dataset_hash() == last_sign.dataset_hash()
    assert ridge.features[0].params["vol_windows"] == [15, 30]
    assert last_sign.features[0].params["vol_windows"] == [15, 30]
    assert ridge.model.name == "logit_ridge"
    assert last_sign.model.name == "last_sign"
    assert last_sign.model.params["column"] == "f_logret_5"
    assert ridge.cv.min_train_rows == last_sign.cv.min_train_rows == 1000

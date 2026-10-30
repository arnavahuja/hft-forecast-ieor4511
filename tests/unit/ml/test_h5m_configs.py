"""The h5m baseline configs stay aligned on one reproducible dataset."""

from pathlib import Path

from hft_forecast.ml.config import ExperimentConfig

H5M_CONFIG_DIR = (
    Path(__file__).resolve().parents[3] / "backend" / "hft_forecast" / "ml" / "horizons" / "h5m"
)


def test_h5m_synthetic_baseline_configs_share_data_and_features() -> None:
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


def test_h5m_taq_config_uses_fixed_manifest_and_point_in_time_universe() -> None:
    taq = ExperimentConfig.load(H5M_CONFIG_DIR / "config_taq.yaml")

    assert taq.experiment == "h5m"
    assert taq.manifest == "sp50_2024h1_1min"
    assert taq.data.source == "wrds_taq"
    assert taq.data.dataset == "taq_nbbo_bars"
    assert taq.data.frequency.value == "1min"
    assert taq.data.start.isoformat() == "2024-01-02"
    assert taq.data.end.isoformat() == "2024-06-28"
    assert len(taq.data.instruments) == 50
    assert len(set(taq.data.instruments)) == 50
    assert taq.features[0].params["vol_windows"] == [15, 30]
    assert taq.cv.min_train_rows == 20_000

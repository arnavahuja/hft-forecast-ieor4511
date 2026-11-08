"""T7 = T6 + the h5m_seq feature set + the GRU model; nothing else changes."""

from pathlib import Path

from hft_forecast.ml.config import ExperimentConfig

H5M = Path(__file__).resolve().parents[3] / "backend" / "hft_forecast" / "ml" / "horizons" / "h5m"


def test_t7_extends_t6_by_the_sequence_only() -> None:
    t6 = ExperimentConfig.load(H5M / "config_taq_flow_mlp.yaml")
    t7 = ExperimentConfig.load(H5M / "config_taq_flow_gru.yaml")
    skip = {"name", "notes", "tags", "features", "model"}
    assert t6.model_dump(mode="json", exclude=skip) == t7.model_dump(mode="json", exclude=skip)
    assert t7.features[:-1] == t6.features
    assert t7.features[-1].name == "h5m_seq" and t7.features[-1].params["length"] == 20
    assert t6.model.name == "h5m_mlp" and t7.model.name == "h5m_gru"
    assert t7.labels.start_lag_bars == 1

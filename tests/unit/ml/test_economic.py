"""The economic test configs and the prediction_signal strategy that executes them."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from hft_forecast.core import columns as C
from hft_forecast.core.errors import ConfigError
from hft_forecast.core.registry import registry
from hft_forecast.core.types import TargetKind
from hft_forecast.ml.economic import (
    COST_VARIANTS,
    PRIMARY_VARIANT,
    RETURN_COLUMN,
    cross_sectional_config,
    per_name_configs,
    pooled_returns,
)


@pytest.fixture
def predictions(tmp_path, quote_bars):
    """Deterministic predictions: long whenever the previous bar rose."""
    frame = quote_bars.select(C.TIMESTAMP, C.INSTRUMENT, C.MID, "session_id").head(600)
    up = (frame[C.MID].diff().fill_null(0.0) > 0).cast(pl.Float64)
    path = tmp_path / "predictions.parquet"
    frame.select(C.TIMESTAMP, C.INSTRUMENT, "session_id").with_columns(
        pl.lit(0, dtype=pl.Int32).alias("fold"),
        pl.lit("logit_ridge@1.0.0").alias("model"),
        (1.0 - up).alias("p_down"),
        pl.lit(0.0).alias("p_flat"),
        up.alias("p_up"),
        ((up - 0.5) * 1e-4).alias("y_hat_logret"),
    ).write_parquet(path)
    return path


def _strategy(path, **params):
    return (
        registry("strategy")
        .get("prediction_signal")
        .create({"predictions_path": str(path), **params})
    )


def test_signals_become_weights_on_the_engine_grid(predictions, quote_bars, experiment):
    from hft_forecast.core.types import MarketData

    data = MarketData(quote_bars, experiment.data.frequency)
    targets = _strategy(predictions, hold_bars=1).generate_targets(data)
    assert targets.kind is TargetKind.WEIGHTS
    assert targets.values.shape == (len(data.timestamps), len(data.instruments))
    assert set(np.unique(targets.values)) <= {-1.0, 0.0, 1.0}
    assert np.abs(targets.values).sum() > 0
    assert not targets.values[600:].any()  # no prediction, no position


def test_overlapping_signals_are_averaged(predictions, quote_bars, experiment):
    from hft_forecast.core.types import MarketData

    data = MarketData(quote_bars, experiment.data.frequency)
    averaged = _strategy(predictions, hold_bars=6, overlap="average").generate_targets(data)
    replaced = _strategy(predictions, hold_bars=6, overlap="replace").generate_targets(data)
    assert np.abs(averaged.values).max() <= 1.0
    assert len(np.unique(averaged.values)) > len(np.unique(replaced.values))


def test_a_missing_or_ambiguous_file_is_a_config_error(tmp_path, quote_bars, experiment):
    from hft_forecast.core.types import MarketData

    data = MarketData(quote_bars, experiment.data.frequency)
    with pytest.raises(ConfigError, match="No predictions file"):
        _strategy(tmp_path / "nope.parquet").generate_targets(data)


def test_duplicate_keys_are_refused(predictions, quote_bars, experiment):
    from hft_forecast.core.types import MarketData

    doubled = pl.concat([pl.read_parquet(predictions)] * 2)
    doubled.write_parquet(predictions)
    data = MarketData(quote_bars, experiment.data.frequency)
    with pytest.raises(ConfigError, match="Duplicate"):
        _strategy(predictions).generate_targets(data)


def test_per_name_configs_are_one_instrument_each(experiment, predictions):
    configs = per_name_configs(experiment, predictions, "logit_ridge@1.0.0")
    assert [c.data.instruments for c in configs] == [["AAA"], ["BBB"]]
    assert all(c.execution.lag_bars == 1 for c in configs)
    assert all(c.slippage is not None and c.slippage.name == "quoted_spread" for c in configs)
    assert all(c.experiment == "h30s" for c in configs)
    free = per_name_configs(experiment, predictions, "logit_ridge@1.0.0", variant="none")
    assert free[0].slippage is None


def test_the_cross_sectional_config_uses_a_constructor(experiment, predictions):
    config = cross_sectional_config(experiment, predictions, "logit_ridge@1.0.0")
    assert config.constructor is not None and config.constructor.name == "quantile"
    assert config.strategy.params["output_kind"] == "signals"
    assert PRIMARY_VARIANT in COST_VARIANTS


def test_an_unknown_variant_is_rejected(experiment, predictions):
    with pytest.raises(KeyError, match="Unknown cost variant"):
        per_name_configs(experiment, predictions, "m", variant="free_lunch")


def test_pooled_returns_average_the_books():
    stamps = np.array(["2024-07-01T14:00", "2024-07-01T14:05"], dtype="datetime64[us]")
    books = {
        "AAA": pl.DataFrame({C.TIMESTAMP: stamps, RETURN_COLUMN: [0.01, -0.02]}),
        "BBB": pl.DataFrame({C.TIMESTAMP: stamps, RETURN_COLUMN: [0.03, 0.00]}),
    }
    pooled = pooled_returns(books)
    assert pooled["pooled_return"].to_list() == [0.02, -0.01]
    assert pooled_returns({}).height == 0


def test_the_shipped_example_configs_are_valid():
    import yaml

    from hft_forecast.core.run_config import BacktestConfig

    for name in ("econ_test_per_name.yaml", "econ_test_cross_sectional.yaml"):
        path = Path("configs/examples") / name
        config = BacktestConfig.model_validate(yaml.safe_load(path.read_text()))
        assert config.strategy.name == "prediction_signal"
        assert config.slippage is not None and config.slippage.name == "quoted_spread"


def test_prediction_signal_has_no_lookahead(predictions, quote_bars, experiment):
    from hft_forecast.core.types import MarketData
    from hft_forecast.engine.lookahead import check_lookahead

    data = MarketData(quote_bars.head(600), experiment.data.frequency)
    report = check_lookahead(_strategy(predictions, hold_bars=6), data)
    assert report.passed, report.mismatches

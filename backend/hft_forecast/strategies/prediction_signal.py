"""Turn out-of-sample predictions into positions (§7.1).

The strategy reads a ``predictions.parquet`` written by ``hftf ml run``; it never imports the
ml package, so a group can run the economic test on predictions produced anywhere. Rows are
out-of-sample by construction (the file only holds test-fold rows), and the strategy refuses
a file with duplicate keys rather than silently taking one of them.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Final

import numpy as np
import polars as pl
from pydantic import Field

from hft_forecast.core import columns as C
from hft_forecast.core.errors import ConfigError
from hft_forecast.core.interfaces import (
    ASSET_EQUITY,
    ENGINE_VECTORIZED,
    FREQ_DAILY,
    FREQ_INTRADAY,
    FREQ_SUBMINUTE,
    SUPPORTS_SHORT,
    DataRequirements,
    Strategy,
)
from hft_forecast.core.params import StrategyParams
from hft_forecast.core.registry import register
from hft_forecast.core.types import FloatArray, MarketData, TargetFrame, TargetKind

PROBABILITY: Final = "probability"
RETURN: Final = "return"
REPLACE: Final = "replace"
AVERAGE: Final = "average"
BPS: Final = 1e4


class PredictionSignalParams(StrategyParams):
    """Where the predictions are and how they become positions."""

    predictions_path: str = Field(
        "", description="Path to a predictions.parquet written by `hftf ml run`"
    )
    model: str = Field("", description="Model label to use; defaults to the only one present")
    mode: str = Field(PROBABILITY, pattern=f"^({PROBABILITY}|{RETURN})$")
    threshold: float = Field(
        0.0, ge=0, description="Probability mode: trade when max(p_up, p_down) - p_flat exceeds it"
    )
    spread_multiple: float = Field(
        1.0, ge=0, description="Return mode: trade when |y_hat| exceeds k x the half spread"
    )
    hold_bars: int = Field(1, ge=1, description="Bars a signal stays on")
    overlap: str = Field(
        AVERAGE,
        pattern=f"^({REPLACE}|{AVERAGE})$",
        description="A new signal replaces the position, or positions average over hold_bars",
    )
    output_kind: str = Field(
        "weights", pattern="^(weights|signals)$", description="signals feed a constructor"
    )
    max_weight: float = Field(1.0, gt=0, le=10, description="Weight of a full-strength signal")


@register(
    "strategy",
    name="prediction_signal",
    version="1.0.0",
    tags=["ml", "intraday"],
)
class PredictionSignal(Strategy):
    """Positions from a predictions file: long on up, short on down, flat otherwise."""

    Params = PredictionSignalParams
    params: PredictionSignalParams
    capabilities: ClassVar[frozenset[str]] = frozenset(
        {
            ENGINE_VECTORIZED,
            ASSET_EQUITY,
            SUPPORTS_SHORT,
            FREQ_DAILY,
            FREQ_INTRADAY,
            FREQ_SUBMINUTE,
        }
    )

    def requirements(self) -> DataRequirements:
        """Needs the mid (as close) and, in return mode, the quoted spread."""
        fields = (C.CLOSE,) if self.params.mode == PROBABILITY else (C.CLOSE, C.SPREAD_BPS)
        return DataRequirements(fields=fields)

    def generate_targets(self, data: MarketData) -> TargetFrame:
        """Align the predictions to the bars and hold each signal for ``hold_bars``.

        Raises:
            ConfigError: If the file is missing, holds duplicate keys, or covers none of the
                requested instruments.
        """
        frame = self._load()
        signal = self._signal(frame, data)
        held = _hold(signal, self.params.hold_bars, self.params.overlap)
        kind = TargetKind.WEIGHTS if self.params.output_kind == "weights" else TargetKind.SIGNALS
        values = held * self.params.max_weight if kind is TargetKind.WEIGHTS else held
        return TargetFrame(data.timestamps, data.instruments, values, kind)

    def _load(self) -> pl.DataFrame:
        if not self.params.predictions_path:
            raise ConfigError("prediction_signal needs 'predictions_path'")
        path = Path(self.params.predictions_path)
        if not path.exists():
            raise ConfigError(f"No predictions file at {path}")
        frame = pl.read_parquet(path)
        models = frame.get_column("model").unique().to_list()
        wanted = self.params.model or (models[0] if len(models) == 1 else "")
        if not wanted:
            raise ConfigError(
                "predictions hold several models; set 'model'", details={"models": models}
            )
        frame = frame.filter(pl.col("model") == wanted)
        if frame.is_empty():
            raise ConfigError(f"No rows for model '{wanted}'", details={"models": models})
        keys = [C.TIMESTAMP, C.INSTRUMENT]
        if frame.select(keys).is_duplicated().any():
            raise ConfigError(f"Duplicate (timestamp, instrument) rows in {path}")
        return frame

    def _signal(self, frame: pl.DataFrame, data: MarketData) -> FloatArray:
        """``(T, N)`` of -1 / 0 / +1 (probability mode) or scaled ``y_hat`` (return mode)."""
        if self.params.mode == PROBABILITY:
            edge = (pl.max_horizontal(pl.col("p_up"), pl.col("p_down")) - pl.col("p_flat")).alias(
                "_edge"
            )
            direction = (
                pl.when(pl.col("_edge") <= self.params.threshold)
                .then(0.0)
                .when(pl.col("p_up") >= pl.col("p_down"))
                .then(1.0)
                .otherwise(-1.0)
            )
            values = frame.with_columns(edge).with_columns(direction.alias("_signal"))
        else:
            values = frame.with_columns(pl.col("y_hat_logret").alias("_signal"))
        aligned = _align(values, data)
        if self.params.mode == RETURN:
            half_spread = _panel_or_nan(data, C.SPREAD_BPS) / 2.0 / BPS
            threshold = self.params.spread_multiple * np.nan_to_num(half_spread)
            aligned = np.where(np.abs(aligned) > threshold, np.sign(aligned), 0.0)
        return aligned


def _align(frame: pl.DataFrame, data: MarketData) -> FloatArray:
    """Predictions pivoted onto the engine's ``(T, N)`` grid; missing bars are flat."""
    index = {t: i for i, t in enumerate(data.timestamps.astype("datetime64[us]").tolist())}
    columns = {name: j for j, name in enumerate(data.instruments)}
    out = np.zeros((len(data.timestamps), len(data.instruments)))
    stamps = frame.get_column(C.TIMESTAMP).to_numpy().astype("datetime64[us]").tolist()
    names = frame.get_column(C.INSTRUMENT).to_list()
    values = frame.get_column("_signal").to_numpy()
    for stamp, name, value in zip(stamps, names, values, strict=True):
        row, column = index.get(stamp), columns.get(name)
        if row is not None and column is not None:
            out[row, column] = value
    return out


def _panel_or_nan(data: MarketData, field_name: str) -> FloatArray:
    if field_name in data.fields:
        return data.panel(field_name)
    return np.full((len(data.timestamps), len(data.instruments)), np.nan)


def _hold(signal: FloatArray, bars: int, overlap: str) -> FloatArray:
    """Keep each signal on for ``bars`` bars.

    ``replace`` lets the newest signal win, which is what a single-position book does.
    ``average`` holds ``bars`` overlapping positions of ``1/bars`` each: the standard way to
    use overlapping forecasts without peeking, and it is what the labels imply.
    """
    if bars == 1:
        return signal
    stacked = np.zeros_like(signal)
    for lag in range(bars):
        shifted = np.zeros_like(signal)
        shifted[lag:] = signal[: len(signal) - lag] if lag else signal
        if overlap == AVERAGE:
            stacked += shifted / bars
        else:
            stacked = np.where(shifted != 0.0, shifted, stacked)
    return stacked

"""The frozen interface: horizons, the prediction schema and the two plugin protocols.

Everything three groups share lives here. Changing anything in this module needs a PR
reviewed by one person from each horizon group (§8.1 of the design).
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Final, Generic, TypeVar

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from hft_forecast.core import columns as C
from hft_forecast.core.errors import ConfigError, PluginError, PluginNotFoundError
from hft_forecast.core.params import PluginParams
from hft_forecast.core.types import FloatArray, Frequency, IntArray

T = TypeVar("T")

SESSION_ID: Final = "session_id"
SECONDS_SINCE_OPEN: Final = "seconds_since_open"
SECONDS_TO_CLOSE: Final = "seconds_to_close"
"""Columns ``ml.datasets`` adds to every bar frame."""

FWD_LOGRET: Final = "fwd_logret"
DIRECTION: Final = "direction"
MAGNITUDE_BPS: Final = "magnitude_bps"
LABEL_VALID: Final = "label_valid"
LABEL_END: Final = "label_end_timestamp"
LABEL_COLUMNS: Final = (FWD_LOGRET, DIRECTION, MAGNITUDE_BPS, LABEL_VALID, LABEL_END)

FEATURE_PREFIX: Final = "f_"
BPS: Final = 1e4
N_CLASSES: Final = 3
"""Direction classes: down (-1), flat (0), up (+1), stored as 0, 1, 2 in the probabilities."""

PREDICTION_COLUMNS: Final = (
    C.TIMESTAMP,
    C.INSTRUMENT,
    SESSION_ID,
    "fold",
    "model",
    "p_down",
    "p_flat",
    "p_up",
    "y_hat_logret",
    FWD_LOGRET,
    DIRECTION,
    MAGNITUDE_BPS,
    LABEL_END,
)
"""Column order of ``predictions.parquet``. The true labels travel with the predictions so
evaluation never has to join back to the dataset."""


class Prediction(BaseModel):
    """One out-of-sample prediction row (the schema of ``predictions.parquet``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    p_down: float = Field(ge=0, le=1)
    p_flat: float = Field(ge=0, le=1)
    p_up: float = Field(ge=0, le=1)
    y_hat_logret: float
    fold: int = Field(ge=0)
    model: str
    """Registry name and version, e.g. ``logit_ridge@1.0.0``."""


class HorizonSpec(BaseModel):
    """A forecast horizon and the bar grid it is measured on."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    seconds: int = Field(gt=0, description="Forecast horizon in seconds")
    grid: Frequency = Frequency.S5

    @model_validator(mode="after")
    def _whole_steps(self) -> HorizonSpec:
        bar = self.grid.seconds
        if bar is None:
            raise ValueError(f"{self.grid} is not an intraday frequency")
        if self.seconds % bar:
            raise ValueError(f"horizon {self.seconds}s is not a whole number of {bar}s bars")
        return self

    @property
    def steps(self) -> int:
        """Bars between the decision bar and the target bar."""
        bar = self.grid.seconds
        assert bar is not None  # guaranteed by the validator
        return self.seconds // bar

    @property
    def delta(self) -> np.timedelta64:
        """One bar as a timedelta."""
        bar = self.grid.seconds
        assert bar is not None
        return np.timedelta64(bar, "s")


H30S: Final = HorizonSpec(name="h30s", seconds=30, grid=Frequency.S5)
H5M: Final = HorizonSpec(name="h5m", seconds=300, grid=Frequency.MIN1)
H30M: Final = HorizonSpec(name="h30m", seconds=1800, grid=Frequency.MIN1)
HORIZONS: Final[dict[str, HorizonSpec]] = {h.name: h for h in (H30S, H5M, H30M)}
"""The three groups' horizons. A group may change its own ``grid`` in its config."""


class FeatureSet(ABC):
    """Turns a bar frame into ``f_*`` columns, without ever looking forward.

    Implementations must build every column from ``.shift(k >= 0)`` or ``.rolling_*`` windows
    taken ``.over`` the session, so a feature can never see a later bar or the previous
    session's close. :func:`~hft_forecast.ml.features.registry.check_feature_lookahead`
    enforces this for every registered set.
    """

    Params: ClassVar[type[PluginParams]] = PluginParams
    lookback_bars: ClassVar[int] = 0
    requires: ClassVar[tuple[str, ...]] = (C.MID,)

    def __init__(self, params: PluginParams | None = None) -> None:
        self.params = params or self.Params()

    @abstractmethod
    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Return ``frame`` with ``f_*`` columns added and the same rows in the same order."""


class Forecaster(ABC):
    """Arrays in, arrays out: the only thing a model has to be.

    A model never loads data, splits folds or scores itself; that is what lets three groups
    add models in parallel without touching shared code.
    """

    Params: ClassVar[type[PluginParams]] = PluginParams
    supports_sample_weight: ClassVar[bool] = True

    def __init__(self, params: PluginParams | None = None) -> None:
        self.params = params or self.Params()

    @abstractmethod
    def fit(
        self,
        x: FloatArray,
        y_dir: IntArray,
        y_ret: FloatArray,
        sample_weight: FloatArray | None,
        seed: int,
    ) -> None:
        """Fit on one training fold. ``y_dir`` is in ``{-1, 0, 1}``."""

    @abstractmethod
    def predict(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return ``(probabilities (n, 3) ordered down/flat/up, expected log return (n,))``."""

    def state_dict(self) -> dict[str, Any]:
        """Everything needed to restore the fitted model (baselines pickle, torch saves tensors)."""
        return {}


@dataclass(frozen=True)
class MlSpec(Generic[T]):
    """A registered model or feature set."""

    kind: str
    name: str
    version: str
    description: str
    cls: type[T]
    module: str

    @property
    def params_model(self) -> type[PluginParams]:
        """The plugin's pydantic ``Params`` model."""
        model: type[PluginParams] = getattr(self.cls, "Params", PluginParams)
        return model

    def create(self, params: dict[str, Any] | PluginParams | None = None) -> T:
        """Instantiate with validated parameters."""
        validated = (
            params
            if isinstance(params, PluginParams)
            else self.params_model.model_validate(params or {})
        )
        return self.cls(validated)  # type: ignore[call-arg]  # plugins take (params)

    @property
    def label(self) -> str:
        """``name@version``, as stored in the predictions."""
        return f"{self.name}@{self.version}"


class MlRegistry(Generic[T]):
    """One registry per ML plugin kind, with the mechanics of :mod:`core.registry`.

    Separate from the plugin registries in ``core`` on purpose: models and feature sets are
    not backtest plugins and never show up in the strategy UI.
    """

    def __init__(self, kind: str, base: type[Any]) -> None:
        self.kind = kind
        self._base = base
        self._specs: dict[str, MlSpec[T]] = {}

    def decorator(self, *, name: str, version: str = "1.0.0") -> Callable[[type[T]], type[T]]:
        """Class decorator registering the class under ``name``.

        Raises:
            PluginError: If the class does not subclass the registry's base, or the name is
                already taken by a different class.
        """

        def wrap(cls: type[T]) -> type[T]:
            if not (inspect.isclass(cls) and issubclass(cls, self._base)):
                raise PluginError(f"{self.kind} '{name}' must subclass {self._base.__name__}")
            existing = self._specs.get(name)
            if existing is not None and existing.cls.__qualname__ != cls.__qualname__:
                raise PluginError(
                    f"Duplicate {self.kind} '{name}' in '{existing.module}' and '{cls.__module__}'"
                )
            doc = inspect.cleandoc(cls.__doc__ or "").split("\n\n")[0].replace("\n", " ")
            self._specs[name] = MlSpec(self.kind, name, version, doc, cls, cls.__module__)
            return cls

        return wrap

    def get(self, name: str) -> MlSpec[T]:
        """Look up by name.

        Raises:
            PluginNotFoundError: If nothing is registered under ``name``.
        """
        try:
            return self._specs[name]
        except KeyError:
            raise PluginNotFoundError(
                f"No {self.kind} named '{name}'",
                details={"available": sorted(self._specs)},
            ) from None

    def specs(self) -> list[MlSpec[T]]:
        """All specs sorted by name."""
        return [self._specs[k] for k in sorted(self._specs)]

    def names(self) -> list[str]:
        """Sorted names."""
        return sorted(self._specs)

    def __contains__(self, name: object) -> bool:
        return name in self._specs


def require_columns(frame: pl.DataFrame, columns: tuple[str, ...], who: str) -> None:
    """Check that a frame carries the columns a step needs.

    Raises:
        ConfigError: If any are missing.
    """
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ConfigError(f"{who} needs columns {missing}", details={"have": frame.columns})

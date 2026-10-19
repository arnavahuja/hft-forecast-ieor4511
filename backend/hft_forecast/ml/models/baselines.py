"""The three baselines every group's model has to beat.

They are deliberately tiny. ``zero`` is the floor, ``last_sign`` is the naive persistence
answer, and ``logit_ridge`` is the linear model that usually turns out to be hard to beat on
30-second horizons.
"""

from __future__ import annotations

from typing import Any, ClassVar, Final

import numpy as np
from pydantic import Field

from hft_forecast.core.errors import ConfigError
from hft_forecast.core.params import PluginParams
from hft_forecast.core.types import FloatArray, IntArray
from hft_forecast.ml.base import N_CLASSES, Forecaster
from hft_forecast.ml.models.registry import register_model

CLASSES: Final = (-1, 0, 1)
EPS: Final = 1e-12


def _priors(y_dir: IntArray, sample_weight: FloatArray | None) -> FloatArray:
    """Weighted class frequencies in ``down, flat, up`` order."""
    weight = np.ones(len(y_dir)) if sample_weight is None else sample_weight
    total = weight.sum()
    if total <= 0:
        return np.full(N_CLASSES, 1.0 / N_CLASSES)
    out = np.array([weight[y_dir == c].sum() for c in CLASSES]) / total
    return out


class ZeroParams(PluginParams):
    """No parameters."""


@register_model(name="zero", version="1.0.0")
class Zero(Forecaster):
    """Predicts the training class priors and a zero return: the floor for every metric."""

    Params = ZeroParams

    def __init__(self, params: PluginParams | None = None) -> None:
        super().__init__(params)
        self._priors = np.full(N_CLASSES, 1.0 / N_CLASSES)

    def fit(
        self,
        x: FloatArray,
        y_dir: IntArray,
        y_ret: FloatArray,
        sample_weight: FloatArray | None,
        seed: int,
    ) -> None:
        """Count the classes."""
        self._priors = _priors(y_dir, sample_weight)

    def predict(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        """The same priors for every row."""
        return np.tile(self._priors, (len(x), 1)), np.zeros(len(x))

    def state_dict(self) -> dict[str, Any]:
        """The priors."""
        return {"priors": self._priors.tolist()}


class LastSignParams(PluginParams):
    """Which feature carries the momentum signal."""

    column: str = Field(
        "f_logret_10",
        description="Feature used as the signal; set it to your horizon in bars",
    )
    scale: float = Field(1.0, gt=0, description="Shrinkage applied to the predicted return")


@register_model(name="last_sign", version="1.0.0")
class LastSign(Forecaster):
    """Momentum: the direction of the last multi-bar return, carried forward."""

    Params = LastSignParams
    params: LastSignParams
    supports_sample_weight: ClassVar[bool] = False

    def __init__(self, params: PluginParams | None = None) -> None:
        super().__init__(params)
        self._column = -1
        self._tau = 0.0

    def fit(
        self,
        x: FloatArray,
        y_dir: IntArray,
        y_ret: FloatArray,
        sample_weight: FloatArray | None,
        seed: int,
    ) -> None:
        """Locate the signal column and take the dead zone from the training labels.

        Raises:
            ConfigError: If the configured feature is not in the design matrix.
        """
        if self.params.column not in self.feature_names:
            raise ConfigError(
                f"last_sign needs feature '{self.params.column}'",
                details={"available": list(self.feature_names)},
            )
        self._column = self.feature_names.index(self.params.column)
        flat = y_ret[y_dir == 0]
        self._tau = float(np.abs(flat).max()) if flat.size else 0.0

    def predict(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Sign of the signal, with the training dead zone reused as the flat band."""
        signal = np.nan_to_num(x[:, self._column])
        probabilities = np.zeros((len(x), N_CLASSES))
        probabilities[signal < -self._tau, 0] = 1.0
        probabilities[np.abs(signal) <= self._tau, 1] = 1.0
        probabilities[signal > self._tau, 2] = 1.0
        return probabilities, signal * self.params.scale

    def state_dict(self) -> dict[str, Any]:
        """Column index and dead zone."""
        return {"column": self._column, "tau": self._tau}


class LogitRidgeParams(PluginParams):
    """Regularisation of the two linear heads."""

    c: float = Field(1.0, gt=0, description="Inverse L2 strength of the logistic head")
    alpha: float = Field(1.0, ge=0, description="L2 strength of the ridge head")
    max_iter: int = Field(200, ge=10, le=10_000)


@register_model(name="logit_ridge", version="1.0.0")
class LogitRidge(Forecaster):
    """Standardised logistic regression for the direction and ridge for the return."""

    Params = LogitRidgeParams
    params: LogitRidgeParams

    def __init__(self, params: PluginParams | None = None) -> None:
        super().__init__(params)
        self._classifier: Any = None
        self._regressor: Any = None
        self._classes: IntArray = np.array(CLASSES)

    def fit(
        self,
        x: FloatArray,
        y_dir: IntArray,
        y_ret: FloatArray,
        sample_weight: FloatArray | None,
        seed: int,
    ) -> None:
        """Fit both heads on the same rows; scaling happens inside the pipeline."""
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        p = self.params
        self._classifier = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "logit",
                    LogisticRegression(C=p.c, max_iter=p.max_iter, random_state=seed),
                ),
            ]
        )
        self._regressor = Pipeline(
            [("scale", StandardScaler()), ("ridge", Ridge(alpha=p.alpha, random_state=seed))]
        )
        clean = np.nan_to_num(x)
        kwargs = {} if sample_weight is None else {"sample_weight": sample_weight}
        self._classifier.fit(clean, y_dir, **{f"logit__{k}": v for k, v in kwargs.items()})
        self._regressor.fit(clean, y_ret, **{f"ridge__{k}": v for k, v in kwargs.items()})
        self._classes = np.asarray(self._classifier.named_steps["logit"].classes_)

    def predict(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Probabilities padded back to three classes when a fold saw only two.

        Raises:
            ConfigError: If called before ``fit``.
        """
        if self._classifier is None:
            raise ConfigError("logit_ridge must be fitted before predicting")
        clean = np.nan_to_num(x)
        fitted = self._classifier.predict_proba(clean)
        out = np.zeros((len(x), N_CLASSES))
        for column, label in enumerate(self._classes):
            out[:, CLASSES.index(int(label))] = fitted[:, column]
        predicted: FloatArray = self._regressor.predict(clean)
        return out, predicted

    def state_dict(self) -> dict[str, Any]:
        """The fitted pipelines (pickled by the caller)."""
        return {"classifier": self._classifier, "regressor": self._regressor}

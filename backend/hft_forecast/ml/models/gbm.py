"""Gradient boosting. Stub: the strong tabular baseline, left for the groups."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from hft_forecast.core.params import PluginParams
from hft_forecast.core.types import FloatArray, IntArray
from hft_forecast.ml.base import Forecaster
from hft_forecast.ml.models.registry import register_model

MESSAGE = (
    "gbm is a stub: install lightgbm, then fit one booster for the direction "
    "(objective='multiclass', 3 classes) and one for the signed return (objective='l2'), "
    "passing sample_weight through and seeding with `seed`."
)


class GbmParams(PluginParams):
    """The usual handful of boosting knobs."""

    n_estimators: int = Field(300, ge=1, le=10_000)
    learning_rate: float = Field(0.05, gt=0, le=1)
    num_leaves: int = Field(31, ge=2, le=4096)
    min_child_samples: int = Field(100, ge=1)


@register_model(name="gbm", version="0.1.0")
class Gbm(Forecaster):
    """Gradient-boosted trees (not implemented)."""

    Params = GbmParams
    params: GbmParams
    implemented: ClassVar[bool] = False

    def fit(
        self,
        x: FloatArray,
        y_dir: IntArray,
        y_ret: FloatArray,
        sample_weight: FloatArray | None,
        seed: int,
    ) -> None:
        """Not implemented.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(MESSAGE)

    def predict(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Not implemented.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(MESSAGE)

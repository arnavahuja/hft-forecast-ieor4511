"""h30s group: LightGBM, as the shared ``gbm`` stub describes it.

One booster for the direction (``objective='multiclass'``, 3 classes) and one for the signed
log return (``objective='l2'``), with ``sample_weight`` passed through and ``seed`` used for
every source of randomness. ``lightgbm`` is imported inside ``fit`` so the package still
imports where it is not installed (``uv pip install lightgbm``).

Early stopping uses a blocked hold-out inside the training fold: the training rows arrive
sorted by instrument and then time, so every ``valid_every``-th contiguous block of
``valid_block`` rows (about four sessions of one stock on the 5 s grid) is held out. That
spreads the validation rows across stocks and dates, and the number of trees is chosen
without touching the test fold. Overlapping labels can only connect the edges of a block.
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
RET_SCALE: Final = 1e4  # fit the return head in bps: keeps gradients and leaf values sensible


class LgbmParams(PluginParams):
    """Moderate boosting settings; early stopping picks the number of trees."""

    n_estimators: int = Field(500, ge=1, le=10_000, description="Upper bound on trees")
    learning_rate: float = Field(0.05, gt=0, le=1)
    num_leaves: int = Field(31, ge=2, le=4096)
    min_child_samples: int = Field(500, ge=1)
    subsample: float = Field(0.5, gt=0, le=1, description="Row fraction per tree")
    colsample_bytree: float = Field(0.8, gt=0, le=1)
    reg_lambda: float = Field(1.0, ge=0)
    valid_block: int = Field(20_000, ge=100, description="Rows per hold-out block")
    valid_every: int = Field(10, ge=2, description="Hold out every n-th block for stopping")
    early_stopping_rounds: int = Field(30, ge=1)
    n_jobs: int = Field(-1)


@register_model(name="lgbm", version="1.0.0")
class Lgbm(Forecaster):
    """LightGBM multiclass direction head plus L2 return head."""

    Params = LgbmParams
    params: LgbmParams
    supports_sample_weight: ClassVar[bool] = True

    def __init__(self, params: PluginParams | None = None) -> None:
        super().__init__(params)
        self._classifier: Any = None
        self._regressor: Any = None
        self._classes: IntArray = np.array(CLASSES)

    def holdout_mask(self, n_rows: int) -> np.ndarray:
        """True for rows in every ``valid_every``-th block of ``valid_block`` rows.

        Falls back to the last block, or to the last tenth of the rows when there is only one
        block, so both parts are never empty (for at least two rows).
        """
        rows = np.arange(n_rows)
        block = rows // self.params.valid_block
        mask: np.ndarray = (block % self.params.valid_every) == self.params.valid_every - 1
        if not mask.any():
            mask = (
                block == block.max() if block.max() > 0 else rows >= n_rows - max(n_rows // 10, 1)
            )
        return mask

    def _common(self, seed: int) -> dict[str, Any]:
        p = self.params
        return {
            "n_estimators": p.n_estimators,
            "learning_rate": p.learning_rate,
            "num_leaves": p.num_leaves,
            "min_child_samples": p.min_child_samples,
            "subsample": p.subsample,
            "subsample_freq": 1,
            "colsample_bytree": p.colsample_bytree,
            "reg_lambda": p.reg_lambda,
            "random_state": seed,
            "n_jobs": p.n_jobs,
            "verbose": -1,
        }

    def fit(
        self,
        x: FloatArray,
        y_dir: IntArray,
        y_ret: FloatArray,
        sample_weight: FloatArray | None,
        seed: int,
    ) -> None:
        """Fit both heads, early-stopping each on blocked hold-out rows of the training fold."""
        import lightgbm as lgb

        valid_rows = self.holdout_mask(len(x))
        fit_rows = ~valid_rows
        weight = np.ones(len(x)) if sample_weight is None else sample_weight
        stop = [lgb.early_stopping(self.params.early_stopping_rounds, verbose=False)]

        # no explicit objective: LightGBM picks multiclass for three classes and binary when a
        # fold happens to contain only two (forcing multiclass would refuse that fold)
        self._classifier = lgb.LGBMClassifier(**self._common(seed))
        self._classifier.fit(
            x[fit_rows],
            y_dir[fit_rows],
            sample_weight=weight[fit_rows],
            eval_set=[(x[valid_rows], y_dir[valid_rows])],
            eval_sample_weight=[weight[valid_rows]],
            callbacks=stop,
        )
        self._classes = np.asarray(self._classifier.classes_)

        target = y_ret * RET_SCALE
        self._regressor = lgb.LGBMRegressor(objective="l2", **self._common(seed))
        self._regressor.fit(
            x[fit_rows],
            target[fit_rows],
            sample_weight=weight[fit_rows],
            eval_set=[(x[valid_rows], target[valid_rows])],
            eval_sample_weight=[weight[valid_rows]],
            callbacks=stop,
        )

    def predict(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Probabilities in down/flat/up order (padded if a fold saw two classes) and the return.

        Raises:
            ConfigError: If called before ``fit``.
        """
        if self._classifier is None or self._regressor is None:
            raise ConfigError("lgbm must be fitted before predicting")
        fitted = self._classifier.predict_proba(x)
        out = np.zeros((len(x), N_CLASSES))
        for column, label in enumerate(self._classes):
            out[:, CLASSES.index(int(label))] = fitted[:, column]
        predicted: FloatArray = self._regressor.predict(x) / RET_SCALE
        return out, predicted

    def state_dict(self) -> dict[str, Any]:
        """The fitted boosters (pickled by the caller), plus the chosen tree counts."""
        return {
            "classifier": self._classifier,
            "regressor": self._regressor,
            "best_iterations": (
                getattr(self._classifier, "best_iteration_", None),
                getattr(self._regressor, "best_iteration_", None),
            ),
        }

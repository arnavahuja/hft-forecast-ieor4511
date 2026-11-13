"""Sequence models: the GRU, LSTM and TCN encoders ported from the group's Colab notebook.

The notebook (``WRDS_TAQ_SP500_1Min_Research_v4``) regresses the signed 60 s return with a
single head. The repository's :class:`Forecaster` also has to return class probabilities, so
each encoder here feeds the notebook's ``LayerNorm -> Linear`` magnitude head *and* a
3-class direction head, trained together with
``cross_entropy(direction) + lambda_magnitude * huber(signed return)``.

Everything else follows the notebook: AdamW, ``ReduceLROnPlateau``, gradient clipping at 1.0,
feature standardisation clipped to +-15 sigma, a standardised target, and early stopping on a
held-out tail of the training fold. ``torch`` is imported lazily, like ``wrds`` in the data
sources.
"""

from __future__ import annotations

from typing import Any, ClassVar, Final

import numpy as np
import structlog
from pydantic import Field

from hft_forecast.core.errors import ConfigError
from hft_forecast.core.params import PluginParams
from hft_forecast.core.types import FloatArray, IntArray
from hft_forecast.ml.base import N_CLASSES, Forecaster
from hft_forecast.ml.models.baselines import CLASSES
from hft_forecast.ml.models.registry import register_model

log = structlog.get_logger(__name__)

ENCODERS: Final = ("gru", "lstm", "tcn")
DILATIONS: Final = (1, 2, 4, 8, 16)
CLIP_SIGMA: Final = 15.0
GRID_SECONDS: Final = 5.0
CLOCK: Final = "f_seconds_since_open"


class TorchSeqParams(PluginParams):
    """Shape of the network, of the loss and of the window sampling."""

    encoder: str = Field("gru", description=f"One of {ENCODERS}")
    lookback: int = Field(60, ge=2, le=2000, description="Bars per sequence window")
    hidden: int = Field(64, ge=1, le=2048)
    layers: int = Field(2, ge=1, le=8)
    dropout: float = Field(0.1, ge=0, lt=1)
    lambda_magnitude: float = Field(1.0, ge=0, description="Weight of the Huber head")
    epochs: int = Field(10, ge=1, le=1000)
    patience: int = Field(3, ge=1, description="Epochs without improvement before stopping")
    batch_size: int = Field(512, ge=1)
    learning_rate: float = Field(3e-4, gt=0)
    weight_decay: float = Field(1e-4, ge=0)
    train_stride: int = Field(
        12, ge=1, description="Keep every n-th training window; 12 is one per minute on a 5 s grid"
    )
    max_train_windows: int = Field(
        200_000, ge=100, description="Cap on training windows, sampled evenly over the fold"
    )
    validation_fraction: float = Field(
        0.1, gt=0, lt=0.5, description="Tail of the training fold held out for early stopping"
    )


def build_module(params: TorchSeqParams, n_features: int) -> Any:
    """The ``nn.Module``: one of the three encoders plus the direction and magnitude heads.

    Each encoder returns ``(n, lookback, hidden)`` so the heads can read the last step the
    same way, whichever encoder is in use.

    Raises:
        ConfigError: If ``encoder`` is not one of :data:`ENCODERS`.
    """
    from torch import nn

    if params.encoder not in ENCODERS:
        raise ConfigError(f"unknown encoder '{params.encoder}'; expected one of {ENCODERS}")

    class CausalConv1d(nn.Module):  # type: ignore[misc]  # torch is untyped here
        """Convolution that cannot see the future: left-padded, then the right tail trimmed."""

        def __init__(self, in_c: int, out_c: int, kernel: int = 3, dilation: int = 1) -> None:
            super().__init__()
            self.trim = (kernel - 1) * dilation
            self.conv = nn.Conv1d(in_c, out_c, kernel, padding=self.trim, dilation=dilation)

        def forward(self, x: Any) -> Any:
            z = self.conv(x)
            return z[:, :, : -self.trim] if self.trim else z

    class ResidualTCNBlock(nn.Module):  # type: ignore[misc]
        """Dilated causal convolution with a GELU and a residual projection."""

        def __init__(self, in_c: int, out_c: int, dilation: int) -> None:
            super().__init__()
            self.conv = CausalConv1d(in_c, out_c, 3, dilation)
            self.act = nn.GELU()
            self.proj = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

        def forward(self, x: Any) -> Any:
            return self.act(self.conv(x) + self.proj(x))

    class TcnEncoder(nn.Module):  # type: ignore[misc]
        """Stack of dilated blocks; receptive field ``1 + 2 * sum(DILATIONS)`` = 63 bars."""

        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                *[
                    ResidualTCNBlock(n_features if i == 0 else params.hidden, params.hidden, d)
                    for i, d in enumerate(DILATIONS)
                ]
            )

        def forward(self, x: Any) -> Any:
            return self.net(x.transpose(1, 2)).transpose(1, 2)

    class RnnEncoder(nn.Module):  # type: ignore[misc]
        """Two-layer GRU or LSTM, as in the notebook's ``RNNForecaster``."""

        def __init__(self) -> None:
            super().__init__()
            net = nn.GRU if params.encoder == "gru" else nn.LSTM
            self.rnn = net(
                n_features,
                params.hidden,
                num_layers=params.layers,
                batch_first=True,
                dropout=params.dropout if params.layers > 1 else 0.0,
            )

        def forward(self, x: Any) -> Any:
            out, _ = self.rnn(x)
            return out

    class SeqModel(nn.Module):  # type: ignore[misc]
        """Shared encoder, a 3-class direction head and a signed-return head."""

        def __init__(self) -> None:
            super().__init__()
            self.encoder: nn.Module = TcnEncoder() if params.encoder == "tcn" else RnnEncoder()
            self.norm = nn.LayerNorm(params.hidden)
            self.direction = nn.Linear(params.hidden, N_CLASSES)
            self.magnitude = nn.Linear(params.hidden, 1)

        def forward(self, x: Any) -> tuple[Any, Any]:
            """``(n, lookback, f)`` in, ``(logits (n, 3), y_hat (n,))`` out."""
            hidden = self.norm(self.encoder(x)[:, -1, :])
            return self.direction(hidden), self.magnitude(hidden).squeeze(-1)

    return SeqModel()  # the caller seeds torch before building, so init is reproducible


def session_windows(seconds_since_open: FloatArray, lookback: int) -> tuple[Any, Any]:
    """Window start offsets and a per-step mask that never crosses a session boundary.

    ``fit`` is handed a plain ``(n, f)`` design matrix with no session column, so session
    identity is recovered from the ``f_seconds_since_open`` feature: within one session it
    rises by exactly one grid step per row, and it drops at every new session.

    Returns:
        ``(index (n, lookback), valid (n, lookback))`` — rows to gather for each window and
        which of those steps are genuinely the preceding bars of the same session.
    """
    n = len(seconds_since_open)
    offsets = np.arange(-(lookback - 1), 1)
    index = np.clip(np.arange(n)[:, None] + offsets[None, :], 0, n - 1)
    expected = seconds_since_open[:, None] + offsets[None, :] * GRID_SECONDS
    valid = np.isclose(seconds_since_open[index], expected)
    return index, valid


@register_model(name="torch_seq", version="1.0.0")
class TorchSeq(Forecaster):
    """GRU, LSTM or TCN over the last ``lookback`` bars, with a direction and a return head."""

    Params = TorchSeqParams
    params: TorchSeqParams
    supports_sample_weight: ClassVar[bool] = False

    def __init__(self, params: PluginParams | None = None) -> None:
        super().__init__(params)
        self._model: Any = None
        self._x_mean: FloatArray = np.zeros(0)
        self._x_std: FloatArray = np.ones(0)
        self._y_mean = 0.0
        self._y_std = 1.0
        self._clock = -1

    def _prepare(self, x: FloatArray, y_ret: FloatArray) -> tuple[Any, Any]:
        """Record the standardisation and return the window index and mask.

        ``fit`` is handed a plain design matrix with no session column, so the clock comes
        from the ``f_seconds_since_open`` feature of the ``basic`` set. Without it the rows
        are treated as one unbroken session, which is right for a synthetic single-session
        fold and wrong for real data.

        shortcut: no clock feature means windows can span a session boundary; the pipeline
        always runs the `basic` set, so upgrade this only if a group drops it.
        """
        self._clock = self.feature_names.index(CLOCK) if CLOCK in self.feature_names else -1
        if self._clock < 0:
            log.warning("torch_seq_no_clock", feature=CLOCK)
        self._x_mean = np.nan_to_num(np.nanmean(x, axis=0))
        std = np.nan_to_num(np.nanstd(x, axis=0))
        self._x_std = np.where(std < 1e-6, 1.0, std)
        self._y_mean, self._y_std = float(y_ret.mean()), max(float(y_ret.std()), 1e-9)
        return self._windows(x)

    def _windows(self, x: FloatArray) -> tuple[Any, Any]:
        """Window index and mask, from the clock feature or from the row order."""
        clock = (
            x[:, self._clock]
            if self._clock >= 0
            else np.arange(len(x), dtype=np.float64) * GRID_SECONDS
        )
        return session_windows(clock, self.params.lookback)

    def _batch(self, x: FloatArray, rows: Any, index: Any, valid: Any) -> Any:
        """Standardised ``(len(rows), lookback, f)`` windows, padding steps masked to zero."""
        import torch

        window = x[index[rows]]
        window = np.clip((window - self._x_mean) / self._x_std, -CLIP_SIGMA, CLIP_SIGMA)
        window[~valid[rows]] = 0.0  # zero is the standardised mean, i.e. mean imputation
        return torch.from_numpy(np.nan_to_num(window).astype(np.float32))

    def fit(
        self,
        x: FloatArray,
        y_dir: IntArray,
        y_ret: FloatArray,
        sample_weight: FloatArray | None,
        seed: int,
    ) -> None:
        """Train on strided windows of the fold, early-stopping on its last sessions."""
        import torch
        from torch import nn

        p = self.params
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        index, valid = self._prepare(x, y_ret)
        rows = np.arange(0, len(x), p.train_stride)
        if len(rows) > p.max_train_windows:
            rows = rows[np.sort(rng.choice(len(rows), p.max_train_windows, replace=False))]
        cut = max(int(len(rows) * (1 - p.validation_fraction)), 1)
        train_rows, val_rows = rows[:cut], rows[cut:]  # time-ordered, so the tail is the future
        if len(train_rows) < 2 or len(val_rows) == 0:
            raise ConfigError(
                f"torch_seq needs at least 3 training windows after striding by "
                f"{p.train_stride}; got {len(rows)} from {len(x)} rows"
            )
        # a short fold (a test fixture, or a fold the purge almost emptied) gets whole-fold
        # batches rather than an error
        batch_size = min(p.batch_size, len(train_rows))

        self._model = build_module(p, x.shape[1])
        target = torch.from_numpy(((y_ret - self._y_mean) / self._y_std).astype(np.float32))
        classes = torch.from_numpy(np.searchsorted(np.array(CLASSES), y_dir).astype(np.int64))
        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=p.learning_rate, weight_decay=p.weight_decay
        )
        schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )
        huber, cross = nn.HuberLoss(delta=1.0), nn.CrossEntropyLoss()

        def loss_of(batch: Any) -> Any:
            logits, y_hat = self._model(self._batch(x, batch, index, valid))
            return cross(logits, classes[batch]) + p.lambda_magnitude * huber(y_hat, target[batch])

        best, best_state, misses = float("inf"), None, 0
        for epoch in range(1, p.epochs + 1):
            self._model.train()
            order = rng.permutation(train_rows)
            for start in range(0, len(order) - batch_size + 1, batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss = loss_of(order[start : start + batch_size])
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
            self._model.eval()
            with torch.no_grad():
                losses = [
                    float(loss_of(val_rows[s : s + batch_size]))
                    for s in range(0, len(val_rows), batch_size)
                ]
            validation = float(np.mean(losses))
            schedule.step(validation)
            log.info(
                "torch_seq_epoch", encoder=p.encoder, epoch=epoch, validation=round(validation, 6)
            )
            if validation < best - 1e-5:
                best, misses = validation, 0
                best_state = {k: v.detach().clone() for k, v in self._model.state_dict().items()}
            else:
                misses += 1
                if misses >= p.patience:
                    break
        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._model.eval()

    def predict(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Probabilities and expected return for every row, in batches.

        Raises:
            ConfigError: If called before ``fit``.
        """
        import torch

        if self._model is None:
            raise ConfigError("torch_seq must be fitted before predicting")
        p = self.params
        index, valid = self._windows(x)
        probabilities = np.zeros((len(x), N_CLASSES))
        predicted = np.zeros(len(x))
        with torch.no_grad():
            for start in range(0, len(x), p.batch_size):
                rows = np.arange(start, min(start + p.batch_size, len(x)))
                logits, y_hat = self._model(self._batch(x, rows, index, valid))
                probabilities[rows] = torch.softmax(logits, dim=1).numpy()
                predicted[rows] = y_hat.numpy() * self._y_std + self._y_mean
        return probabilities, predicted

    def state_dict(self) -> dict[str, Any]:
        """Weights and the standardisation needed to reproduce a prediction."""
        if self._model is None:
            return {}
        return {
            "encoder": self.params.encoder,
            "weights": self._model.state_dict(),
            "x_mean": self._x_mean,
            "x_std": self._x_std,
            "y_mean": self._y_mean,
            "y_std": self._y_std,
        }

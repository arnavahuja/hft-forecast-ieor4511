"""Sequence model. Stub: the two heads and the batching are written, the encoder is not.

Choosing between a GRU, a TCN and a transformer is the modelling work this capstone is for,
so ``encoder`` is a parameter with no implementation behind it. ``torch`` is imported lazily,
like ``wrds`` and ``yfinance`` in the data sources.
"""

from __future__ import annotations

from typing import Any, ClassVar, Final

from pydantic import Field

from hft_forecast.core.params import PluginParams
from hft_forecast.core.types import FloatArray, IntArray
from hft_forecast.ml.base import N_CLASSES, Forecaster
from hft_forecast.ml.models.registry import register_model

ENCODERS: Final = ("gru", "tcn", "transformer")
MESSAGE = (
    "torch_seq is a stub: build the encoder named by `encoder`, then train the two heads "
    "with loss = cross_entropy(direction) + lambda_magnitude * huber(signed return). "
    "Use ml.datasets.windowed() for the (n, lookback, f) batches; it never spans a session."
)


class TorchSeqParams(PluginParams):
    """Shape of the network and of the loss."""

    encoder: str = Field("gru", description=f"One of {ENCODERS}")
    lookback: int = Field(60, ge=2, le=2000, description="Bars per sequence window")
    hidden: int = Field(64, ge=1, le=2048)
    layers: int = Field(2, ge=1, le=8)
    dropout: float = Field(0.1, ge=0, lt=1)
    lambda_magnitude: float = Field(1.0, ge=0, description="Weight of the Huber head")
    epochs: int = Field(10, ge=1, le=1000)
    batch_size: int = Field(1024, ge=1)
    learning_rate: float = Field(1e-3, gt=0)


def build_module(params: TorchSeqParams, n_features: int) -> Any:
    """The ``nn.Module`` skeleton: an encoder slot and the two heads.

    Raises:
        NotImplementedError: The encoder itself is the group's job.
    """
    import torch
    from torch import nn

    class SeqModel(nn.Module):  # type: ignore[misc]  # torch is untyped here
        """Shared encoder, a 3-class direction head and a signed-return head."""

        def __init__(self) -> None:
            super().__init__()
            self.encoder: nn.Module = nn.Identity()  # the group fills this in
            self.direction = nn.Linear(params.hidden, N_CLASSES)
            self.magnitude = nn.Linear(params.hidden, 1)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """``(n, lookback, f)`` in, ``(logits (n, 3), y_hat (n,))`` out."""
            hidden = self.encoder(x)[:, -1, :]
            return self.direction(hidden), self.magnitude(hidden).squeeze(-1)

    if params.encoder in ENCODERS:
        raise NotImplementedError(MESSAGE)
    raise NotImplementedError(f"unknown encoder '{params.encoder}'; expected one of {ENCODERS}")


@register_model(name="torch_seq", version="0.1.0")
class TorchSeq(Forecaster):
    """Sequence model with a direction head and a magnitude head (not implemented)."""

    Params = TorchSeqParams
    params: TorchSeqParams
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

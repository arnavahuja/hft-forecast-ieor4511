"""h5m group: a GRU over the raw recent history, plus the engineered features.

Input columns are split by name:
  - ``f_seq_<var>_l<lag>`` (from the ``h5m_seq`` feature set) are reshaped into a sequence
    (batch, length, variables), oldest bar first, and read by a one-layer GRU;
  - every other column (``basic`` + ``h30s_flow``) goes through a small dense layer.
The GRU's last hidden state and the dense embedding are concatenated and feed the same two
heads as ``h5m_mlp`` (3-class direction, return in bps).

Training, scaling, early stopping, weights and determinism are inherited unchanged from
``h5m_mlp``, so T7 differs from T6 only in what the network can see and how it reads it.
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Final

import numpy as np
from pydantic import Field

from hft_forecast.core.errors import ConfigError
from hft_forecast.ml.base import N_CLASSES
from hft_forecast.ml.models.h5m_mlp import H5mMlp, MlpParams
from hft_forecast.ml.models.registry import register_model

SEQ: Final = re.compile(r"^f_seq_(.+)_l(\d+)$")


class GruParams(MlpParams):
    """MLP settings plus the size of the sequence encoder."""

    gru_hidden: int = Field(32, ge=1, description="GRU hidden state size")
    static_hidden: int = Field(64, ge=1, description="Dense embedding of the other features")


def sequence_layout(names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Column indices: (lags, variables) for the sequence, oldest lag first; then the rest.

    With no ``f_seq_*`` columns the grid is empty (shape (0, 0)) and the network has no GRU.

    Raises:
        ConfigError: If the lag x variable grid is incomplete.
    """
    found: dict[tuple[str, int], int] = {}
    static: list[int] = []
    for i, name in enumerate(names):
        m = SEQ.match(name)
        if m:
            found[(m.group(1), int(m.group(2)))] = i
        else:
            static.append(i)
    if not found:
        return np.zeros((0, 0), dtype=int), np.array(static, dtype=int)
    variables = sorted({v for v, _ in found})
    lags = sorted({lag for _, lag in found}, reverse=True)  # oldest first
    try:
        grid = np.array([[found[(v, lag)] for v in variables] for lag in lags])
    except KeyError as exc:
        raise ConfigError(f"incomplete f_seq_* grid, missing {exc}") from exc
    return grid, np.array(static, dtype=int)


@register_model(name="h5m_gru", version="1.0.0")
class H5mGru(H5mMlp):
    """GRU sequence encoder + dense static encoder, two heads."""

    Params = GruParams
    params: GruParams

    def _build(self, n_in: int) -> Any:
        import torch
        from torch import nn

        if not self.feature_names:
            raise ConfigError("h5m_gru needs the feature names (set_feature_names)")
        if len(self.feature_names) != n_in:
            raise ConfigError(f"{len(self.feature_names)} feature names for {n_in} columns")
        grid, static = sequence_layout(self.feature_names)
        if grid.size == 0:
            warnings.warn("h5m_gru: no f_seq_* columns, so no GRU branch (add h5m_seq)",
                          stacklevel=2)
        p = self.params

        class Net(nn.Module):  # type: ignore[misc]
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("seq_idx", torch.from_numpy(grid.reshape(-1)).long())
                self.register_buffer("static_idx", torch.from_numpy(static).long())
                self.n_lags, self.n_vars = grid.shape
                self.gru = (
                    nn.GRU(self.n_vars, p.gru_hidden, batch_first=True) if grid.size else None
                )
                width = p.gru_hidden if grid.size else 0
                if len(static):
                    self.static = nn.Sequential(nn.Linear(len(static), p.static_hidden), nn.ReLU())
                    width += p.static_hidden
                else:
                    self.static = None
                layers: list[nn.Module] = []
                for h in p.hidden:
                    layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(p.dropout)]
                    width = h
                self.trunk = nn.Sequential(*layers)
                self.direction = nn.Linear(width, N_CLASSES)
                self.ret = nn.Linear(width, 1)

            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                parts = []
                if self.gru is not None:
                    seq = x.index_select(1, self.seq_idx).view(-1, self.n_lags, self.n_vars)
                    _, h = self.gru(seq)
                    parts.append(h[-1])
                if self.static is not None:
                    parts.append(self.static(x.index_select(1, self.static_idx)))
                z = self.trunk(torch.cat(parts, dim=1))
                return self.direction(z), self.ret(z).squeeze(-1)

        return Net()

"""h5m group: a small multilayer perceptron with the two heads every model must have.

Same contract and the same early-stopping scheme as the h30s ``lgbm`` model, so the two are
compared on equal terms (same features, folds, labels and hold-out rows):

- inputs are standardised with the mean and standard deviation of the fitting rows only
  (stored in the model, so the test fold never informs them), then clipped to +-10;
- a shared trunk (Linear -> ReLU -> Dropout, twice) feeds two heads: 3-class direction
  logits (down / flat / up) and the log return in bps;
- loss = weighted cross-entropy + ``ret_weight`` x weighted Huber loss on the return (Huber,
  not squared error, because a few extreme 5-minute returns would otherwise dominate);
- AdamW, mini-batches in a seeded random order; early stopping on the blocked hold-out rows
  of the training fold (every ``valid_every``-th block of ``valid_block`` rows), keeping the
  best epoch's weights.

Deterministic for a given seed: torch, numpy and the batch order are seeded, torch runs with
deterministic algorithms and a fixed number of threads (both restored afterwards).
``torch`` is imported inside ``fit``/``predict`` so the package imports without it.
"""

from __future__ import annotations

import copy
from typing import Any, ClassVar, Final

import numpy as np
from pydantic import Field

from hft_forecast.core.errors import ConfigError
from hft_forecast.core.params import PluginParams
from hft_forecast.core.types import FloatArray, IntArray
from hft_forecast.ml.base import N_CLASSES, Forecaster
from hft_forecast.ml.models.registry import register_model

CLASSES: Final = (-1, 0, 1)
RET_SCALE: Final = 1e4  # the return head works in bps
CLIP: Final = 10.0


class MlpParams(PluginParams):
    """Small on purpose: the signal is weak, so capacity is not the bottleneck."""

    hidden: list[int] = Field(default_factory=lambda: [64, 32], description="Trunk layer widths")
    dropout: float = Field(0.1, ge=0, lt=1)
    learning_rate: float = Field(1e-3, gt=0)
    weight_decay: float = Field(1e-4, ge=0)
    batch_size: int = Field(4096, ge=1)
    max_epochs: int = Field(10, ge=1)
    patience: int = Field(2, ge=1, description="Epochs without hold-out improvement")
    ret_weight: float = Field(1.0, ge=0, description="Weight of the return loss")
    huber_delta_bps: float = Field(10.0, gt=0, description="About one 5-minute std")
    valid_block: int = Field(20_000, ge=100, description="Rows per hold-out block")
    valid_every: int = Field(10, ge=2, description="Hold out every n-th block for stopping")
    threads: int = Field(4, ge=1)


@register_model(name="h5m_mlp", version="1.0.0")
class H5mMlp(Forecaster):
    """MLP trunk with a direction head and a return head."""

    Params = MlpParams
    params: MlpParams
    supports_sample_weight: ClassVar[bool] = True

    def __init__(self, params: PluginParams | None = None) -> None:
        super().__init__(params)
        self._net: Any = None
        self._mean: FloatArray | None = None
        self._std: FloatArray | None = None
        self.best_epoch: int = -1
        self.history: list[float] = []

    def holdout_mask(self, n_rows: int) -> np.ndarray:
        """Same blocked hold-out as ``lgbm``; never empty on either side (n >= 2)."""
        rows = np.arange(n_rows)
        block = rows // self.params.valid_block
        mask: np.ndarray = (block % self.params.valid_every) == self.params.valid_every - 1
        if not mask.any():
            mask = (
                block == block.max() if block.max() > 0 else rows >= n_rows - max(n_rows // 10, 1)
            )
        return mask

    def _scale(self, x: FloatArray) -> np.ndarray:
        assert self._mean is not None and self._std is not None
        z = (np.asarray(x, dtype=np.float64) - self._mean) / self._std
        return np.clip(np.nan_to_num(z, nan=0.0, posinf=CLIP, neginf=-CLIP), -CLIP, CLIP).astype(
            np.float32
        )

    def _build(self, n_in: int) -> Any:
        import torch
        from torch import nn

        p = self.params

        class Net(nn.Module):  # type: ignore[misc]
            def __init__(self) -> None:
                super().__init__()
                layers: list[nn.Module] = []
                width = n_in
                for h in p.hidden:
                    layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(p.dropout)]
                    width = h
                self.trunk = nn.Sequential(*layers)
                self.direction = nn.Linear(width, N_CLASSES)
                self.ret = nn.Linear(width, 1)

            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                z = self.trunk(x)
                return self.direction(z), self.ret(z).squeeze(-1)

        return Net()

    def fit(
        self,
        x: FloatArray,
        y_dir: IntArray,
        y_ret: FloatArray,
        sample_weight: FloatArray | None,
        seed: int,
    ) -> None:
        """Fit on one training fold, early-stopping on its blocked hold-out rows."""
        import torch
        from torch.nn import functional as F

        p = self.params
        was_det = torch.are_deterministic_algorithms_enabled()
        threads = torch.get_num_threads()
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(p.threads)
        try:
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)

            valid = self.holdout_mask(len(x))
            fit_rows = ~valid
            x64 = np.asarray(x, dtype=np.float64)
            self._mean = np.nanmean(x64[fit_rows], axis=0)
            std = np.nanstd(x64[fit_rows], axis=0)
            self._std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
            self._mean = np.where(np.isfinite(self._mean), self._mean, 0.0)

            xs = torch.from_numpy(self._scale(x64))
            cls = torch.from_numpy(np.searchsorted(CLASSES, np.asarray(y_dir)).astype(np.int64))
            ret = torch.from_numpy((np.asarray(y_ret, dtype=np.float64) * RET_SCALE).astype(np.float32))
            w = np.ones(len(x)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
            wt = torch.from_numpy((w / w[fit_rows].mean()).astype(np.float32))

            self._net = self._build(xs.shape[1])
            opt = torch.optim.AdamW(
                self._net.parameters(), lr=p.learning_rate, weight_decay=p.weight_decay
            )

            def loss_on(idx: torch.Tensor) -> torch.Tensor:
                logits, r_hat = self._net(xs[idx])
                ce = F.cross_entropy(logits, cls[idx], reduction="none")
                hub = F.huber_loss(r_hat, ret[idx], reduction="none", delta=p.huber_delta_bps)
                ww = wt[idx]
                return ((ce + p.ret_weight * hub) * ww).sum() / ww.sum()

            fit_idx = np.flatnonzero(fit_rows)
            val_idx = torch.from_numpy(np.flatnonzero(valid))
            best, best_state, stale = float("inf"), None, 0
            self.history = []
            for epoch in range(p.max_epochs):
                self._net.train()
                order = fit_idx[rng.permutation(len(fit_idx))]
                for start in range(0, len(order), p.batch_size):
                    batch = torch.from_numpy(order[start : start + p.batch_size])
                    opt.zero_grad()
                    loss_on(batch).backward()
                    opt.step()
                self._net.eval()
                with torch.no_grad():
                    val = float(loss_on(val_idx))
                self.history.append(val)
                if val < best - 1e-9:
                    best, best_state, stale, self.best_epoch = (
                        val,
                        copy.deepcopy(self._net.state_dict()),
                        0,
                        epoch,
                    )
                else:
                    stale += 1
                    if stale >= p.patience:
                        break
            if best_state is not None:
                self._net.load_state_dict(best_state)
            self._net.eval()
        finally:
            torch.use_deterministic_algorithms(was_det)
            torch.set_num_threads(threads)

    def predict(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Probabilities in down/flat/up order and the expected log return.

        Raises:
            ConfigError: If called before ``fit``.
        """
        if self._net is None:
            raise ConfigError("h5m_mlp must be fitted before predicting")
        import torch

        xs = torch.from_numpy(self._scale(x))
        probs, rets = [], []
        with torch.no_grad():
            for start in range(0, len(xs), 65_536):
                logits, r_hat = self._net(xs[start : start + 65_536])
                probs.append(torch.softmax(logits, dim=-1).double().numpy())
                rets.append(r_hat.double().numpy())
        if not probs:
            return np.zeros((0, N_CLASSES)), np.zeros(0)
        out: FloatArray = np.concatenate(probs)
        predicted: FloatArray = np.concatenate(rets) / RET_SCALE
        return out, predicted

    def state_dict(self) -> dict[str, Any]:
        """Network weights, the input scaling and the early-stopping record."""
        return {
            "net": None if self._net is None else self._net.state_dict(),
            "mean": self._mean,
            "std": self._std,
            "best_epoch": self.best_epoch,
            "history": self.history,
        }

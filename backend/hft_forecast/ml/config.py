"""The experiment configuration: one YAML file per horizon group, hashed for reproducibility."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from hft_forecast.core.run_config import DEFAULT_SEED, DataSpec, PluginRef
from hft_forecast.ml.base import H5M, HorizonSpec

HASH_LENGTH: Final = 16
DEFAULT_MAX_ROWS: Final = 20_000_000


class LabelParams(BaseModel):
    """How the forward return becomes a label (§4.2).

    The dead zone is the research knob: roughly the half-spread at 30 s, wider at 30 min.
    The template only makes it a parameter and reports the class balance.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dead_zone_bps: float = Field(1.0, ge=0, description="|fwd_logret| <= tau is class 0")
    drop_stale_targets: bool = Field(
        True, description="Drop rows whose target bar carried no quote"
    )
    warmup_seconds: int = Field(60, ge=0, description="Masked after the opening cross")
    cooldown_seconds: int = Field(60, ge=0, description="Masked before the closing cross")


class CVSpec(BaseModel):
    """Purged, embargoed walk-forward in session units (§5)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scheme: str = Field("walk_forward", pattern="^walk_forward$")
    train_sessions: int = Field(40, ge=1)
    test_sessions: int = Field(10, ge=1)
    anchored: bool = False
    embargo_bars: int | None = Field(None, description="Defaults to the horizon in bars when unset")
    purge: bool = True
    min_train_rows: int = Field(100_000, ge=0)
    stride: int | None = Field(
        None, description="Keep every n-th training row for a non-overlapping sanity run"
    )


class ExperimentConfig(BaseModel):
    """Everything one ``hftf ml run`` needs.

    The hash covers the data, horizon, labels, features and CV, so two configs that would
    build the same dataset share the cached feature files.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment: str = Field(description="Trial-counting bucket; one per horizon (h30s, ...)")
    name: str = ""
    horizon: HorizonSpec = H5M
    data: DataSpec
    manifest: str | None = Field(
        None, description="Name of a data/manifests entry; overrides 'data' when set"
    )
    features: list[PluginRef] = Field(default_factory=lambda: [PluginRef(name="basic")])
    labels: LabelParams = LabelParams()
    cv: CVSpec = CVSpec()
    model: PluginRef = PluginRef(name="logit_ridge")
    seed: int = DEFAULT_SEED
    max_rows_in_memory: int = Field(DEFAULT_MAX_ROWS, ge=1)
    tags: list[str] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _grid_matches_data(self) -> ExperimentConfig:
        if self.data.frequency is not self.horizon.grid:
            raise ValueError(
                f"data.frequency {self.data.frequency} is not the horizon grid {self.horizon.grid}"
            )
        return self

    @property
    def embargo_bars(self) -> int:
        """Embargo length, defaulting to the label horizon."""
        return self.horizon.steps if self.cv.embargo_bars is None else self.cv.embargo_bars

    def dataset_hash(self) -> str:
        """Hash of everything that determines the feature files."""
        payload = self.model_dump(mode="json", include={"horizon", "data", "manifest", "labels"})
        payload["features"] = [f.model_dump(mode="json") for f in self.features]
        return _hash(payload)

    def config_hash(self) -> str:
        """Hash of the whole configuration (the run's identity)."""
        return _hash(self.model_dump(mode="json", exclude={"name", "notes", "tags"}))

    @classmethod
    def load(cls, path: Path) -> ExperimentConfig:
        """Read a YAML or JSON experiment config."""
        return cls.model_validate(yaml.safe_load(path.read_text()))


def _hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:HASH_LENGTH]

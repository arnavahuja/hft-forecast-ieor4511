"""Validates a full run configuration before it runs.

Returns human-readable reasons for any mismatch so the UI can disable invalid combinations
instead of failing mid-run. Checks are generic: they read capabilities and interface
implementations, never plugin names.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from hft_forecast.core.errors import PluginNotFoundError
from hft_forecast.core.interfaces import (
    ENGINE_VECTORIZED,
    FREQ_DAILY,
    FREQ_INTRADAY,
    FREQ_SUBMINUTE,
    NEEDS_BENCHMARK,
    Overlay,
    Strategy,
)
from hft_forecast.core.registry import PluginKind, PluginSpec, registry
from hft_forecast.core.run_config import BacktestConfig, PluginRef
from hft_forecast.core.types import TargetKind


class IssueLevel(StrEnum):
    """Severity of a compatibility issue."""

    ERROR = "error"
    WARNING = "warning"


class CompatibilityIssue(BaseModel):
    """One problem with a configuration."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    level: IssueLevel
    field: str
    message: str


class CompatibilityReport(BaseModel):
    """All issues found."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    issues: list[CompatibilityIssue]

    @property
    def ok(self) -> bool:
        """True when there are no errors (warnings allowed)."""
        return not any(i.level is IssueLevel.ERROR for i in self.issues)


def _freq_caps(caps: frozenset[str]) -> set[str]:
    return {c for c in caps if c.startswith("freq:")}


def _check_frequency(config: BacktestConfig, spec: PluginSpec[Any], fld: str, out: _Issues) -> None:
    """A plugin that declares any ``freq:*`` capability must declare the one it is run on.

    Sub-minute bars need both ``freq:intraday`` and ``freq:subminute``: most plugins written
    for daily or minute bars have windows and schedules that are meaningless at 5 s.
    """
    caps = _freq_caps(spec.capabilities)
    if not caps:
        return
    frequency = config.data.frequency
    wanted = [FREQ_INTRADAY if frequency.is_intraday else FREQ_DAILY]
    if frequency.is_subminute:
        wanted.append(FREQ_SUBMINUTE)
    for want in wanted:
        if want not in caps:
            out.err(fld, f"{spec.kind} '{spec.name}' does not support {want}")


class _Issues:
    """Collects issues."""

    def __init__(self) -> None:
        self.items: list[CompatibilityIssue] = []

    def err(self, fld: str, msg: str) -> None:
        self.items.append(CompatibilityIssue(level=IssueLevel.ERROR, field=fld, message=msg))

    def warn(self, fld: str, msg: str) -> None:
        self.items.append(CompatibilityIssue(level=IssueLevel.WARNING, field=fld, message=msg))


class CompatibilityChecker:
    """Checks a :class:`BacktestConfig` against plugin capabilities."""

    def check(self, config: BacktestConfig) -> CompatibilityReport:
        """Return all issues for a configuration."""
        out = _Issues()
        strategy = self._check_strategy(config, out)
        if config.constructor is not None:
            spec = self._resolve(
                PluginKind.PORTFOLIO_CONSTRUCTOR, config.constructor, "constructor", out
            )
            if spec is not None:
                _check_frequency(config, spec, "constructor", out)
        self._check_overlays(config, strategy, out)
        self._check_costs(config, out)
        self._check_data(config, out)
        return CompatibilityReport(issues=out.items)

    def _check_strategy(self, config: BacktestConfig, out: _Issues) -> PluginSpec[Any] | None:
        strategy = self._resolve(PluginKind.STRATEGY, config.strategy, "strategy", out)
        if strategy is None:
            return None
        cls: type[Strategy] = strategy.cls
        if not cls.implements_vectorized():
            out.err("strategy", f"Strategy '{strategy.name}' has no generate_targets")
        _check_frequency(config, strategy, "data.frequency", out)
        if NEEDS_BENCHMARK in strategy.capabilities and not config.benchmark:
            out.err("benchmark", f"Strategy '{strategy.name}' needs a benchmark")
        if cls.output_kind is TargetKind.SIGNALS and config.constructor is None:
            out.err(
                "constructor",
                f"Strategy '{strategy.name}' emits signals; choose a portfolio constructor",
            )
        if cls.output_kind is TargetKind.WEIGHTS and config.constructor is not None:
            out.warn(
                "constructor",
                "The strategy already emits weights; the constructor "
                "will re-weight them as signals",
            )
        return strategy

    def _check_overlays(
        self, config: BacktestConfig, strategy: PluginSpec[Any] | None, out: _Issues
    ) -> None:
        del strategy  # every overlay is checked the same way now that there is one engine
        for k, ref in enumerate(config.overlays):
            fld = f"overlays[{k}]"
            spec = self._resolve(PluginKind.OVERLAY, ref, fld, out)
            if spec is None:
                continue
            ocls: type[Overlay] = spec.cls
            if not ocls.implements_vectorized():
                out.err(fld, f"Overlay '{spec.name}' has no vectorized form")
            if NEEDS_BENCHMARK in spec.capabilities and not config.benchmark:
                out.err(fld, f"Overlay '{spec.name}' needs a benchmark")
            _check_frequency(config, spec, fld, out)

    def _check_costs(self, config: BacktestConfig, out: _Issues) -> None:
        for k, ref in enumerate(config.costs):
            spec = self._resolve(PluginKind.COST_MODEL, ref, f"costs[{k}]", out)
            if spec is not None and ENGINE_VECTORIZED not in spec.capabilities:
                out.err(f"costs[{k}]", f"Cost model '{spec.name}' has no vectorized form")
        if config.slippage is not None:
            spec = self._resolve(PluginKind.SLIPPAGE_MODEL, config.slippage, "slippage", out)
            if spec is not None and ENGINE_VECTORIZED not in spec.capabilities:
                out.err("slippage", f"Slippage model '{spec.name}' has no vectorized form")
        if not config.costs and config.slippage is None:
            out.warn("costs", "No cost or slippage model: results ignore trading costs")

    def _check_data(self, config: BacktestConfig, out: _Issues) -> None:
        if config.data.universe is not None:
            self._resolve(PluginKind.UNIVERSE, config.data.universe, "data.universe", out)
        try:
            registry(PluginKind.DATA_SOURCE).get(config.data.source)
        except PluginNotFoundError:
            out.err("data.source", f"Unknown data source '{config.data.source}'")
        if not config.data.instruments and config.data.universe is None:
            out.err("data.instruments", "Choose at least one instrument or a universe")

    @staticmethod
    def _resolve(
        kind: PluginKind, ref: PluginRef, fld: str, out: _Issues
    ) -> PluginSpec[Any] | None:
        try:
            spec = registry(kind).get(ref.name)
        except PluginNotFoundError:
            out.err(fld, f"Unknown {kind} '{ref.name}'")
            return None
        if ref.version and ref.version != spec.version:
            out.err(
                fld,
                f"{kind} '{ref.name}' version {ref.version} requested, {spec.version} installed",
            )
        try:
            spec.validate_params(ref.params)
        except ValidationError as exc:
            for e in exc.errors():
                loc = ".".join(str(x) for x in e["loc"])
                out.err(f"{fld}.params.{loc}" if loc else f"{fld}.params", e["msg"])
            return None
        return spec

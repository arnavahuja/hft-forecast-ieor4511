"""Generic plugin registry.

There is exactly one :class:`Registry` per :class:`PluginKind`. These registries are the only
module-level singletons in the code base. Plugins register themselves with the
:func:`register` decorator when their module is imported; see :mod:`hft_forecast.core.discovery`
for how modules are found.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Generic, TypeVar, cast

from hft_forecast.core.errors import DuplicatePluginError, PluginError, PluginNotFoundError
from hft_forecast.core.params import PluginParams

T = TypeVar("T")


class PluginKind(StrEnum):
    """Kinds of plugins."""

    DATA_SOURCE = "data_source"
    STRATEGY = "strategy"
    PORTFOLIO_CONSTRUCTOR = "portfolio_constructor"
    OVERLAY = "overlay"
    COST_MODEL = "cost_model"
    SLIPPAGE_MODEL = "slippage_model"
    FILL_MODEL = "fill_model"
    METRIC = "metric"
    CHART = "chart"
    UNIVERSE = "universe"


@dataclass(frozen=True)
class PluginSpec(Generic[T]):
    """Registered plugin metadata.

    Attributes:
        kind: Plugin kind.
        name: Unique name within the kind.
        version: Semantic version string.
        description: One-paragraph description (defaults to the class docstring).
        tags: Free-form tags for filtering in the UI.
        capabilities: Declared capabilities, e.g. ``engine:vectorized``.
        cls: The plugin class.
        module: Module the class was defined in.
        origin: ``builtin``, ``user`` or ``entrypoint``.
        extra: Kind-specific metadata (e.g. metric descriptors).
    """

    kind: PluginKind
    name: str
    version: str
    description: str
    tags: tuple[str, ...]
    capabilities: frozenset[str]
    cls: type[T]
    module: str
    origin: str = "builtin"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def params_model(self) -> type[PluginParams]:
        """The plugin's pydantic ``Params`` model."""
        model: type[PluginParams] = getattr(self.cls, "Params", PluginParams)
        return model

    def params_schema(self) -> dict[str, Any]:
        """JSON Schema for the plugin parameters."""
        return self.params_model.json_schema()

    def validate_params(self, params: dict[str, Any] | None) -> PluginParams:
        """Validate raw params into the plugin's ``Params`` model."""
        return self.params_model.model_validate(params or {})

    def create(self, params: dict[str, Any] | PluginParams | None = None, **deps: Any) -> T:
        """Instantiate the plugin with validated parameters.

        Args:
            params: Raw or validated parameters.
            **deps: Extra constructor dependencies (e.g. ``env`` for data sources).
        """
        validated = params if isinstance(params, PluginParams) else self.validate_params(params)
        return self.cls(validated, **deps)  # type: ignore[call-arg]  # plugins take (params)


class Registry(Generic[T]):
    """Registry of plugins of one kind."""

    def __init__(self, kind: PluginKind) -> None:
        self.kind = kind
        self._specs: dict[str, PluginSpec[T]] = {}

    def add(self, spec: PluginSpec[T]) -> None:
        """Add a spec. Re-registering the same class (module reload) replaces it.

        Raises:
            DuplicatePluginError: If a different class already uses the name.
        """
        existing = self._specs.get(spec.name)
        if existing is not None and (
            existing.module != spec.module or existing.cls.__qualname__ != spec.cls.__qualname__
        ):
            raise DuplicatePluginError(
                f"Duplicate {self.kind} plugin name '{spec.name}': defined in "
                f"'{existing.module}' and '{spec.module}'",
                details={"kind": str(self.kind), "name": spec.name},
            )
        self._specs[spec.name] = spec

    def get(self, name: str) -> PluginSpec[T]:
        """Look up a plugin by name."""
        try:
            return self._specs[name]
        except KeyError:
            raise PluginNotFoundError(
                f"No {self.kind} plugin named '{name}'",
                details={"kind": str(self.kind), "available": sorted(self._specs)},
            ) from None

    def specs(self) -> list[PluginSpec[T]]:
        """All specs sorted by name."""
        return [self._specs[k] for k in sorted(self._specs)]

    def names(self) -> list[str]:
        """Sorted plugin names."""
        return sorted(self._specs)

    def remove_module(self, module: str) -> list[str]:
        """Unregister every plugin defined in ``module``. Returns removed names."""
        removed = [n for n, s in self._specs.items() if s.module == module]
        for name in removed:
            del self._specs[name]
        return removed

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __iter__(self) -> Iterator[PluginSpec[T]]:
        return iter(self.specs())

    def __len__(self) -> int:
        return len(self._specs)


REGISTRIES: dict[PluginKind, Registry[Any]] = {kind: Registry(kind) for kind in PluginKind}
"""The per-kind registries (the only allowed module-level singletons)."""


def registry(kind: PluginKind | str) -> Registry[Any]:
    """Return the registry for a kind."""
    try:
        return REGISTRIES[PluginKind(kind)]
    except ValueError:
        raise PluginError(f"Unknown plugin kind '{kind}'") from None


def _origin_of(module: str) -> str:
    if module.startswith("user_plugins"):
        return "user"
    if module.startswith("hft_forecast."):
        return "builtin"
    return "entrypoint"


def register(
    kind: PluginKind | str,
    *,
    name: str,
    version: str = "1.0.0",
    description: str | None = None,
    tags: tuple[str, ...] | list[str] = (),
    capabilities: set[str] | frozenset[str] | None = None,
    **extra: Any,
) -> Callable[[type[T]], type[T]]:
    """Class decorator registering a plugin.

    Args:
        kind: Plugin kind.
        name: Unique name within the kind.
        version: Semantic version.
        description: Description; defaults to the class docstring.
        tags: Tags for filtering.
        capabilities: Capabilities; defaults to the class's ``capabilities`` attribute.
        **extra: Kind-specific metadata stored on the spec.

    Returns:
        The decorator.
    """
    plugin_kind = PluginKind(kind)

    def decorator(cls: type[T]) -> type[T]:
        from hft_forecast.core.interfaces import BASE_FOR_KIND

        base = BASE_FOR_KIND[plugin_kind]
        if not (inspect.isclass(cls) and issubclass(cast(type, cls), base)):
            raise PluginError(
                f"Plugin '{name}' must subclass {base.__name__} to register as {plugin_kind}"
            )
        caps = capabilities if capabilities is not None else getattr(cls, "capabilities", set())
        doc = description or inspect.cleandoc(cls.__doc__ or "").split("\n\n")[0]
        spec: PluginSpec[T] = PluginSpec(
            kind=plugin_kind,
            name=name,
            version=version,
            description=doc.replace("\n", " "),
            tags=tuple(tags),
            capabilities=frozenset(caps),
            cls=cls,
            module=cls.__module__,
            origin=_origin_of(cls.__module__),
            extra=dict(extra),
        )
        REGISTRIES[plugin_kind].add(spec)
        setattr(cls, "plugin_name", name)  # noqa: B010 - back-reference on arbitrary T
        setattr(cls, "plugin_version", version)  # noqa: B010
        return cls

    return decorator

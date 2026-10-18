"""Plugin loading, description and hot reload (composition root for the registries)."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Final

import structlog

from hft_forecast.core.discovery import (
    DiscoveryReport,
    discover_all,
    load_user_file,
    unload_user_file,
)
from hft_forecast.core.registry import REGISTRIES, PluginKind, PluginSpec, registry

log = structlog.get_logger(__name__)

BUILTIN_PACKAGES: Final = [
    "hft_forecast.data.sources",
    "hft_forecast.data.universes",
    "hft_forecast.strategies",
    "hft_forecast.portfolio.construction",
    "hft_forecast.portfolio.overlays",
    "hft_forecast.engine.costs",
    "hft_forecast.engine.fills",
    "hft_forecast.analytics.metrics",
    "hft_forecast.analytics.charts",
]


def describe_spec(spec: PluginSpec[Any]) -> dict[str, Any]:
    """JSON-serializable description of a plugin (used by the API and CLI)."""
    cls = spec.cls
    implements: list[str] = []
    for flag, label in (("implements_vectorized", "vectorized"), ("implements_event", "event")):
        fn = getattr(cls, flag, None)
        if callable(fn) and fn():
            implements.append(label)
    extra = dict(spec.extra)
    for attr in ("title", "group", "category", "output_kind"):
        value = getattr(cls, attr, None)
        if isinstance(value, str) and value:
            extra.setdefault(attr, str(value))
    scopes = getattr(cls, "scopes", None)
    if scopes:
        extra.setdefault("scopes", sorted(scopes))
    descriptors = getattr(cls, "descriptors", None)
    if descriptors:
        extra.setdefault("descriptors", [d.model_dump(mode="json") for d in descriptors])
    return {
        "kind": str(spec.kind),
        "name": spec.name,
        "version": spec.version,
        "description": spec.description,
        "tags": list(spec.tags),
        "capabilities": sorted(spec.capabilities),
        "origin": spec.origin,
        "module": spec.module,
        "implements": implements,
        "params_schema": spec.params_schema(),
        "extra": extra,
    }


class PluginService:
    """Loads plugins once and optionally hot-reloads ``user_plugins/``.

    Args:
        user_dir: User plugin directory.
        builtin_packages: Packages scanned for built-in plugins.
    """

    def __init__(self, user_dir: Path | None, builtin_packages: list[str] | None = None) -> None:
        self._user_dir = user_dir
        self._packages = list(builtin_packages or BUILTIN_PACKAGES)
        self._report = DiscoveryReport()
        self._loaded = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._watcher: threading.Thread | None = None

    @property
    def report(self) -> DiscoveryReport:
        """Discovery report (failures appear on the plugin health panel)."""
        return self._report

    def load(self) -> DiscoveryReport:
        """Discover all plugins (idempotent)."""
        with self._lock:
            if not self._loaded:
                self._report = discover_all(self._packages, self._user_dir)
                self._loaded = True
        return self._report

    def specs(self, kind: PluginKind | str | None = None) -> list[PluginSpec[Any]]:
        """Registered plugins, optionally of one kind."""
        self.load()
        if kind is not None:
            return registry(kind).specs()
        return [spec for reg in REGISTRIES.values() for spec in reg.specs()]

    def get(self, kind: PluginKind | str, name: str) -> PluginSpec[Any]:
        """Look up one plugin."""
        self.load()
        return registry(kind).get(name)

    def health(self) -> dict[str, Any]:
        """Counts per kind and import failures."""
        self.load()
        return {
            "counts": {str(k): len(r) for k, r in REGISTRIES.items()},
            "loaded_modules": len(self._report.loaded_modules),
            "failures": [
                {"module": f.module, "origin": f.origin, "error": f.error, "traceback": f.traceback}
                for f in self._report.failures
            ],
        }

    # ---- hot reload

    def reload_user_file(self, path: Path) -> None:
        """Reload one user plugin file (or unload it if deleted)."""
        name = f"user_plugins.{path.stem}"
        with self._lock:
            self._report.failures = [f for f in self._report.failures if f.module != name]
            if path.exists():
                report = load_user_file(path)
                self._report.failures.extend(report.failures)
            else:
                unload_user_file(path)
        log.info("user_plugin_reloaded", path=str(path))

    def start_watcher(self) -> None:
        """Watch ``user_plugins/`` in a daemon thread and hot-reload changed files."""
        if self._user_dir is None or not self._user_dir.is_dir() or self._watcher is not None:
            return
        directory = self._user_dir

        def _watch() -> None:
            from watchfiles import watch

            for changes in watch(directory, stop_event=self._stop, recursive=False):
                for _change, raw in changes:
                    path = Path(raw)
                    if path.suffix == ".py" and not path.name.startswith("_"):
                        try:
                            self.reload_user_file(path)
                        except Exception as exc:
                            log.warning("hot_reload_failed", path=raw, error=str(exc))

        self._watcher = threading.Thread(target=_watch, name="plugin-watcher", daemon=True)
        self._watcher.start()

    def stop_watcher(self) -> None:
        """Stop the hot reload thread."""
        self._stop.set()

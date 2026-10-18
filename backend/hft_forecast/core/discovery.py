"""Plugin discovery: built-in packages, the ``user_plugins`` directory and entry points."""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import pkgutil
import sys
import traceback
import types
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path

import structlog

from hft_forecast.core.errors import DuplicatePluginError
from hft_forecast.core.registry import REGISTRIES

log = structlog.get_logger(__name__)

ENTRY_POINT_GROUP = "hft_forecast.plugins"
USER_PACKAGE = "user_plugins"
_TRACEBACK_LINES = 6


@dataclass(frozen=True)
class PluginFailure:
    """A plugin module that failed to import."""

    module: str
    origin: str
    error: str
    traceback: str


@dataclass
class DiscoveryReport:
    """Result of plugin discovery (surfaced on the UI's plugin health panel)."""

    loaded_modules: list[str] = field(default_factory=list)
    failures: list[PluginFailure] = field(default_factory=list)

    def merge(self, other: DiscoveryReport) -> None:
        """Merge another report into this one."""
        self.loaded_modules.extend(other.loaded_modules)
        self.failures.extend(other.failures)


def _short_tb(exc: BaseException) -> str:
    lines = traceback.format_exception(exc)
    return "".join(lines[-_TRACEBACK_LINES:])


def _import(name: str, origin: str, report: DiscoveryReport) -> None:
    try:
        importlib.import_module(name)
    except DuplicatePluginError:
        raise
    except Exception as exc:
        log.warning("plugin_import_failed", module=name, error=str(exc))
        report.failures.append(
            PluginFailure(name, origin, f"{type(exc).__name__}: {exc}", _short_tb(exc))
        )
    else:
        report.loaded_modules.append(name)


def discover_package(package: str) -> DiscoveryReport:
    """Import every module of a built-in plugin package (recursively)."""
    report = DiscoveryReport()
    try:
        pkg = importlib.import_module(package)
    except Exception as exc:
        report.failures.append(PluginFailure(package, "builtin", str(exc), _short_tb(exc)))
        return report
    for info in pkgutil.walk_packages(pkg.__path__, prefix=f"{package}."):
        if info.name.rsplit(".", 1)[-1].startswith("_"):
            continue
        _import(info.name, "builtin", report)
    return report


def user_module_name(path: Path) -> str:
    """Module name used for a user plugin file."""
    return f"{USER_PACKAGE}.{path.stem}"


def load_user_file(path: Path, report: DiscoveryReport | None = None) -> DiscoveryReport:
    """(Re)load one user plugin file, replacing anything it registered before."""
    report = report or DiscoveryReport()
    name = user_module_name(path)
    for reg in REGISTRIES.values():
        reg.remove_module(name)
    sys.modules.pop(name, None)
    # compile from source every time: cached bytecode could hide an edit made within the
    # same second with the same file size (hot reload)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    try:
        code = compile(path.read_text(), str(path), "exec")
        exec(code, module.__dict__)
    except DuplicatePluginError:
        sys.modules.pop(name, None)
        raise
    except Exception as exc:
        sys.modules.pop(name, None)
        for reg in REGISTRIES.values():
            reg.remove_module(name)
        log.warning("user_plugin_failed", path=str(path), error=str(exc))
        report.failures.append(
            PluginFailure(name, "user", f"{type(exc).__name__}: {exc}", _short_tb(exc))
        )
    else:
        report.loaded_modules.append(name)
    return report


def unload_user_file(path: Path) -> None:
    """Unregister plugins from a deleted user plugin file."""
    name = user_module_name(path)
    for reg in REGISTRIES.values():
        reg.remove_module(name)
    sys.modules.pop(name, None)


def discover_user_dir(directory: Path) -> DiscoveryReport:
    """Load every ``*.py`` file (not starting with ``_``) in the user plugin directory."""
    report = DiscoveryReport()
    if not directory.is_dir():
        return report
    if USER_PACKAGE not in sys.modules:
        pkg_spec = importlib.machinery.ModuleSpec(USER_PACKAGE, None, is_package=True)
        pkg = importlib.util.module_from_spec(pkg_spec)
        pkg.__path__ = [str(directory)]
        sys.modules[USER_PACKAGE] = pkg
    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("_"):
            continue
        load_user_file(path, report)
    return report


def discover_entry_points(group: str = ENTRY_POINT_GROUP) -> DiscoveryReport:
    """Import modules advertised through Python entry points."""
    report = DiscoveryReport()
    for ep in entry_points(group=group):
        _import(ep.module, "entrypoint", report)
    return report


def discover_all(builtin_packages: list[str], user_dir: Path | None) -> DiscoveryReport:
    """Run all discovery steps.

    Raises:
        DuplicatePluginError: If two plugins of the same kind share a name.
    """
    report = DiscoveryReport()
    for package in builtin_packages:
        report.merge(discover_package(package))
    report.merge(discover_entry_points())
    if user_dir is not None:
        report.merge(discover_user_dir(user_dir))
    log.info(
        "plugins_discovered", modules=len(report.loaded_modules), failures=len(report.failures)
    )
    return report

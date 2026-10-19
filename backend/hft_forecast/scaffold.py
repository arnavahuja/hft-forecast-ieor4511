"""Scaffolding: ``hftf new <kind> <name>`` writes a plugin file and a test from templates."""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path
from typing import Final

from hft_forecast.core.errors import ConfigError

KINDS: Final = (
    "strategy",
    "overlay",
    "metric",
    "chart",
    "data_source",
    "cost_model",
    "portfolio_constructor",
    "model",
    "feature_set",
)
ML_KINDS: Final = ("model", "feature_set")
"""Kinds registered in the ml registries rather than in ``core.registry``."""
NAME_PATTERN: Final = re.compile(r"^[a-z][a-z0-9_]*$")


def _template(name: str) -> str:
    return resources.files("hft_forecast").joinpath("templates", name).read_text()


def scaffold(kind: str, name: str, directory: Path, overwrite: bool = False) -> list[Path]:
    """Create ``<directory>/<name>.py`` and ``<directory>/tests/test_<name>.py``.

    Raises:
        ConfigError: For an unknown kind, an invalid name or an existing file.
    """
    if kind not in KINDS:
        raise ConfigError(f"Unknown kind '{kind}'", details={"kinds": list(KINDS)})
    if not NAME_PATTERN.match(name):
        raise ConfigError("Name must be snake_case (letters, digits, underscores)")
    cls = "".join(part.capitalize() for part in name.split("_"))
    title = name.replace("_", " ").capitalize()
    plugin = directory / f"{name}.py"
    test = directory / "tests" / f"test_{name}.py"
    for path in (plugin, test):
        if path.exists() and not overwrite:
            raise ConfigError(f"{path} already exists")
    directory.mkdir(parents=True, exist_ok=True)
    test.parent.mkdir(parents=True, exist_ok=True)
    plugin.write_text(_template(f"{kind}.py.tmpl").format(name=name, cls=cls, title=title))
    template = "test_ml_plugin.py.tmpl" if kind in ML_KINDS else "test_plugin.py.tmpl"
    test.write_text(_template(template).format(name=name, kind=kind))
    return [plugin, test]

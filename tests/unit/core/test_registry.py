from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import Field

from hft_forecast.core import Strategy, StrategyParams, register
from hft_forecast.core.discovery import discover_user_dir, load_user_file
from hft_forecast.core.errors import DuplicatePluginError, PluginError, PluginNotFoundError
from hft_forecast.core.registry import PluginKind, registry


@pytest.fixture
def clean_strategy_names():
    reg = registry(PluginKind.STRATEGY)
    before = set(reg.names())
    yield
    for name in set(reg.names()) - before:
        spec = reg.get(name)
        reg.remove_module(spec.module)


def test_register_exposes_schema_and_capabilities(clean_strategy_names):
    @register("strategy", name="_test_dummy", version="0.1.0", tags=["test"])
    class Dummy(Strategy):
        """A dummy strategy.

        Longer description that should not appear.
        """

        class Params(StrategyParams):
            window: int = Field(10, ge=1, description="Window")

    spec = registry("strategy").get("_test_dummy")
    assert spec.version == "0.1.0"
    assert spec.description == "A dummy strategy."
    schema = spec.params_schema()
    assert schema["properties"]["window"]["default"] == 10
    assert "engine:vectorized" in spec.capabilities
    inst = spec.create({"window": 3})
    assert inst.params.window == 3  # type: ignore[attr-defined]


def test_duplicate_name_raises(clean_strategy_names):
    @register("strategy", name="_test_dup")
    class A(Strategy):
        pass

    with pytest.raises(DuplicatePluginError):

        @register("strategy", name="_test_dup")
        class B(Strategy):
            pass


def test_wrong_base_class_rejected():
    with pytest.raises(PluginError):

        @register("strategy", name="_test_bad")
        class NotAStrategy:  # type: ignore[type-var]
            pass


def test_missing_plugin_error_lists_available():
    with pytest.raises(PluginNotFoundError) as info:
        registry("strategy").get("does_not_exist")
    assert "available" in info.value.details


DUMMY = textwrap.dedent(
    '''
    from hft_forecast.core import Strategy, StrategyParams, register

    @register("strategy", name="_user_dummy")
    class UserDummy(Strategy):
        """User dummy."""
    '''
)


def test_user_dir_discovery_and_reload(tmp_path: Path, clean_strategy_names):
    (tmp_path / "dummy.py").write_text(DUMMY)
    (tmp_path / "broken.py").write_text("raise RuntimeError('boom')\n")
    report = discover_user_dir(tmp_path)
    assert "user_plugins.dummy" in report.loaded_modules
    assert [f.module for f in report.failures] == ["user_plugins.broken"]
    spec = registry("strategy").get("_user_dummy")
    assert spec.origin == "user"
    # reloading the same file is idempotent (same module + class)
    load_user_file(tmp_path / "dummy.py")
    assert "_user_dummy" in registry("strategy")

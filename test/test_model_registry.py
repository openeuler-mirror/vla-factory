"""Model registry extension and external plugin discovery."""

import pytest

from vla_factory.model.model_interface import ModelMetadata
from vla_factory.model.registry import (
    ModelEntry,
    ModelRegistry,
    RegistryLoadError,
    get_entry,
)


def _factory(recipe, assembly):
    return "plugin-model"


@pytest.fixture(autouse=True)
def _restore_registry_state():
    ModelRegistry._ensure_builtins_loaded()
    entries = dict(ModelRegistry._entries)
    plugins = set(ModelRegistry._plugins_loaded)
    builtins_loaded = ModelRegistry._builtins_loaded
    builtin_error = ModelRegistry._builtin_error
    yield
    ModelRegistry._entries.clear()
    ModelRegistry._entries.update(entries)
    ModelRegistry._plugins_loaded.clear()
    ModelRegistry._plugins_loaded.update(plugins)
    ModelRegistry._builtins_loaded = builtins_loaded
    ModelRegistry._builtin_error = builtin_error


def test_external_model_entry_is_discovered_by_name(monkeypatch):
    from vla_factory.model import registry

    class EntryPoint:
        name = "_plugin-model"

        @staticmethod
        def load():
            return ModelEntry(ModelMetadata(name="_plugin-model"), _factory)

    monkeypatch.setattr(
        registry,
        "entry_points",
        lambda *, group: (EntryPoint(),)
        if group == ModelRegistry.ENTRY_POINT_GROUP else (),
    )

    entry = get_entry("_plugin-model")
    assert entry.factory(None, None) == "plugin-model"


def test_plugin_name_must_match_its_metadata(monkeypatch):
    from vla_factory.model import registry

    class EntryPoint:
        name = "_plugin-model"

        @staticmethod
        def load():
            return ModelEntry(ModelMetadata(name="different-name"), _factory)

    monkeypatch.setattr(
        registry,
        "entry_points",
        lambda *, group: (EntryPoint(),)
        if group == ModelRegistry.ENTRY_POINT_GROUP else (),
    )

    with pytest.raises(RegistryLoadError, match="metadata"):
        get_entry("_plugin-model")

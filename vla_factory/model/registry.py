"""Model registration, built-in adapter loading, and plugin discovery."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points
from typing import Any

from .model_interface import ModelMetadata, VLAModel


ModelFactory = Callable[[Any, Any], VLAModel]


class RegistryLoadError(RuntimeError):
    """One or more model adapters could not be imported safely."""


@dataclass(frozen=True)
class ModelEntry:
    """One registered model family declaration and its construction factory."""

    metadata: ModelMetadata
    factory: ModelFactory


class ModelRegistry:
    """Registry for built-in adapters and external ``vla_factory.models`` plugins."""

    ENTRY_POINT_GROUP = "vla_factory.models"
    _entries: dict[str, ModelEntry] = {}
    _builtins_loaded = False
    _builtin_error: RegistryLoadError | None = None
    _plugins_loaded: set[str] = set()

    @classmethod
    def register(cls, metadata: ModelMetadata):
        """Register a model factory under ``metadata.name``."""
        if not isinstance(metadata, ModelMetadata):
            raise TypeError(
                "ModelRegistry.register expects ModelMetadata, "
                f"got {metadata!r}"
            )
        name = cls._normalise(metadata.name)

        def decorator(factory: ModelFactory) -> ModelFactory:
            if not callable(factory):
                raise TypeError(f"Model factory must be callable, got {factory!r}")
            if name in cls._entries:
                existing = cls._entries[name].factory
                raise ValueError(
                    f"Model {name!r} already registered by "
                    f"{getattr(existing, '__name__', repr(existing))}"
                )
            cls._entries[name] = ModelEntry(metadata=metadata, factory=factory)
            return factory

        return decorator

    @classmethod
    def get(cls, name: str) -> ModelEntry:
        """Return a registered entry, loading built-ins and a named plugin lazily."""
        key = cls._normalise(name)
        cls._ensure_builtins_loaded()
        if key not in cls._entries:
            cls._load_plugin(key)
        if key not in cls._entries:
            available = ", ".join(cls.names()) or "(none)"
            raise KeyError(f"Model {name!r} not registered. Available: {available}")
        return cls._entries[key]

    @classmethod
    def entries(cls) -> dict[str, ModelMetadata]:
        """Return a name-to-metadata snapshot including installed plugins."""
        cls._ensure_builtins_loaded()
        cls._load_all_plugins()
        return {name: entry.metadata for name, entry in cls._entries.items()}

    @classmethod
    def names(cls) -> tuple[str, ...]:
        """Return registered and externally discoverable model names."""
        external = {
            cls._normalise(entry_point.name)
            for entry_point in cls._plugin_entry_points()
        }
        return tuple(sorted(set(cls._entries) | external))

    @classmethod
    def _ensure_builtins_loaded(cls) -> None:
        if cls._builtins_loaded:
            return
        if cls._builtin_error is not None:
            raise cls._builtin_error

        import vla_factory.model.adapters as adapters

        errors: list[str] = []
        for module_info in pkgutil.iter_modules(adapters.__path__):
            module_name = f"vla_factory.model.adapters.{module_info.name}"
            try:
                importlib.import_module(module_name)
            except BaseException as exc:
                errors.append(
                    f"  {module_name}: {type(exc).__name__}: {exc}"
                )
        if errors:
            cls._builtin_error = RegistryLoadError(
                "Failed to load built-in model adapters:\n" + "\n".join(errors)
            )
            raise cls._builtin_error
        cls._builtins_loaded = True

    @classmethod
    def _load_plugin(cls, name: str) -> None:
        matches = [
            entry_point
            for entry_point in cls._plugin_entry_points()
            if cls._normalise(entry_point.name) == name
        ]
        if len(matches) > 1:
            raise RegistryLoadError(
                f"Multiple external model plugins are registered as {name!r}"
            )
        if matches:
            cls._install_plugin(matches[0])

    @classmethod
    def _load_all_plugins(cls) -> None:
        errors: list[str] = []
        for entry_point in cls._plugin_entry_points():
            try:
                cls._install_plugin(entry_point)
            except BaseException as exc:
                errors.append(
                    f"  {entry_point.name}: {type(exc).__name__}: {exc}"
                )
        if errors:
            raise RegistryLoadError(
                "Failed to load external model plugins:\n" + "\n".join(errors)
            )

    @classmethod
    def _install_plugin(cls, entry_point: EntryPoint) -> None:
        name = cls._normalise(entry_point.name)
        if name in cls._plugins_loaded:
            return
        existing = cls._entries.get(name)
        if existing is not None:
            raise RegistryLoadError(f"Model {name!r} is already registered")
        loaded = entry_point.load()

        # Importing the entry point may execute @register_vla. Alternatively,
        # an external package may expose a fully constructed ModelEntry.
        registered = cls._entries.get(name)
        if registered is None and isinstance(loaded, ModelEntry):
            if cls._normalise(loaded.metadata.name) != name:
                raise RegistryLoadError(
                    f"Plugin {name!r} returned metadata for "
                    f"{loaded.metadata.name!r}"
                )
            cls.register(loaded.metadata)(loaded.factory)
            registered = cls._entries.get(name)
        if registered is None:
            raise RegistryLoadError(
                f"Plugin {name!r} did not register a model with the same name"
            )
        cls._plugins_loaded.add(name)

    @classmethod
    def _plugin_entry_points(cls) -> tuple[EntryPoint, ...]:
        return tuple(entry_points(group=cls.ENTRY_POINT_GROUP))

    @staticmethod
    def _normalise(name: str) -> str:
        key = name.strip().lower()
        if not key:
            raise ValueError("Model name must not be empty")
        return key


def register_vla(metadata: ModelMetadata):
    """Compatibility-friendly decorator backed by :class:`ModelRegistry`."""
    return ModelRegistry.register(metadata)


def get_entry(name: str) -> ModelEntry:
    """Return the registered entry for ``name``."""
    return ModelRegistry.get(name)


def list_entries() -> dict[str, ModelMetadata]:
    """Return a name-to-metadata snapshot of all installed models."""
    return ModelRegistry.entries()


__all__ = [
    "ModelEntry",
    "ModelRegistry",
    "RegistryLoadError",
    "get_entry",
    "list_entries",
    "register_vla",
]

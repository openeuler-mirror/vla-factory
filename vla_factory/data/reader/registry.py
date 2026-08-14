"""Registration and plugin discovery for dataset readers."""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path

from .base import FormatReader


ReaderFactory = Callable[[], FormatReader]


class ReaderRegistry:
    """Name-to-reader registry with optional external entry-point plugins."""

    ENTRY_POINT_GROUP = "vla_factory.readers"
    _factories: dict[str, ReaderFactory] = {}

    @classmethod
    def register(cls, name: str, *, aliases: tuple[str, ...] = ()):
        """Register a reader class or zero-argument factory."""
        names = tuple(cls._normalise(value) for value in (name, *aliases))

        def decorator(factory: ReaderFactory) -> ReaderFactory:
            cls._validate_registration(names, factory)
            for registered_name in names:
                cls._factories[registered_name] = factory
            return factory

        return decorator

    @classmethod
    def create(cls, name: str) -> FormatReader:
        """Construct the reader registered under ``name``."""
        key = cls._normalise(name)
        if key not in cls._factories:
            cls._load_plugin(key)
        try:
            reader = cls._factories[key]()
        except KeyError:
            available = ", ".join(cls.names()) or "(none)"
            raise ValueError(
                f"Unknown dataset format {name!r}. Available: {available}"
            ) from None
        if not isinstance(reader, FormatReader):
            raise TypeError(
                f"Reader {name!r} does not implement FormatReader: {reader!r}"
            )
        return reader

    @classmethod
    def detect(cls, path: Path) -> FormatReader:
        """Return the first registered reader that recognises ``path``."""
        for reader in cls._instances():
            if reader.can_read(path):
                return reader

        # External readers are loaded only when built-ins cannot recognise the
        # path, so an unrelated third-party dependency cannot break built-ins.
        for entry_point in cls._plugin_entry_points():
            cls._install_entry_point(entry_point)
        for reader in cls._instances():
            if reader.can_read(path):
                return reader
        raise ValueError(f"No reader found for dataset path: {path}")

    @classmethod
    def names(cls) -> tuple[str, ...]:
        """Return registered and discoverable external names."""
        plugin_names = {
            cls._normalise(entry_point.name)
            for entry_point in cls._plugin_entry_points()
        }
        return tuple(sorted(set(cls._factories) | plugin_names))

    @classmethod
    def _instances(cls) -> list[FormatReader]:
        readers: list[FormatReader] = []
        seen: set[int] = set()
        for factory in cls._factories.values():
            if id(factory) in seen:
                continue
            seen.add(id(factory))
            reader = factory()
            if not isinstance(reader, FormatReader):
                raise TypeError(f"Registered reader is invalid: {reader!r}")
            readers.append(reader)
        return readers

    @classmethod
    def _load_plugin(cls, name: str) -> None:
        matches = [
            entry_point
            for entry_point in cls._plugin_entry_points()
            if cls._normalise(entry_point.name) == name
        ]
        if len(matches) > 1:
            raise ValueError(
                f"Multiple external readers are registered as {name!r}"
            )
        if matches:
            cls._install_entry_point(matches[0])

    @classmethod
    def _install_entry_point(cls, entry_point: EntryPoint) -> None:
        name = cls._normalise(entry_point.name)
        factory = entry_point.load()
        existing = cls._factories.get(name)
        if existing is not None and existing is not factory:
            raise ValueError(f"Dataset reader {name!r} is already registered")
        if existing is None:
            cls._add(name, factory)

    @classmethod
    def _plugin_entry_points(cls) -> tuple[EntryPoint, ...]:
        return tuple(entry_points(group=cls.ENTRY_POINT_GROUP))

    @classmethod
    def _add(cls, name: str, factory: ReaderFactory) -> None:
        cls._validate_registration((name,), factory)
        cls._factories[name] = factory

    @classmethod
    def _validate_registration(
        cls,
        names: tuple[str, ...],
        factory: ReaderFactory,
    ) -> None:
        if not callable(factory):
            raise TypeError(f"Reader factory must be callable, got {factory!r}")
        if len(set(names)) != len(names):
            raise ValueError(f"Reader registration repeats a name: {names}")
        conflicts = [name for name in names if name in cls._factories]
        if conflicts:
            raise ValueError(
                f"Dataset reader {conflicts[0]!r} is already registered"
            )

    @staticmethod
    def _normalise(name: str) -> str:
        key = name.strip().lower()
        if not key:
            raise ValueError("Reader name must not be empty")
        return key

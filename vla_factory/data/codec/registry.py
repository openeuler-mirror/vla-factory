"""Registration and plugin discovery for video codecs."""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import EntryPoint, entry_points

from .base import VideoCodec


CodecFactory = Callable[[], VideoCodec]


class CodecRegistry:
    """Name-to-codec registry with optional external entry-point plugins."""

    ENTRY_POINT_GROUP = "vla_factory.codecs"
    _factories: dict[str, CodecFactory] = {}

    @classmethod
    def register(cls, name: str, *, aliases: tuple[str, ...] = ()):
        """Register a codec class or zero-argument factory."""
        names = tuple(cls._normalise(value) for value in (name, *aliases))

        def decorator(factory: CodecFactory) -> CodecFactory:
            cls._validate_registration(names, factory)
            for registered_name in names:
                cls._factories[registered_name] = factory
            return factory

        return decorator

    @classmethod
    def create(cls, name: str) -> VideoCodec:
        """Construct the codec registered under ``name``."""
        key = cls._normalise(name)
        if key not in cls._factories:
            cls._load_plugin(key)
        try:
            codec = cls._factories[key]()
        except KeyError:
            available = ", ".join(cls.names()) or "(none)"
            raise ValueError(
                f"Unknown video codec {name!r}. Available: {available}"
            ) from None
        if not isinstance(codec, VideoCodec):
            raise TypeError(
                f"Codec {name!r} does not implement VideoCodec: {codec!r}"
            )
        return codec

    @classmethod
    def names(cls) -> tuple[str, ...]:
        """Return registered and discoverable external names."""
        plugin_names = {
            cls._normalise(entry_point.name)
            for entry_point in cls._plugin_entry_points()
        }
        return tuple(sorted(set(cls._factories) | plugin_names))

    @classmethod
    def _load_plugin(cls, name: str) -> None:
        matches = [
            entry_point
            for entry_point in cls._plugin_entry_points()
            if cls._normalise(entry_point.name) == name
        ]
        if len(matches) > 1:
            raise ValueError(f"Multiple external codecs are registered as {name!r}")
        if matches:
            cls._install_entry_point(matches[0])

    @classmethod
    def _install_entry_point(cls, entry_point: EntryPoint) -> None:
        name = cls._normalise(entry_point.name)
        factory = entry_point.load()
        existing = cls._factories.get(name)
        if existing is not None and existing is not factory:
            raise ValueError(f"Video codec {name!r} is already registered")
        if existing is None:
            cls._add(name, factory)

    @classmethod
    def _plugin_entry_points(cls) -> tuple[EntryPoint, ...]:
        return tuple(entry_points(group=cls.ENTRY_POINT_GROUP))

    @classmethod
    def _add(cls, name: str, factory: CodecFactory) -> None:
        cls._validate_registration((name,), factory)
        cls._factories[name] = factory

    @classmethod
    def _validate_registration(
        cls,
        names: tuple[str, ...],
        factory: CodecFactory,
    ) -> None:
        if not callable(factory):
            raise TypeError(f"Codec factory must be callable, got {factory!r}")
        if len(set(names)) != len(names):
            raise ValueError(f"Codec registration repeats a name: {names}")
        conflicts = [name for name in names if name in cls._factories]
        if conflicts:
            raise ValueError(f"Video codec {conflicts[0]!r} is already registered")

    @staticmethod
    def _normalise(name: str) -> str:
        key = name.strip().lower()
        if not key:
            raise ValueError("Codec name must not be empty")
        return key

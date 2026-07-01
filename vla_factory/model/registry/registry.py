"""Model registry: decorator-based registration + lookup.

Usage::

    from vla_factory.model.registry import register_vla
    from vla_factory.model.protocols import ModelMetadata

    @register_vla(ModelMetadata(name="act", ...))
    def load_act(model_path, action_spec, **kw):
        return ACTModelWrapper(...)

    # Later:
    from vla_factory.model.registry import get_entry
    entry = get_entry("act")
    model = entry.factory(model_path=None, action_spec=action_spec)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from vla_factory.model.protocols.model import ModelMetadata

logger = logging.getLogger(__name__)


class RegistryLoadError(RuntimeError):
    """Raised when an entry module fails to import during registry load.

    A failure here means an entry is broken (syntax error, bug, or a missing
    *hard* dependency) — it must surface as itself, not be silently masked as
    'model not registered'.  Optional-dependency fallbacks belong inside the
    entry, not in the loader.
    """


@dataclass
class ModelEntry:
    """A registered model: its metadata + a factory callable."""

    metadata: ModelMetadata
    factory: Callable  # (recipe, schema) -> VLAModel


# Global registry (module-level singleton)
_REGISTRY: dict[str, ModelEntry] = {}


def register_vla(metadata: ModelMetadata):
    """Decorator that registers a model factory under *metadata.name*."""

    def decorator(factory_fn: Callable) -> Callable:
        if metadata.name in _REGISTRY:
            raise ValueError(
                f"Model '{metadata.name}' already registered. "
                f"Existing: {_REGISTRY[metadata.name].factory.__name__}"
            )
        _REGISTRY[metadata.name] = ModelEntry(
            metadata=metadata, factory=factory_fn
        )
        return factory_fn

    return decorator


def get_entry(name: str) -> ModelEntry:
    """Look up a registered model by name.  Raises KeyError if not found."""
    # Lazy-import entries so that @register_vla decorators execute on first use.
    _ensure_entries_loaded()
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"Model '{name}' not registered. Available: {available}")
    return _REGISTRY[name]


def list_entries() -> dict[str, ModelMetadata]:
    """Return a name→metadata snapshot of all registered models."""
    _ensure_entries_loaded()
    return {name: entry.metadata for name, entry in _REGISTRY.items()}


# ── Internal helpers ───────────────────────────────────────────────

_ENTRIES_LOADED = False


def _ensure_entries_loaded() -> None:
    """Import ``vla_factory.model.registry.entries.*`` once to trigger registrations.

    Every exception from an entry module — including ``ModuleNotFoundError`` —
    is re-raised as :class:`RegistryLoadError`.  Optional-dependency errors
    belong inside the entry (e.g. ``act.py`` defers its lerobot import to
    factory-call time so a missing extra surfaces as a clear ``ImportError``
    rather than breaking registry load), so any error that escapes an entry
    is a real bug: a syntax error, a broken internal import, or a missing
    hard dependency.  Masking it would turn those into a misleading
    ``KeyError: Model '...' not registered``.
    """
    global _ENTRIES_LOADED
    if _ENTRIES_LOADED:
        return
    _ENTRIES_LOADED = True

    import importlib
    import pkgutil

    import vla_factory.model.registry.entries as _pkg

    errors: list[str] = []
    for module_info in pkgutil.iter_modules(_pkg.__path__):
        mod_name = f"vla_factory.model.registry.entries.{module_info.name}"
        try:
            importlib.import_module(mod_name)
        except BaseException as exc:
            errors.append(f"  {mod_name}: {type(exc).__name__}: {exc}")

    if errors:
        raise RegistryLoadError(
            "Failed to load registry entries:\n" + "\n".join(errors)
        )

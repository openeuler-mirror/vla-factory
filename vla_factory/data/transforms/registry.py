"""Transform registry: name → step class + forward/reverse pairing metadata.

Pairing is declared via ``paired_reverse`` on :meth:`TransformRegistry.register`
— no extra data class is required.  Phase 1 uses registration + ``create``; the
pairing metadata is wired now and consumed once the postprocessor split lands
(in a later phase), where ``get_reverse`` drives automatic pre/post separation.
"""

from __future__ import annotations

import inspect

from .base import TransformStep


class TransformRegistry:
    """Step registration + pairing metadata (class-level singleton state)."""

    _steps: dict[str, type[TransformStep]] = {}
    _pairings: dict[str, str] = {}  # forward name → reverse name

    @classmethod
    def register(cls, name: str, *, paired_reverse: str | None = None):
        """Class decorator: register a ``TransformStep`` subclass under *name*.

        Parameters
        ----------
        name
            Registered type name, used in config dicts (``{"type": "normalize"}``).
        paired_reverse
            Optional name of the reverse step that undoes this one
            (e.g. ``"normalize"`` ↔ ``"unnormalize_action"``).  ``None`` for
            forward-only steps (ResizeImages, PadDimensions, ...).
        """

        def decorator(step_cls):
            if not (isinstance(step_cls, type) and issubclass(step_cls, TransformStep)):
                raise TypeError(
                    "@TransformRegistry.register expects a TransformStep subclass, "
                    f"got {step_cls!r}"
                )
            # Stamp the registered name on the class so a step instance can be
            # reverse-looked-up (used to derive a postprocessor from a built
            # preprocessor — see build_transforms).
            step_cls._registry_name = name
            cls._steps[name] = step_cls
            if paired_reverse:
                cls._pairings[name] = paired_reverse
            return step_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[TransformStep]:
        """Look up a registered step class by name."""
        try:
            return cls._steps[name]
        except KeyError:
            available = ", ".join(sorted(cls._steps)) or "(none)"
            raise KeyError(f"Transform '{name}' not registered. Available: {available}")

    @classmethod
    def create(cls, item: dict, **shared) -> TransformStep:
        """Build a step instance from a config dict.

        ``item`` looks like ``{"type": "resize_images", "height": 480, ...}``.
        All keys except ``type`` are passed as kwargs.

        ``shared`` carries objects not stored in the config (e.g.
        ``stats=norm_stats``).  Only the subset a step's ``__init__`` actually
        accepts is forwarded — so a shared ``stats`` reaches ``Normalize`` but
        is silently dropped for self-contained steps like ``ResizeImages``.
        This lets a single ``shared`` payload flow through a whole pipeline
        without every step having to swallow ``**kwargs``.
        """
        if "type" not in item:
            raise KeyError(f"Transform config missing 'type': {item!r}")
        step_cls = cls.get(item["type"])
        params = {k: v for k, v in item.items() if k != "type"}
        return step_cls(**params, **_filter_shared(step_cls, shared))

    @classmethod
    def get_reverse(cls, name: str) -> str | None:
        """Return the paired reverse step name, or ``None`` if forward-only."""
        return cls._pairings.get(name)

    @classmethod
    def name_of(cls, step: TransformStep) -> str | None:
        """Return the registered name of a step instance, or ``None``.

        Relies on the ``_registry_name`` stamp set by :meth:`register`.
        """
        return getattr(step, "_registry_name", None)

    @classmethod
    def names(cls) -> list[str]:
        """Sorted list of all registered step type names."""
        return sorted(cls._steps)


def _accepted_kwargs(step_cls: type) -> set[str] | None:
    """Return the kwargs ``step_cls.__init__`` accepts, or ``None`` to mean "all".

    ``None`` is returned when the constructor accepts ``**kwargs`` (so any shared
    payload may be passed through unchanged).
    """
    try:
        params = inspect.signature(step_cls.__init__).parameters
    except (ValueError, TypeError):
        return None
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return None
    return {name for name in params if name != "self"}


def _filter_shared(step_cls: type, shared: dict) -> dict:
    """Keep only the ``shared`` kwargs ``step_cls.__init__`` accepts."""
    accepted = _accepted_kwargs(step_cls)
    if accepted is None:
        return dict(shared)
    return {k: v for k, v in shared.items() if k in accepted}

"""Transform registry: name → step class."""

from __future__ import annotations

from .base import TransformStep


class TransformRegistry:
    """Step registration (class-level singleton state)."""

    _steps: dict[str, type[TransformStep]] = {}

    @classmethod
    def register(cls, name: str):
        """Class decorator: register a ``TransformStep`` subclass under *name*.

        Parameters
        ----------
        name
            Registered type name, used in config dicts (``{"type": "normalize"}``).
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
    def create_from_config(cls, item: dict, ctx=None) -> TransformStep:
        """Build a step instance through the step's ``from_config`` hook."""
        if "type" not in item:
            raise KeyError(f"Transform config missing 'type': {item!r}")
        step_cls = cls.get(item["type"])
        return step_cls.from_config(item, ctx)

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

"""Registry for fine-tuning strategies."""

from __future__ import annotations

from .base import FinetuningStrategy


class StrategyRegistry:
    """Name-to-strategy registry used by the training orchestration."""

    _strategies: dict[str, FinetuningStrategy] = {}

    @classmethod
    def register(cls, name: str):
        """Register a strategy class under a stable recipe name."""
        key = cls._normalise(name)

        def decorator(strategy_type: type[FinetuningStrategy]):
            if not isinstance(strategy_type, type) or not issubclass(
                strategy_type, FinetuningStrategy
            ):
                raise TypeError(
                    "StrategyRegistry.register expects a FinetuningStrategy class"
                )
            if key in cls._strategies:
                raise ValueError(f"Fine-tuning strategy {key!r} already registered")
            cls._strategies[key] = strategy_type()
            return strategy_type

        return decorator

    @classmethod
    def get(cls, name: str) -> FinetuningStrategy:
        """Return a registered strategy or raise with the available names."""
        key = cls._normalise(name)
        try:
            return cls._strategies[key]
        except KeyError:
            raise ValueError(
                f"Unknown fine-tuning strategy {name!r}. "
                f"Available: {list(cls.names())}"
            ) from None

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._strategies))

    @staticmethod
    def _normalise(name: str) -> str:
        key = name.strip().lower()
        if not key:
            raise ValueError("Fine-tuning strategy name must not be empty")
        return key


def register_strategy(name: str):
    return StrategyRegistry.register(name)


def get_strategy(name: str) -> FinetuningStrategy:
    return StrategyRegistry.get(name)


def list_strategies() -> tuple[str, ...]:
    return StrategyRegistry.names()

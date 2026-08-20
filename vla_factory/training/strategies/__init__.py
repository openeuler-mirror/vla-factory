"""Registered parameter-selection and parameter-efficient strategies."""

from .base import FinetuningStrategy
from .registry import (
    StrategyRegistry,
    get_strategy,
    list_strategies,
    register_strategy,
)

# Import built-ins for registration. Optional heavy dependencies stay lazy
# inside strategy methods, so importing this package remains lightweight.
from . import basic as _basic  # noqa: F401,E402
from . import lora as _lora  # noqa: F401,E402


def apply_strategy(model, recipe, metadata):
    """Prepare ``model`` using the strategy selected by ``recipe``."""
    strategy = get_strategy(recipe.finetuning.strategy)
    config = strategy.parse_config(recipe.finetuning.config)
    return strategy.prepare_model(model, config, metadata)


__all__ = [
    "FinetuningStrategy",
    "StrategyRegistry",
    "apply_strategy",
    "get_strategy",
    "list_strategies",
    "register_strategy",
]

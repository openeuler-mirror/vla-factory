"""Fine-tuning strategies: full, freeze, selective, lora."""

from vla_factory.training.strategies.full import apply_strategy
from vla_factory.training.strategies.lora import apply_lora

__all__ = ["apply_strategy", "apply_lora"]

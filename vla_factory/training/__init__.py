"""VLA Factory training engine.

Exports:
    VLATrainer     — transformers.Trainer subclass for VLA models
    train          — high-level training orchestration function
    apply_strategy — parameter freezing based on recipe strategy
"""

from vla_factory.training.pytorch_trainer import VLATrainer
from vla_factory.training.strategies import apply_strategy

__all__ = ["VLATrainer", "apply_strategy"]

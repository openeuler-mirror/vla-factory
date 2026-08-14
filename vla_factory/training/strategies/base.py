"""Fine-tuning strategy lifecycle and strict configuration parsing."""

from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass
from typing import Any, Generic, TypeVar

import torch.nn as nn

from vla_factory.model.model_interface import ModelMetadata


ConfigT = TypeVar("ConfigT")


class FinetuningStrategy(Generic[ConfigT]):
    """Parameter-selection or parameter-efficient fine-tuning strategy.

    Strategies may prepare a model before training and finalize it before its
    inference state dict is saved. They do not own loss computation, sampling,
    or the training loop; an algorithm that changes those belongs to a future
    training-method abstraction instead.
    """

    config_type: type[ConfigT]

    def parse_config(self, raw: dict[str, Any]) -> ConfigT:
        """Parse a strategy-owned mapping into its strict dataclass config."""
        if not isinstance(raw, dict):
            raise TypeError("finetuning.config must be a mapping")
        if not is_dataclass(self.config_type):
            raise TypeError(
                f"{type(self).__name__}.config_type must be a dataclass type"
            )
        config_fields = {item.name: item for item in fields(self.config_type)}
        unknown = sorted(set(raw) - set(config_fields))
        if unknown:
            raise ValueError(
                f"Unknown config field(s) for strategy {type(self).__name__}: "
                f"{unknown}. Known fields: {sorted(config_fields)}"
            )
        missing = [
            name for name, item in config_fields.items()
            if item.default is MISSING
            and item.default_factory is MISSING
            and name not in raw
        ]
        if missing:
            raise ValueError(
                f"Missing config field(s) for strategy {type(self).__name__}: "
                f"{missing}"
            )
        return self.config_type(**raw)

    def prepare_model(
        self,
        model: nn.Module,
        config: ConfigT,
        metadata: ModelMetadata,
    ) -> nn.Module:
        """Freeze, unfreeze, or wrap the model before training."""
        return model

    def finalize_model(self, model: nn.Module) -> nn.Module:
        """Merge temporary adapters and return the model ready to save."""
        return model

    def state_dict(self, model: nn.Module) -> dict:
        """Return the inference state dict after finalization."""
        return model.state_dict()

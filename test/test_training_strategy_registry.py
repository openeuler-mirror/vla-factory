"""Fine-tuning strategies are isolated, strict, and registry-extensible."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch.nn as nn

from vla_factory.model.model_interface import ModelMetadata
from vla_factory.user_interface import parse_recipe_from_string
from vla_factory.training.strategies import (
    FinetuningStrategy,
    get_strategy,
    list_strategies,
    register_strategy,
)


def test_builtin_strategies_are_registered():
    assert set(list_strategies()) >= {"full", "freeze", "selective", "lora"}


def test_unknown_strategy_reports_available_names():
    with pytest.raises(ValueError, match="Available"):
        get_strategy("does-not-exist")


def test_strategy_config_is_strict():
    with pytest.raises(ValueError, match="Unknown config field"):
        get_strategy("full").parse_config({"unused": True})
    with pytest.raises(TypeError, match="components"):
        get_strategy("freeze").parse_config({"components": "backbone"})


@pytest.mark.parametrize(
    ("name", "config", "backbone_trainable", "head_trainable"),
    [
        ("full", {}, True, True),
        ("freeze", {"components": ["backbone"]}, False, True),
        ("selective", {"components": ["head"]}, False, True),
    ],
)
def test_basic_strategies_apply_declared_components(
    name, config, backbone_trainable, head_trainable
):
    model = nn.Module()
    model.backbone = nn.Linear(2, 2)
    model.head = nn.Linear(2, 2)
    metadata = ModelMetadata(
        name="test",
        components={"backbone": ("backbone.",), "head": ("head.",)},
    )
    strategy = get_strategy(name)

    strategy.prepare_model(model, strategy.parse_config(config), metadata)

    assert all(
        p.requires_grad is backbone_trainable
        for p in model.backbone.parameters()
    )
    assert all(p.requires_grad is head_trainable for p in model.head.parameters())


def test_recipe_rejects_strategy_specific_legacy_fields():
    with pytest.raises(ValueError, match="Unknown finetuning field"):
        parse_recipe_from_string(
            """
model: {name: act}
finetuning:
  strategy: freeze
  freeze_components: [backbone]
"""
        )


def test_new_strategy_requires_only_one_registered_class():
    @dataclass(frozen=True)
    class Config:
        enabled: bool

    @register_strategy("test-custom")
    class CustomStrategy(FinetuningStrategy[Config]):
        config_type = Config

        def prepare_model(self, model, config, metadata):
            model.custom_enabled = config.enabled
            return model

    strategy = get_strategy("test-custom")
    config = strategy.parse_config({"enabled": True})
    model = strategy.prepare_model(nn.Linear(2, 2), config, metadata=None)
    assert model.custom_enabled is True

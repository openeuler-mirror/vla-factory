"""L0 error-path tests for the freeze / selective strategies.

These cover the failure mode Issue #7 calls out: a component name that does not
exist in ``ModelMetadata.components`` used to be logged and skipped, so a typo
in a recipe silently changed *which parameters train* — ``freeze`` froze
nothing and trained the whole model, ``selective`` froze everything and trained
nothing. Both runs succeed, both produce a useless checkpoint.

Deliberately model-free: a toy ``nn.Module`` plus a hand-built ModelMetadata,
so the whole matrix runs in the dependency-light CI tier (no lerobot, no peft).
The LoRA counterparts live in ``test_lora_strategy.py``, next to its fake-peft
fixture.
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from vla_factory.model import ModelMetadata
from vla_factory.training.strategies import get_strategy
from vla_factory.user_interface import FinetuningConfig, TrainRecipe


class _ToyModel(nn.Module):
    """Two named subtrees so component prefixes have something to match."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)


_METADATA = ModelMetadata(
    name="toy",
    components={
        "backbone": ["backbone."],
        "head": ["head."],
    },
)


def _recipe(strategy, *, freeze=None, trainable=None):
    config = {}
    if freeze is not None:
        config["components"] = freeze
    if trainable is not None:
        config["components"] = trainable
    return TrainRecipe(
        finetuning=FinetuningConfig(strategy=strategy, config=config),
    )


def _apply_strategy(model, recipe, metadata):
    strategy = get_strategy(recipe.finetuning.strategy)
    config = strategy.parse_config(recipe.finetuning.config)
    return strategy.prepare_model(model, config, metadata)


# ── Unknown component names ──────────────────────────────────────────


@pytest.mark.parametrize("strategy,kwargs", [
    ("freeze", {"freeze": ["backbon"]}),          # typo'd
    ("selective", {"trainable": ["action_head"]}),  # plausible but undeclared
])
def test_unknown_component_raises_and_lists_available(strategy, kwargs):
    """A component name not in metadata.components must fail, not warn."""
    model = _ToyModel()
    with pytest.raises(ValueError) as exc:
        _apply_strategy(model, _recipe(strategy, **kwargs), _METADATA)

    message = str(exc.value)
    assert strategy in message
    # The message must name the offender and enumerate the valid choices —
    # that is what turns a silent misconfiguration into a fixable one.
    assert "backbone" in message and "head" in message
    # The model must be left untouched when resolution fails.
    assert all(p.requires_grad for p in model.parameters())


def test_unknown_component_among_valid_ones_still_raises():
    """One bad name in an otherwise valid list must not be silently dropped."""
    model = _ToyModel()
    with pytest.raises(ValueError, match="typo"):
        _apply_strategy(model, _recipe("freeze", freeze=["backbone", "typo"]), _METADATA)


# ── Empty component lists ────────────────────────────────────────────


def test_freeze_with_no_components_raises():
    """strategy='freeze' with an empty list would train everything."""
    model = _ToyModel()
    with pytest.raises(ValueError, match="freeze_components is empty"):
        _apply_strategy(model, _recipe("freeze", freeze=[]), _METADATA)


def test_selective_with_no_components_raises():
    """strategy='selective' with an empty list would train nothing."""
    model = _ToyModel()
    with pytest.raises(ValueError, match="trainable_components is empty"):
        _apply_strategy(model, _recipe("selective", trainable=[]), _METADATA)


def test_component_declaring_no_prefixes_raises():
    """A declared-but-empty component resolves to no patterns — also a no-op."""
    metadata = ModelMetadata(name="toy", components={"ghost": []})
    model = _ToyModel()
    with pytest.raises(ValueError, match="no parameter-name prefixes"):
        _apply_strategy(model, _recipe("freeze", freeze=["ghost"]), metadata)


# ── Happy paths (the error checks must not break normal use) ─────────


def test_freeze_freezes_only_the_named_component():
    model = _ToyModel()
    _apply_strategy(model, _recipe("freeze", freeze=["backbone"]), _METADATA)

    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    assert frozen == {"backbone.weight", "backbone.bias"}


def test_selective_trains_only_the_named_component():
    model = _ToyModel()
    _apply_strategy(model, _recipe("selective", trainable=["head"]), _METADATA)

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"head.weight", "head.bias"}


def test_unknown_strategy_raises():
    model = _ToyModel()
    with pytest.raises(ValueError, match="Unknown fine-tuning strategy"):
        _apply_strategy(model, _recipe("finetune_everything"), _METADATA)

"""Tests for the LoRA strategy (peft injection, target_components → subtree).

Patches peft with a fake module so tests run without peft installed (the
strategy imports peft lazily inside apply_lora). Also runnable directly via
`python test/test_lora_strategy.py`.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch.nn as nn

from vla_factory.recipe.parser import parse_recipe_from_string
from vla_factory.recipe import FinetuningConfig, TrainRecipe
from vla_factory.training.strategies import get_strategy
from vla_factory.training.strategies.lora import (
    merge_lora_adapters,
)


# ── Fake peft: record the subtree passed to get_peft_model ──
_FAKE_PEFT = types.ModuleType("peft")
_FAKE_PEFT.__spec__ = importlib.util.spec_from_loader("peft", loader=None)  # satisfies find_spec
_injected: list = []


class _FakeLoraConfig:
    def __init__(self, **kw):
        self.kw = kw


class _FakePeftModel(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def merge_and_unload(self):
        return self.module


def _fake_get_peft_model(module, cfg):
    _injected.append((module, cfg))
    return _FakePeftModel(module)


_FAKE_PEFT.LoraConfig = _FakeLoraConfig
_FAKE_PEFT.PeftModel = _FakePeftModel
_FAKE_PEFT.PeftMixedModel = _FakePeftModel  # transformers imports this at top level
_FAKE_PEFT.get_peft_model = _fake_get_peft_model


class _FakeLoraLayer(nn.Module):
    """Stand-in for peft.tuners.lora.LoraLayer (merge helpers isinstance-check it)."""


# peft.tuners.lora is imported by train._merge_lora_layers_inplace; provide it.
_FAKE_PEFT_TUNERS = types.ModuleType("peft.tuners")
_FAKE_PEFT_TUNERS.__spec__ = importlib.util.spec_from_loader("peft.tuners", loader=None)
_FAKE_PEFT_TUNERS_LORA = types.ModuleType("peft.tuners.lora")
_FAKE_PEFT_TUNERS_LORA.__spec__ = importlib.util.spec_from_loader("peft.tuners.lora", loader=None)
_FAKE_PEFT_TUNERS_LORA.LoraLayer = _FakeLoraLayer
_FAKE_PEFT_TUNERS.lora = _FAKE_PEFT_TUNERS_LORA
_FAKE_PEFT.tuners = _FAKE_PEFT_TUNERS


@pytest.fixture(autouse=True)
def _fake_peft_installed():
    """Install the fake peft for the duration of each test, then restore."""
    fakes = {
        "peft": _FAKE_PEFT,
        "peft.tuners": _FAKE_PEFT_TUNERS,
        "peft.tuners.lora": _FAKE_PEFT_TUNERS_LORA,
    }
    originals = {name: sys.modules.get(name) for name in fakes}
    sys.modules.update(fakes)
    _injected.clear()
    yield
    for name, original in originals.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


class _FakePaliGemma(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.o_proj = nn.Linear(8, 8)


class _FakeExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)


class _FakePI0(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.paligemma = _FakePaliGemma()
        self.paligemma_with_expert.gemma_expert = _FakeExpert()


# A metadata-shaped object (avoid importing real ModelMetadata fields).
class _Meta:
    name = "pi0"
    support_lora = True
    components = {
        "llm": ["paligemma_with_expert.paligemma."],
        "action_expert": ["paligemma_with_expert.gemma_expert."],
    }


class _MetaNoLora(_Meta):
    support_lora = False


def _make_recipe(targets):
    return TrainRecipe(
        finetuning=FinetuningConfig(
            strategy="lora",
            config={
                "r": 16,
                "lora_alpha": 16,
                "target_components": list(targets),
            },
        ),
    )


def _prepare(model, recipe, metadata):
    strategy = get_strategy(recipe.finetuning.strategy)
    config = strategy.parse_config(recipe.finetuning.config)
    return strategy.prepare_model(model, config, metadata)


def test_single_subtree_target_wraps_only_that_subtree():
    model = _FakePI0()
    out = _prepare(model, _make_recipe(["llm"]), _Meta())
    assert out is model, "returned model must be the same top-level object"
    assert len(_injected) == 1, "exactly one subtree injected"
    assert isinstance(_injected[0][0], _FakePaliGemma), "injected subtree is the paligemma (llm)"
    # parent's leaf should now be the peft-wrapped version
    assert isinstance(model.paligemma_with_expert.paligemma, _FakePeftModel)
    # action expert untouched
    assert not isinstance(model.paligemma_with_expert.gemma_expert, _FakePeftModel)


def test_multi_target_wraps_whole_model():
    model = _FakePI0()
    out = _prepare(model, _make_recipe(["llm", "action_expert"]), _Meta())
    assert isinstance(out, _FakePeftModel), "multi-target → whole-model wrap"
    assert len(_injected) == 1, "single get_peft_model call on the whole model"


def test_merge_unwraps_whole_model_wrap():
    """Whole-model LoRA: the top-level PeftModel itself must be merged away."""
    model = _FakePI0()
    wrapped = _prepare(model, _make_recipe(["llm", "action_expert"]), _Meta())
    assert isinstance(wrapped, _FakePeftModel)

    merged = merge_lora_adapters(wrapped)
    assert merged is model, "top-level PeftModel merged back to its base model"
    assert not any(
        isinstance(m, _FakePeftModel) for m in merged.modules()
    ), "no PeftModel wrapper may survive the merge"


def test_merge_unwraps_subtree_wrap():
    """Subtree-LoRA: the PeftModel child is replaced by its merged base."""
    model = _FakePI0()
    _prepare(model, _make_recipe(["llm"]), _Meta())
    assert isinstance(model.paligemma_with_expert.paligemma, _FakePeftModel)

    merged = merge_lora_adapters(model)
    assert merged is model
    assert isinstance(model.paligemma_with_expert.paligemma, _FakePaliGemma)
    assert not any(isinstance(m, _FakePeftModel) for m in merged.modules())


def test_empty_target_components_raises():
    recipe = TrainRecipe(
        finetuning=FinetuningConfig(
            strategy="lora",
            config={"r": 16, "lora_alpha": 16, "target_components": []},
        ),
    )
    with pytest.raises(ValueError):
        _prepare(_FakePI0(), recipe, _Meta())


def test_support_lora_false_raises():
    with pytest.raises(ValueError):
        _prepare(_FakePI0(), _make_recipe(["llm"]), _MetaNoLora())


def test_unknown_lora_fields_are_rejected_by_strategy():
    recipe = parse_recipe_from_string(
        """
model: {name: pi0}
finetuning:
  strategy: lora
  config:
    rank: 8
    alpha: 8
    target_components: [llm]
"""
    )
    strategy = get_strategy("lora")
    with pytest.raises(ValueError, match="Unknown config field"):
        strategy.parse_config(recipe.finetuning.config)


def test_peft_field_names_forwarded():
    recipe = parse_recipe_from_string(
        """
model: {name: pi0}
finetuning:
  strategy: lora
  config:
    r: 32
    lora_alpha: 32
    lora_dropout: 0.1
    target_components: [llm]
"""
    )
    config = get_strategy("lora").parse_config(recipe.finetuning.config)
    assert config.r == 32
    assert config.lora_alpha == 32
    assert config.lora_dropout == 0.1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

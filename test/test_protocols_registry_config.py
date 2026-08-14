#!/usr/bin/env python3
"""Test: protocols (Observation, VLAModel hierarchy, ModelMetadata),
registry (register_vla, get_entry, list_entries),
config parser (YAML → TrainRecipe, example configs)."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def test_protocols():
    """Protocol layer: Observation, ModelMetadata, VLAModel hierarchy."""
    from vla_factory.model.model_interface import Observation
    from vla_factory.model.model_interface import (
        ModelMetadata, VLAModel, VLAModelPyTorch, VLAModelJAX,
    )


    # Observation (generic over T)
    obs = Observation(images={"front": "tensor"}, image_masks={"front": "mask"}, state="state_tensor")
    assert obs.tokenized_prompt is None  # ACT doesn't use prompt
    print(f"  Observation: images={list(obs.images)}, state={obs.state}")

    # ModelMetadata
    meta = ModelMetadata(
        name="act",
        action_dim=14,
        action_horizon=100,
        action_head_type="regression",
            training_paradigm="from_scratch",
        requires_prompt=False,
    )
    assert meta.support_lora
    assert not meta.requires_prompt
    print(f"  ModelMetadata: name={meta.name}, paradigm={meta.training_paradigm}")

    # ── VLAModel hierarchy: mock implementations ──

    # A PyTorch-style wrapper satisfies both VLAModel and VLAModelPyTorch
    class MockPTModel:
        def compute_loss(self, obs, actions):
            return 0.0
        def predict_actions(self, obs, **kw):
            return []
        def parameters(self):
            return iter([])
        def named_parameters(self):
            return iter([])
        def train(self, mode=True):
            return self
        def to(self, *a, **kw):
            return self

    pt_model = MockPTModel()
    assert isinstance(pt_model, VLAModel), "PT model should satisfy VLAModel"
    assert isinstance(pt_model, VLAModelPyTorch), "PT model should satisfy VLAModelPyTorch"
    print(f"  VLAModelPyTorch: MockPTModel ✓ (isinstance VLAModel={isinstance(pt_model, VLAModel)}, VLAModelPyTorch={isinstance(pt_model, VLAModelPyTorch)})")

    # A JAX-style wrapper satisfies both VLAModel and VLAModelJAX
    class MockJAXModel:
        def __init__(self):
            self._params = {"params": {"w": 1.0}}
            self._apply_fn = lambda *a, **kw: None
        def compute_loss(self, obs, actions):
            return 0.0
        def predict_actions(self, obs, **kw):
            return []
        @property
        def params(self):
            return self._params
        @params.setter
        def params(self, value):
            self._params = value
        @property
        def apply_fn(self):
            return self._apply_fn
        def init_params(self, rng_key, obs, actions):
            return self._params

    jax_model = MockJAXModel()
    assert isinstance(jax_model, VLAModel), "JAX model should satisfy VLAModel"
    assert isinstance(jax_model, VLAModelJAX), "JAX model should satisfy VLAModelJAX"
    # JAX model should NOT satisfy VLAModelPyTorch
    assert not isinstance(jax_model, VLAModelPyTorch), "JAX model must NOT satisfy VLAModelPyTorch"
    print(f"  VLAModelJAX:     MockJAXModel ✓ (isinstance VLAModel={isinstance(jax_model, VLAModel)}, VLAModelJAX={isinstance(jax_model, VLAModelJAX)}, VLAModelPyTorch={isinstance(jax_model, VLAModelPyTorch)})")

    print("  [PASS] protocols")


def test_registry():
    """Registry: register_vla, get_entry, list_entries."""
    from vla_factory.model.model_interface import ModelMetadata
    from vla_factory.model.registry import (
        ModelRegistry,
        get_entry,
        list_entries,
        register_vla,
    )

    # Register a mock model
    @register_vla(ModelMetadata(
        name="__test_mock",
        action_dim=6,
        action_horizon=50,
        action_head_type="regression",
    ))
    def load_mock(recipe, assembly):
        return "mock_model"

    # Lookup
    entry = get_entry("__test_mock")
    assert entry.metadata.name == "__test_mock"
    model = entry.factory(recipe=None, assembly=None)
    assert model == "mock_model"

    # List
    names = list_entries()
    assert "__test_mock" in names

    # Duplicate registration should raise
    try:
        @register_vla(ModelMetadata(name="__test_mock"))
        def dup(): pass
        assert False, "Should have raised"
    except ValueError as e:
        assert "already registered" in str(e)

    # Unknown model should raise
    try:
        get_entry("__nonexistent")
        assert False, "Should have raised"
    except KeyError as e:
        assert "__nonexistent" in str(e)

    # Cleanup
    del ModelRegistry._entries["__test_mock"]

    print("  [PASS] registry")


def test_config_parser():
    """Config: parse YAML → TrainRecipe."""
    from vla_factory.recipe.parser import parse_recipe_from_string
    from vla_factory.recipe import TrainRecipe

    # Minimal config
    minimal = "model:\n  name: act"
    recipe = parse_recipe_from_string(minimal)
    assert recipe.model.name == "act"
    assert recipe.model.path is None
    assert recipe.finetuning.strategy == "full"  # default
    print(f"  Minimal: model={recipe.model.name}, strategy={recipe.finetuning.strategy}")

    # Full ACT config
    act_yaml = """
model:
  name: act
  path: null

data:
  path: /data/lift_cube
  format: lerobot-v3

finetuning:
  strategy: full

training:
  backend: pytorch
  lr: 1e-4
  lr_backbone: 1e-5
  batch_size: 8
  total_steps: 50000

output:
  output_dir: outputs/act_aloha
"""
    recipe = parse_recipe_from_string(act_yaml)
    assert recipe.model.name == "act"
    assert recipe.data.format == "lerobot-v3"
    assert recipe.data.path == "/data/lift_cube"
    assert recipe.finetuning.strategy == "full"
    assert recipe.training.lr == 1e-4
    assert recipe.training.lr_backbone == 1e-5
    assert recipe.output.output_dir == "outputs/act_aloha"
    print(f"  Full ACT: model={recipe.model.name}, lr={recipe.training.lr}, steps={recipe.training.total_steps}")

    # LoRA config
    lora_yaml = """
model:
  name: pi0
  path: checkpoints/pi0_base

finetuning:
  strategy: lora
  config:
    r: 16
    lora_alpha: 16
    target_components: ["llm", "action_expert"]

training:
  lr: 2.5e-5
  batch_size: 64

output:
  output_dir: outputs/pi0_lora
"""
    recipe = parse_recipe_from_string(lora_yaml)
    assert recipe.model.name == "pi0"
    assert recipe.model.path == "checkpoints/pi0_base"
    assert recipe.finetuning.strategy == "lora"
    assert recipe.finetuning.config == {
        "r": 16,
        "lora_alpha": 16,
        "target_components": ["llm", "action_expert"],
    }
    print(
        f"  LoRA: model={recipe.model.name}, "
        f"r={recipe.finetuning.config['r']}, "
        f"targets={recipe.finetuning.config['target_components']}"
    )

    print("  [PASS] config parser")


def test_yaml_files():
    """Parse the actual YAML config files from examples/."""
    from vla_factory.recipe.parser import parse_recipe

    examples_dir = Path(__file__).resolve().parent.parent / "examples"
    if not examples_dir.exists():
        print("  [SKIP] no examples/ directory")
        return

    for yaml_path in sorted(examples_dir.glob("*.yaml")):
        recipe = parse_recipe(yaml_path)
        assert recipe.model.name, f"{yaml_path.name}: model name is empty"
        print(f"  {yaml_path.name}: model={recipe.model.name}, strategy={recipe.finetuning.strategy}, "
              f"lr={recipe.training.lr}, steps={recipe.training.total_steps}")

    print("  [PASS] yaml files")


def main():
    print("=" * 60)
    print("Phase 1 Verification")
    print("=" * 60)
    test_protocols()
    test_registry()
    test_config_parser()
    test_yaml_files()
    print("=" * 60)
    print("All Phase 1 checks passed!")


if __name__ == "__main__":
    main()

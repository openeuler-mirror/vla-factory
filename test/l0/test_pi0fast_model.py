"""Tests for the pi0fast adapter (lerobot PI0FastPolicy, thin composition).

Patches the entry's lerobot handle so the core tests run without lerobot
installed; the factory-path test additionally needs ``lerobot.configs.types``
and is skipped when lerobot is absent (mirrors test_act_model.py).
Runnable both via pytest and directly: `python test/l0/test_pi0fast_model.py`.
"""
from __future__ import annotations

import sys
import types

import pytest
import torch
import torch.nn as nn

import vla_factory.model.adapters.pi0fast as pi0fast_mod
from vla_factory.model.model_interface import Observation
from vla_factory.model.registry import get_entry


# ── Fake lerobot pi0_fast (so the wrapper runs without lerobot installed) ──


class _FakeCheckpointable(nn.Module):
    """Stand-in for an HF sub-backbone found by name suffix."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Linear(1, 1)
        self.checkpointing_enabled = False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.checkpointing_enabled = True


class _FakePI0FastPytorch(nn.Module):
    """Stand-in for the inner PI0FastPytorch (gradient-checkpointing target)."""

    def __init__(self):
        super().__init__()
        # named as the real backbone so the wrapper's suffix lookup finds it
        # (peft subtree wrapping is what makes hardcoded paths unreliable)
        self.language_model = _FakeCheckpointable()
        self.vision_tower = _FakeCheckpointable()


class _FakePI0FastPolicy(nn.Module):
    """Minimal PI0FastPolicy stand-in: forward / predict_action_chunk."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.model = _FakePI0FastPytorch()
        self.seen_batches: list[dict] = []

    def forward(self, batch):
        self.seen_batches.append(batch)
        loss = torch.zeros((), requires_grad=True)
        return loss, {"loss": 0.0, "ce_loss": 0.0}

    def predict_action_chunk(self, batch, **kwargs):
        self.seen_batches.append(batch)
        return torch.zeros(batch["observation.language.tokens"].shape[0], 50, 32)


class _FakeActionTokenizerStep:
    """Minimal ActionTokenizerProcessorStep stand-in."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls: list[torch.Tensor] = []

    def _tokenize_action(self, action: torch.Tensor):
        self.calls.append(action)
        b = action.shape[0]
        return (
            torch.zeros(b, 256, dtype=torch.long),
            torch.ones(b, 256, dtype=torch.bool),
        )


class _FakePI0FastConfig:
    """PI0FastConfig stand-in: records kwargs, carries tokenizer names."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.action_tokenizer_name = "lerobot/fast-action-tokenizer"
        self.max_action_tokens = 256
        self.fast_skip_tokens = 128
        self.text_tokenizer_name = "google/paligemma-3b-pt-224"


pi0fast_mod._try_import_lerobot_pi0fast._cached = (
    _FakePI0FastPolicy,       # PI0FastPolicy
    _FakePI0FastConfig,       # PI0FastConfig
    _FakeActionTokenizerStep,  # ActionTokenizerProcessorStep
)


def _make_wrapper(camera_mapping=None):
    return pi0fast_mod.PI0FASTModelWrapper(
        _FakePI0FastPolicy(),
        camera_mapping=(
            {"base_0_rgb": "front", "left_wrist_0_rgb": "wrist"}
            if camera_mapping is None else camera_mapping
        ),
        action_tokenizer_step=_FakeActionTokenizerStep(),
    )


def _make_obs(B=2):
    return Observation(
        images={
            "front": torch.rand(B, 3, 224, 224),
            "wrist": torch.rand(B, 3, 224, 224),
        },
        image_masks={
            "front": torch.ones(B, dtype=torch.bool),
            "wrist": torch.ones(B, dtype=torch.bool),
        },
        state=torch.zeros(B, 9),
        task=["pick the banana"] * B,
        tokenized_prompt=torch.zeros(B, 200, dtype=torch.long),
        tokenized_prompt_mask=torch.ones(B, 200, dtype=torch.long),
    )


# ── Metadata ─────────────────────────────────────────────────────────


def test_metadata():
    meta = get_entry("pi0fast").metadata
    assert meta.name == "pi0fast"
    assert meta.backend == "pytorch"
    assert meta.action_head_type == "autoregressive"
    assert meta.training_paradigm == "pretrained_finetune"
    assert meta.action_dim == 32
    assert meta.action_horizon == 50
    assert meta.support_lora is True
    # pi05-style discrete-state prompt + quantile normalization
    assert meta.prompt_includes_state is True
    # ...but the FAST prefix ends at ";\n": lerobot's action tokenizer puts
    # "<bos>Action: " at the head of the action segment itself, so a marker in
    # the prompt would duplicate it out of distribution.
    assert meta.prompt_action_marker is False
    assert meta.vector_normalization == "quantile"
    # FAST images are [0,1] (policy converts to [-1,1] internally)
    assert meta.image_input_range == (0.0, 1.0)
    # no action expert in the FAST family
    assert "action_expert" not in meta.components
    assert "llm" in meta.components


# ── Prompt format ────────────────────────────────────────────────────


def test_fast_prefix_ends_at_newline():
    """The FAST discrete-state prefix carries no "Action: " marker — that
    marker belongs to the action segment (openpi FASTTokenizer / lerobot's
    _tokenize_action prepends "<bos>Action: " itself)."""
    import numpy as np

    from vla_factory.assembly.transform.task_tokenize import build_prompt

    state = np.zeros(9, dtype=np.float32)
    prompt = build_prompt("pick the object", state, action_marker=False)
    assert prompt.startswith("Task: pick the object, State: ")
    assert prompt.endswith(";\n")
    assert "Action" not in prompt
    # pi05 keeps the marker (the openpi PaligemmaTokenizer convention)
    assert build_prompt("t", state).endswith(";\nAction: ")


# ── Batch translation ────────────────────────────────────────────────


def test_batch_translation_maps_cameras_and_tokens():
    wrapper = _make_wrapper()
    batch = wrapper._to_lerobot_batch(_make_obs())
    # mapped slots land under their lerobot image keys
    assert "observation.images.base_0_rgb" in batch
    assert "observation.images.left_wrist_0_rgb" in batch
    # unmapped slot is simply absent — the policy fills its own placeholder
    assert "observation.images.right_wrist_0_rgb" not in batch
    # framework-tokenized discrete-state prompt passes through untouched
    assert torch.equal(
        batch["observation.language.tokens"], _make_obs().tokenized_prompt
    )
    assert "observation.language.attention_mask" in batch


def test_training_batch_fast_encodes_actions():
    wrapper = _make_wrapper()
    actions = torch.rand(2, 50, 32)
    batch = wrapper._to_lerobot_batch(_make_obs(), actions=actions)
    assert "action.tokens" in batch
    assert "action.token_mask" in batch
    # the FAST tokenizer saw exactly the normalized/padded chunk
    assert wrapper._action_step.calls[-1] is actions


def test_missing_prompt_raises():
    wrapper = _make_wrapper()
    obs = _make_obs()
    obs.tokenized_prompt = None
    with pytest.raises(ValueError, match="tokenized_prompt"):
        wrapper._to_lerobot_batch(obs)


def test_actions_without_action_step_raise():
    wrapper = pi0fast_mod.PI0FASTModelWrapper(
        _FakePI0FastPolicy(),
        camera_mapping={"base_0_rgb": "front"},
        action_tokenizer_step=None,
    )
    with pytest.raises(ValueError, match="action tokenizer"):
        wrapper._to_lerobot_batch(_make_obs(), actions=torch.rand(2, 50, 32))


# ── Loss / predict delegation ────────────────────────────────────────


def test_loss_and_predict_delegate_to_upstream():
    wrapper = _make_wrapper()
    obs = _make_obs()
    loss, loss_dict = wrapper.compute_loss(obs, torch.rand(2, 50, 32))
    assert loss.dim() == 0
    assert "loss" in loss_dict and "ce_loss" in loss_dict
    pred = wrapper.predict_actions(obs)
    assert isinstance(pred, torch.Tensor)
    assert pred.shape == (2, 50, 32)
    # inference batch carries no action tokens (the model generates them)
    last = wrapper.model.seen_batches[-1]
    assert "action.tokens" not in last


def test_gradient_checkpointing_delegates_to_inner_model():
    wrapper = _make_wrapper()
    wrapper.gradient_checkpointing_enable()
    # the name-suffix lookup found and enabled both sub-backbones
    assert wrapper.model.model.language_model.checkpointing_enabled is True
    assert wrapper.model.model.vision_tower.checkpointing_enabled is True
    assert wrapper.model.model.gradient_checkpointing_enabled is True


# ── Factory path (needs lerobot for FeatureType/PolicyFeature) ───────


def _lerobot_available():
    try:
        from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _lerobot_available(), reason="lerobot not installed")
def test_factory_builds_wrapper_from_assembly():
    from helpers import make_assembly, make_schema
    from vla_factory.user_interface import (
        AssemblyOverrides,
        ModelConfig,
        TrainRecipe,
        merge_model_config,
    )

    # The schema cameras carry no semantic, so the mapping must come from the
    # controlled override — exactly how examples/pi0fast_lora.yaml pins
    # front→base_0_rgb / wrist→left_wrist_0_rgb.
    recipe = merge_model_config(TrainRecipe(
        model=ModelConfig(name="pi0fast"),
    ))
    schema = make_schema(
        state_dim=9,
        action_dim=14,
        cameras=("front", "wrist"),
        image_sizes={"front": (224, 224), "wrist": (224, 224)},
        has_language=True,
    )
    assembly = make_assembly(
        schema, "pi0fast", recipe=recipe,
        overrides=AssemblyOverrides(
            camera_mapping={"base_0_rgb": "front", "left_wrist_0_rgb": "wrist"},
        ),
    )

    # The cached handle is the fake triple, so the factory wires the fake
    # policy (path=None → direct construction, no weights, no downloads).
    wrapper = pi0fast_mod.load_pi0fast(recipe, assembly)
    assert isinstance(wrapper, pi0fast_mod.PI0FASTModelWrapper)
    assert wrapper._camera_mapping.get("base_0_rgb") == "front"
    assert wrapper._action_step is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

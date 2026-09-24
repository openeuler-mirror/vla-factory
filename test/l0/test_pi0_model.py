"""Tests for the pi0 adapter (lerobot 0.5 PI0Policy, thin composition).

Patches the entry's lerobot handle so the core tests run without lerobot
installed; the factory-path test additionally needs ``lerobot.configs.types``
and is skipped when lerobot is absent (mirrors test_pi0fast_model.py).
Runnable both via pytest and directly: `python test/l0/test_pi0_model.py`.
"""
from __future__ import annotations

import sys
import types

import pytest
import torch
import torch.nn as nn

import vla_factory.model.adapters.lerobot_pi as lerobot_pi_mod
from vla_factory.model.model_interface import Observation
from vla_factory.model.registry import get_entry


# ── Fake lerobot pi0 (so the wrapper runs without lerobot installed) ──


class _FakeCheckpointable(nn.Module):
    """Stand-in for an HF sub-backbone found by name suffix."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Linear(1, 1)
        self.checkpointing_enabled = False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.checkpointing_enabled = True


class _FakeGemmaInner(nn.Module):
    """PaliGemma's ``.model`` — hosts the vision tower + language model."""

    def __init__(self):
        super().__init__()
        self.vision_tower = _FakeCheckpointable()
        self.language_model = _FakeCheckpointable()


class _FakePaliGemma(nn.Module):
    """Named so the wrapper's suffix lookup finds the real leaves."""

    def __init__(self):
        super().__init__()
        self.model = _FakeGemmaInner()


class _FakePaliGemmaWithExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma = _FakePaliGemma()
        self.gemma_expert = _FakeCheckpointable()  # flow-matching action expert


class _FakePI0Pytorch(nn.Module):
    """Stand-in for the inner PI0Pytorch (paligemma_with_expert layout).

    Every level is an nn.Module: the wrapper's checkpointing pass reaches the
    leaves through ``named_modules()``, which does not descend into
    SimpleNamespace attributes.
    """

    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = _FakePaliGemmaWithExpert()


class _FakePI0Policy(nn.Module):
    """Minimal PI0Policy stand-in: forward / predict_action_chunk."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.model = _FakePI0Pytorch()
        self.seen_batches: list[dict] = []

    def forward(self, batch):
        self.seen_batches.append(batch)
        loss = torch.zeros((), requires_grad=True)
        return loss, {"loss": 0.0}

    def predict_action_chunk(self, batch, **kwargs):
        self.seen_batches.append(batch)
        return torch.zeros(batch["observation.language.tokens"].shape[0], 50, 32)


class _FakePI0Config:
    """PI0Config stand-in: records the kwargs the factory passes upstream."""

    def __init__(self, **kwargs):
        self.kw = kwargs


lerobot_pi_mod._try_import_lerobot_pi0._cached = (
    _FakePI0Policy,  # PI0Policy
    _FakePI0Config,  # PI0Config
)


def _make_wrapper(camera_mapping=None):
    return lerobot_pi_mod.PILerobotModelWrapper(
        _FakePI0Policy(),
        camera_mapping=(
            {"base_0_rgb": "front", "left_wrist_0_rgb": "wrist"}
            if camera_mapping is None else camera_mapping
        ),
        include_state=True,
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
        tokenized_prompt=torch.zeros(B, 48, dtype=torch.long),
        tokenized_prompt_mask=torch.ones(B, 48, dtype=torch.long),
    )


# ── Metadata ─────────────────────────────────────────────────────────


def test_metadata():
    meta = get_entry("pi0").metadata
    assert meta.name == "pi0"
    assert meta.backend == "pytorch"
    assert meta.action_head_type == "flow_matching"
    assert meta.training_paradigm == "pretrained_finetune"
    assert meta.action_dim == 32
    assert meta.action_horizon == 50
    assert meta.support_lora is True
    assert "llm" in meta.components and "action_expert" in meta.components
    # lerobot _preprocess_images takes [0,1] and maps to [-1,1] internally
    assert meta.image_input_range == (0.0, 1.0)
    # lerobot nesting: wrapper.model (PI0Policy) → .model (PI0Pytorch) → …
    assert meta.components["llm"] == ["model.model.paligemma_with_expert.paligemma."]
    assert meta.components["action_expert"] == [
        "model.model.paligemma_with_expert.gemma_expert."
    ]
    # pi0: continuous state input, plain task prompt
    assert meta.prompt_includes_state is False
    assert meta.vector_normalization == "mean_std"


# ── Batch translation ────────────────────────────────────────────────


def test_batch_translation_maps_cameras_and_tokens():
    wrapper = _make_wrapper()
    batch = wrapper._to_lerobot_batch(_make_obs())
    # mapped slots land under their lerobot image keys
    assert "observation.images.base_0_rgb" in batch
    assert "observation.images.left_wrist_0_rgb" in batch
    # unmapped slot is simply absent — the policy fills its own placeholder
    assert "observation.images.right_wrist_0_rgb" not in batch
    # framework-tokenized prompt passes through untouched, mask cast to bool
    assert torch.equal(
        batch["observation.language.tokens"], _make_obs().tokenized_prompt
    )
    assert batch["observation.language.attention_mask"].dtype is torch.bool
    # pi0 feeds the continuous state vector
    assert "observation.state" in batch


def test_training_batch_carries_action():
    wrapper = _make_wrapper()
    actions = torch.rand(2, 50, 32)
    batch = wrapper._to_lerobot_batch(_make_obs(), actions=actions)
    assert torch.equal(batch["action"], actions)


def test_missing_prompt_raises():
    wrapper = _make_wrapper()
    obs = _make_obs()
    obs.tokenized_prompt = None
    with pytest.raises(ValueError, match="tokenized_prompt"):
        wrapper._to_lerobot_batch(obs)


# ── Loss / predict delegation ────────────────────────────────────────


def test_loss_and_predict_delegate_to_upstream():
    wrapper = _make_wrapper()
    obs = _make_obs()
    loss, loss_dict = wrapper.compute_loss(obs, torch.rand(2, 50, 32))
    assert loss.dim() == 0
    assert "loss" in loss_dict
    pred = wrapper.predict_actions(obs)
    assert isinstance(pred, torch.Tensor)
    assert pred.shape == (2, 50, 32)
    # training batch carried action + state; the inference batch must not
    training = wrapper.model.seen_batches[0]
    assert "action" in training and "observation.state" in training
    inference = wrapper.model.seen_batches[-1]
    assert "action" not in inference


def test_predict_actions_swallows_num_steps_kwarg():
    # the inference engine passes num_steps=; lerobot reads the step count
    # from config.num_inference_steps, so the kwarg must not blow up.
    wrapper = _make_wrapper()
    pred = wrapper.predict_actions(_make_obs(), num_steps=4)
    assert pred.shape == (2, 50, 32)


# ── Gradient checkpointing (name-suffix lookup, three leaves) ────────


def test_gradient_checkpointing_delegates_to_inner_model():
    wrapper = _make_wrapper()
    wrapper.gradient_checkpointing_enable()
    inner = wrapper.model.model.paligemma_with_expert
    assert inner.paligemma.model.vision_tower.checkpointing_enabled is True
    assert inner.paligemma.model.language_model.checkpointing_enabled is True
    assert inner.gemma_expert.checkpointing_enabled is True
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

    recipe = merge_model_config(TrainRecipe(model=ModelConfig(name="pi0")))
    schema = make_schema(
        state_dim=9,
        action_dim=14,
        cameras=("front", "wrist"),
        image_sizes={"front": (224, 224), "wrist": (224, 224)},
        has_language=True,
    )
    assembly = make_assembly(
        schema, "pi0", recipe=recipe,
        overrides=AssemblyOverrides(
            camera_mapping={"base_0_rgb": "front", "left_wrist_0_rgb": "wrist"},
        ),
    )

    # The cached handle is the fake pair, so the factory wires the fake
    # policy (path=None → direct construction, no weights, no downloads).
    wrapper = lerobot_pi_mod.load_lerobot_pi(
        recipe, assembly, _FakePI0Policy, _FakePI0Config, "pi0",
    )
    assert isinstance(wrapper, lerobot_pi_mod.PILerobotModelWrapper)
    assert wrapper._camera_mapping.get("base_0_rgb") == "front"
    # pi0 feeds the continuous state
    assert wrapper._include_state is True
    # composition-derived shapes reach the upstream config
    config = wrapper.model.config
    assert config.kw["chunk_size"] == config.kw["n_action_steps"] == 50
    assert config.kw["dtype"] == "bfloat16"
    assert config.kw["num_inference_steps"] == 10
    # ALL declared slots enter input_features — mapped or not. A slot left out
    # here never reaches config.image_features, so the policy generates no
    # -1-image/zero-mask placeholder for it and the image-token sequence would
    # shrink against the pretrained checkpoint (right_wrist is the trailing
    # unmapped slot in this mapping, hence allowed).
    assert set(config.kw["input_features"]) == {
        "observation.images.base_0_rgb",
        "observation.images.left_wrist_0_rgb",
        "observation.images.right_wrist_0_rgb",
    }


@pytest.mark.skipif(not _lerobot_available(), reason="lerobot not installed")
def test_factory_rejects_non_trailing_unmapped_slot():
    """lerobot appends missing-slot placeholders at the END of the image
    sequence; an unmapped middle slot would shift later real cameras into the
    wrong pretrained position, so the factory refuses instead."""
    from helpers import make_assembly, make_schema
    from vla_factory.user_interface import (
        AssemblyOverrides,
        ModelConfig,
        TrainRecipe,
        merge_model_config,
    )

    recipe = merge_model_config(TrainRecipe(model=ModelConfig(name="pi0")))
    schema = make_schema(
        state_dim=9,
        action_dim=14,
        cameras=("front", "wrist"),
        image_sizes={"front": (224, 224), "wrist": (224, 224)},
        has_language=True,
    )
    # left_wrist_0_rgb (declaration slot 2 of 3) deliberately unmapped.
    assembly = make_assembly(
        schema, "pi0", recipe=recipe,
        overrides=AssemblyOverrides(
            camera_mapping={"base_0_rgb": "front", "right_wrist_0_rgb": "wrist"},
        ),
    )
    with pytest.raises(ValueError, match="trail"):
        lerobot_pi_mod.load_lerobot_pi(
            recipe, assembly, _FakePI0Policy, _FakePI0Config, "pi0",
        )


@pytest.mark.skipif(not _lerobot_available(), reason="lerobot not installed")
def test_factory_routes_pretrained_through_strict_loader(monkeypatch):
    """``model.path`` must load through ``load_pretrained_strict``, never the
    family's ``from_pretrained`` — whose lerobot overrides print-and-continue
    on every load failure and would silently train random weights."""
    from helpers import make_assembly, make_schema
    from vla_factory.user_interface import (
        AssemblyOverrides,
        ModelConfig,
        TrainRecipe,
        merge_model_config,
    )

    recipe = merge_model_config(
        TrainRecipe(model=ModelConfig(name="pi0", path="/tmp/fake-ckpt"))
    )
    schema = make_schema(
        state_dim=9,
        action_dim=14,
        cameras=("front", "wrist"),
        image_sizes={"front": (224, 224), "wrist": (224, 224)},
        has_language=True,
    )
    assembly = make_assembly(
        schema, "pi0", recipe=recipe,
        overrides=AssemblyOverrides(
            camera_mapping={"base_0_rgb": "front", "left_wrist_0_rgb": "wrist"},
        ),
    )
    strict_calls: list[tuple] = []
    monkeypatch.setattr(
        lerobot_pi_mod, "load_pretrained_strict",
        lambda policy, path, name: strict_calls.append((path, name)),
    )
    lerobot_pi_mod.load_lerobot_pi(
        recipe, assembly, _FakePI0Policy, _FakePI0Config, "pi0",
    )
    assert strict_calls == [("/tmp/fake-ckpt", "pi0")]


# ── load_pretrained_strict: every failure must raise ─────────────────


class _RecordingPolicy(nn.Module):
    """Records load_state_dict calls; optionally fails like torch on a key
    mismatch."""

    def __init__(self, fail_message=None):
        super().__init__()
        self.dummy = nn.Linear(1, 1)
        self.config = types.SimpleNamespace()
        self.load_calls: list[dict] = []
        self._fail_message = fail_message

    def load_state_dict(self, state_dict, strict=False):
        self.load_calls.append({"keys": set(state_dict), "strict": strict})
        if self._fail_message:
            raise RuntimeError(self._fail_message)
        return [], []


def test_load_pretrained_strict_applies_model_prefix_and_strict(monkeypatch):
    """Happy path: file resolved + read, bare keys get the "model." prefix,
    and the load is strict."""
    monkeypatch.setattr(
        "transformers.utils.cached_file",
        lambda path, filename: f"/cache/{path}/{filename}",
    )
    monkeypatch.setattr(
        "safetensors.torch.load_file",
        lambda resolved: {"bare.weight": torch.zeros(1), "model.kept.weight": torch.ones(1)},
    )
    policy = _RecordingPolicy()
    lerobot_pi_mod.load_pretrained_strict(policy, "fake/repo", "pi0")
    assert policy.load_calls == [{
        "keys": {"model.bare.weight", "model.kept.weight"},
        "strict": True,
    }]


def test_load_pretrained_strict_propagates_key_mismatch(monkeypatch):
    """The reviewer's core scenario: an incompatible checkpoint (key
    mismatch) must raise, not print-and-continue with random weights."""
    monkeypatch.setattr(
        "transformers.utils.cached_file",
        lambda path, filename: f"/cache/{path}/{filename}",
    )
    monkeypatch.setattr(
        "safetensors.torch.load_file",
        lambda resolved: {"wrong.layout.weight": torch.zeros(1)},
    )
    policy = _RecordingPolicy(fail_message="Missing key(s) in state_dict")
    with pytest.raises(RuntimeError, match="Missing key"):
        lerobot_pi_mod.load_pretrained_strict(policy, "fake/repo", "pi0")
    assert policy.load_calls[0]["strict"] is True


def test_load_pretrained_strict_propagates_unresolvable_path(monkeypatch):
    """A path that cannot be resolved must raise (cached_file's own error),
    never return a silently-untrained model."""
    def _raise(path, filename):
        raise FileNotFoundError(f"no such checkpoint: {path}")

    monkeypatch.setattr("transformers.utils.cached_file", _raise)
    with pytest.raises(FileNotFoundError, match="no such checkpoint"):
        lerobot_pi_mod.load_pretrained_strict(_RecordingPolicy(), "/nope", "pi0")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

"""Tests for the pi05 adapter and its data-side differences from pi0.

The model-side half patches the shared lerobot handle
(adapters/lerobot_pi._try_import_lerobot_pi05) so tests run without lerobot
installed. The data-side half (discrete-state prompt, quantile
normalization) runs everywhere — those transforms are framework code.
Runnable both via pytest and directly: `python test/l0/test_pi05_model.py`.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import torch
import torch.nn as nn

from helpers import make_assembly, make_schema

import vla_factory.model.adapters.lerobot_pi as lerobot_pi_mod
from vla_factory.user_interface import merge_model_config, parse_recipe_from_string
from vla_factory.data.data_schema import FeatureStats, NormStats
from vla_factory.assembly.transform.normalize import (
    NormalizeVector,
    UnnormalizeActionQuantileStep,
)
from vla_factory.assembly.transform.task_tokenize import TaskTokenize, build_prompt
from vla_factory.model.model_interface import Observation
from vla_factory.model.registry import get_entry


def _assembly_for(recipe, model_name: str):
    """Resolve the composition these factory tests build their model from.

    ``make_assembly`` takes overrides as its own parameter (it does not read
    them off the recipe), so pass the recipe's overrides through — the
    factory rejects a camera mapping with no data source, and without this
    the override never reaches the resolver.
    """
    schema = make_schema(
        state_dim=9, action_dim=9, cameras=("front",),
        image_sizes={"front": (224, 224)}, has_language=True,
    )
    return make_assembly(
        schema, model_name, recipe=merge_model_config(recipe),
        overrides=recipe.overrides,
    )


# ── Fakes: record the config kwargs the factory passes upstream ──


class _FakeCheckpointable(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Linear(1, 1)
        self.checkpointing_enabled = False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.checkpointing_enabled = True


class _FakePI05Policy(nn.Module):
    """Minimal PI05Policy stand-in (same block layout as PI0Policy)."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.model = types.SimpleNamespace(paligemma_with_expert=None)
        self.seen_batches: list[dict] = []

    def forward(self, batch):
        self.seen_batches.append(batch)
        return torch.zeros((), requires_grad=True), {"loss": 0.0}

    def predict_action_chunk(self, batch, **kwargs):
        self.seen_batches.append(batch)
        return torch.zeros(batch["observation.language.tokens"].shape[0], 50, 32)


class _FakePI05Config:
    def __init__(self, **kw):
        self.kw = kw


@pytest.fixture()
def _fake_lerobot_pi05():
    original = getattr(lerobot_pi_mod._try_import_lerobot_pi05, "_cached", None)
    lerobot_pi_mod._try_import_lerobot_pi05._cached = (
        _FakePI05Policy, _FakePI05Config,
    )
    yield
    lerobot_pi_mod._try_import_lerobot_pi05._cached = original


# ── Metadata (lerobot-era interface facts) ──


def test_metadata():
    meta = get_entry("pi05").metadata
    assert meta.name == "pi05"
    assert meta.backend == "pytorch"
    assert meta.action_head_type == "flow_matching"
    assert meta.training_paradigm == "pretrained_finetune"
    assert meta.action_dim == 32
    assert meta.support_lora is True
    assert "llm" in meta.components and "action_expert" in meta.components
    # lerobot _preprocess_images takes [0,1] and maps to [-1,1] internally
    assert meta.image_input_range == (0.0, 1.0)
    # lerobot nesting, same block layout as pi0 (two classes, one structure)
    assert meta.components["llm"] == ["model.model.paligemma_with_expert.paligemma."]
    assert meta.components["action_expert"] == [
        "model.model.paligemma_with_expert.gemma_expert."
    ]
    # pi05 data-side differences: quantile + discrete-state prompt with the
    # "Action: " answer marker (openpi PaligemmaTokenizer lineage)
    assert meta.vector_normalization == "quantile"
    assert meta.prompt_includes_state is True
    assert meta.prompt_action_marker is True
    assert meta.tokenizer_max_length == 200


# ── Wrapper state-split (pi05 never receives observation.state) ──


def _make_obs(B=2):
    return Observation(
        images={"front": torch.rand(B, 3, 224, 224)},
        image_masks={"front": torch.ones(B, dtype=torch.bool)},
        state=torch.zeros(B, 9),  # present, but pi05 must not forward it
        task=["pick the banana"] * B,
        tokenized_prompt=torch.zeros(B, 200, dtype=torch.long),
        tokenized_prompt_mask=torch.ones(B, 200, dtype=torch.long),
    )


def test_pi05_batch_omits_state():
    wrapper = lerobot_pi_mod.PILerobotModelWrapper(
        _FakePI05Policy(),
        camera_mapping={"base_0_rgb": "front"},
        include_state=False,
    )
    batch = wrapper._to_lerobot_batch(_make_obs())
    # PI05Policy never reads observation.state — the state rides inside the
    # discrete prompt the framework's task_tokenize already built.
    assert "observation.state" not in batch
    assert "observation.images.base_0_rgb" in batch
    assert batch["observation.language.attention_mask"].dtype is torch.bool


def test_pi0_batch_includes_state():
    wrapper = lerobot_pi_mod.PILerobotModelWrapper(
        _FakePI05Policy(),  # layout irrelevant here; only the flag matters
        camera_mapping={"base_0_rgb": "front"},
        include_state=True,
    )
    assert "observation.state" in wrapper._to_lerobot_batch(_make_obs())


# ── Factory path (needs lerobot for FeatureType/PolicyFeature) ───────


def _lerobot_available():
    try:
        from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _lerobot_available(), reason="lerobot not installed")
def test_factory_builds_pi05_without_state(_fake_lerobot_pi05):
    recipe = parse_recipe_from_string(
        """
model:
  name: pi05
overrides:
  camera_mapping:
    base_0_rgb: front
"""
    )
    wrapper = get_entry("pi05").factory(
        recipe=recipe, assembly=_assembly_for(recipe, "pi05"),
    )
    assert isinstance(wrapper, lerobot_pi_mod.PILerobotModelWrapper)
    assert wrapper._include_state is False, "pi05 state lives in the prompt"
    config = wrapper.model.config
    assert config.kw["chunk_size"] == config.kw["n_action_steps"] == 50
    assert config.kw["dtype"] == "bfloat16"
    assert config.kw["num_inference_steps"] == 10
    # ALL declared slots enter input_features — mapped or not — so the policy's
    # _preprocess_images generates the -1 image + zero mask placeholder for the
    # two unmapped trailing slots (base_0_rgb is the only mapped one here).
    assert set(config.kw["input_features"]) == {
        "observation.images.base_0_rgb",
        "observation.images.left_wrist_0_rgb",
        "observation.images.right_wrist_0_rgb",
    }


@pytest.mark.skipif(not _lerobot_available(), reason="lerobot not installed")
def test_pi05_factory_routes_pretrained_through_strict_loader(
    _fake_lerobot_pi05, monkeypatch,
):
    # Same contract as pi0: model.path loads through load_pretrained_strict —
    # lerobot's from_pretrained overrides print-and-continue on failure.
    recipe = parse_recipe_from_string(
        """
model:
  name: pi05
  path: /tmp/fake-ckpt
overrides:
  camera_mapping:
    base_0_rgb: front
"""
    )
    strict_calls: list[tuple] = []
    monkeypatch.setattr(
        lerobot_pi_mod, "load_pretrained_strict",
        lambda policy, path, name: strict_calls.append((path, name)),
    )
    get_entry("pi05").factory(
        recipe=recipe, assembly=_assembly_for(recipe, "pi05"),
    )
    assert strict_calls == [("/tmp/fake-ckpt", "pi05")]


@pytest.mark.skipif(not _lerobot_available(), reason="lerobot not installed")
def test_pi0_factory_keeps_state_input(_fake_lerobot_pi05):
    # The pi0 factory routes through the same loader with model_name="pi0"
    # — the include_state split must track the model, not the loader.
    recipe = parse_recipe_from_string(
        """
model:
  name: pi0
overrides:
  camera_mapping:
    base_0_rgb: front
"""
    )
    original = getattr(lerobot_pi_mod._try_import_lerobot_pi0, "_cached", None)
    lerobot_pi_mod._try_import_lerobot_pi0._cached = (
        _FakePI05Policy, _FakePI05Config,
    )
    try:
        wrapper = get_entry("pi0").factory(
            recipe=recipe, assembly=_assembly_for(recipe, "pi0"),
        )
    finally:
        lerobot_pi_mod._try_import_lerobot_pi0._cached = original
    assert wrapper._include_state is True


# ── pi05 discrete-state prompt (openpi PaligemmaTokenizer format) ──


def test_build_prompt_discrete_state():
    state = np.array([-1.0, 0.0, 0.999], dtype=np.float32)
    prompt = build_prompt("pick_up the\nobject ", state)
    # openpi cleaning: strip, "_" → " ", "\n" → " "
    assert prompt.startswith("Task: pick up the object, State: ")
    assert prompt.endswith(";\nAction: ")
    # 256 bins over [-1, 1]: -1.0 → 0, 0.0 → 128, 0.999 → 255
    assert "State: 0 128 255;" in prompt


def test_build_prompt_without_state_is_cleaned_task_plus_newline():
    """pi0: cleaned text + the trailing "start of answer" newline.

    The newline mirrors openpi ``models/tokenizer.py:33``, which appends it as
    a separate token. Dropping it leaves the prefix one token short of what the
    base checkpoint was trained with — see
    ``test/l1/test_lerobot_pipeline_parity.py``.
    """
    assert build_prompt(" say_hello ") == "say hello\n"


def test_task_tokenize_discrete_state_requires_state():
    step = TaskTokenize(tokenizer_repo="unused", discrete_state=True, default_task="t")
    with pytest.raises(ValueError, match="state"):
        step({"images.front": np.zeros((3, 2, 2))})


class _FakeTokenizer:
    """Records the text it tokenizes; returns valid prompt arrays."""

    def __init__(self):
        self.seen: list[str] = []

    def __call__(self, text, max_length, padding, truncation, return_tensors):
        self.seen.append(text)
        return {
            "input_ids": np.zeros((1, max_length), dtype=np.int64),
            "attention_mask": np.ones((1, max_length), dtype=np.int64),
        }


def _tokenize_step(**kwargs):
    step = TaskTokenize(tokenizer_repo="unused", **kwargs)
    step._tokenizer = _FakeTokenizer()
    return step


# pi0 prompts carry a trailing "\n" (openpi's start-of-answer token); the
# fallback chain picks the *text*, build_prompt appends the newline.
def test_task_fallback_chain_sample_task_wins():
    step = _tokenize_step(default_task="default")
    step({"task": "from dataset"})
    assert step._tokenizer.seen == ["from dataset\n"]


def test_task_fallback_chain_default_task():
    step = _tokenize_step(default_task="default")
    step({})
    assert step._tokenizer.seen == ["default\n"]


def test_task_fallback_chain_empty_prompt_never_skips():
    # openpi-style final fallback: no task text anywhere still yields prompt
    # fields (PI0Pytorch embeds the prompt unconditionally — the fields must
    # exist), tokenized from the empty string.
    step = _tokenize_step()
    out = step({})
    assert step._tokenizer.seen == ["\n"]
    assert "tokenized_prompt" in out and "tokenized_prompt_mask" in out


# ── quantile normalisation (openpi use_quantile_norm) ──


def _quantile_stats():
    return NormStats(
        state=FeatureStats(q01=[0.0, -2.0], q99=[1.0, 2.0]),
        action=FeatureStats(q01=[0.0, -2.0], q99=[1.0, 2.0]),
        method="quantile",
    )


def test_normalize_vector_quantile():
    step = NormalizeVector(_quantile_stats(), fields=("state", "actions"), method="quantile")
    sample = {
        "state": np.array([0.5, 0.0], dtype=np.float32),
        "actions": np.array([[0.0, -2.0], [1.0, 2.0]], dtype=np.float32),
    }
    out = step(sample)
    # (x - q01) / (q99 - q01 + eps) * 2 - 1
    np.testing.assert_allclose(out["state"], [0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(out["actions"][0], [-1.0, -1.0], atol=1e-5)
    np.testing.assert_allclose(out["actions"][1], [1.0, 1.0], atol=1e-5)


def test_quantile_unnormalize_roundtrip():
    from vla_factory.assembly.transform import TransformContext, TransformRegistry
    from vla_factory.assembly.transform.base import PlanContext

    stats = _quantile_stats()
    args = {"fields": ["actions"], "method": "quantile", "stats_ref": "norm_stats"}
    step = NormalizeVector.from_call(args, TransformContext(norm_stats=stats))
    # The pairing the resolver plans, built the way deployment builds it.
    name, inverse_args = NormalizeVector.inverse_call(
        args, PlanContext(has_action_stats=True),
    )
    inverse = TransformRegistry.get(name).from_call(
        inverse_args, TransformContext(norm_stats=stats),
    )
    assert isinstance(inverse, UnnormalizeActionQuantileStep)
    actions = np.array([[0.3, -1.7], [0.9, 1.4]], dtype=np.float32)
    normalized = step({"actions": actions.copy()})["actions"]
    restored = inverse({"actions": normalized})["actions"]
    np.testing.assert_allclose(restored, actions, atol=1e-4)


def test_quantile_without_stats_fails_early():
    stats = NormStats(state=FeatureStats(mean=[0.0], std=[1.0]))
    step = NormalizeVector(stats, fields=("state",), method="quantile")
    with pytest.raises(ValueError, match="q01/q99"):
        step({"state": np.zeros(1, dtype=np.float32)})


def test_zscore_default_unchanged():
    stats = NormStats(state=FeatureStats(mean=[1.0], std=[2.0]))
    step = NormalizeVector(stats, fields=("state",), method="zscore")
    out = step({"state": np.array([3.0], dtype=np.float32)})
    np.testing.assert_allclose(out["state"], [1.0], atol=1e-5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

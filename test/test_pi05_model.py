"""Tests for the pi05 adapter and its data-side differences from pi0.

Patches the shared openpi handle (entries/pi0._try_import_openpi) so tests run
without openpi/jax installed. Runnable both via pytest and directly:
`python test/test_pi05_model.py`.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch.nn as nn

import vla_factory.model.registry.entries.pi0 as pi0_mod
from vla_factory.config.parser import parse_recipe_from_string
from vla_factory.data.manifest import FeatureStats, NormStats
from vla_factory.data.transforms.normalize import (
    NormalizeVector,
    UnnormalizeActionQuantileStep,
)
from vla_factory.data.transforms.task_tokenize import TaskTokenize, build_prompt
from vla_factory.model.registry import get_entry


# ── Fakes: record the Pi0Config kwargs the factory passes upstream ──


class _FakePi0Config:
    def __init__(self, **kw):
        self.kw = kw
        self.pi05 = kw.get("pi05", False)


class _FakePaligemmaWithExpert(nn.Module):
    def to_bfloat16_for_selected_params(self, dtype):
        pass


class _FakePI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.paligemma_with_expert = _FakePaligemmaWithExpert()
        self.dummy = nn.Linear(1, 1)


@pytest.fixture(autouse=True)
def _fake_openpi():
    original = getattr(pi0_mod._try_import_openpi, "_cached", None)
    pi0_mod._try_import_openpi._cached = (
        _FakePI0Pytorch,
        _FakePi0Config,
        types.SimpleNamespace,
    )
    yield
    pi0_mod._try_import_openpi._cached = original


def test_metadata():
    meta = get_entry("pi05").metadata
    assert meta.name == "pi05"
    assert meta.backend == "pytorch"
    assert meta.action_head_type == "flow_matching"
    assert meta.training_paradigm == "pretrained_finetune"
    assert meta.action_dim == 32
    assert meta.support_lora is True
    assert "llm" in meta.components and "action_expert" in meta.components


def test_factory_builds_pi05_variant():
    recipe = parse_recipe_from_string(
        """
model:
  name: pi05
  config:
    camera_mapping:
      base_0_rgb: front
action_spec:
  action_dim: 9
  action_horizon: 50
"""
    )
    wrapper = get_entry("pi05").factory(recipe=recipe, schema=None)
    config = wrapper.model.config
    assert config.kw["pi05"] is True, "factory must select the pi05 variant"
    assert config.kw["action_dim"] == 32
    assert config.kw["action_horizon"] == 50


def test_pi0_factory_stays_on_pi0_variant():
    recipe = parse_recipe_from_string(
        """
model:
  name: pi0
  config:
    camera_mapping:
      base_0_rgb: front
action_spec:
  action_dim: 9
  action_horizon: 50
"""
    )
    wrapper = get_entry("pi0").factory(recipe=recipe, schema=None)
    assert wrapper.model.config.kw.get("pi05", False) is False


# ── pi05 discrete-state prompt (openpi PaligemmaTokenizer format) ──


def test_build_prompt_discrete_state():
    state = np.array([-1.0, 0.0, 0.999], dtype=np.float32)
    prompt = build_prompt("pick_up the\nobject ", state)
    # openpi cleaning: strip, "_" → " ", "\n" → " "
    assert prompt.startswith("Task: pick up the object, State: ")
    assert prompt.endswith(";\nAction: ")
    # 256 bins over [-1, 1]: -1.0 → 0, 0.0 → 128, 0.999 → 255
    assert "State: 0 128 255;" in prompt


def test_build_prompt_without_state_is_cleaned_task():
    assert build_prompt(" say_hello ") == "say hello"


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


def test_task_fallback_chain_sample_task_wins():
    step = _tokenize_step(default_task="default")
    step({"task": "from dataset"})
    assert step._tokenizer.seen == ["from dataset"]


def test_task_fallback_chain_default_task():
    step = _tokenize_step(default_task="default")
    step({})
    assert step._tokenizer.seen == ["default"]


def test_task_fallback_chain_empty_prompt_never_skips():
    # openpi-style final fallback: no task text anywhere still yields prompt
    # fields (PI0Pytorch embeds the prompt unconditionally — the fields must
    # exist), tokenized from the empty string.
    step = _tokenize_step()
    out = step({})
    assert step._tokenizer.seen == [""]
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
    stats = _quantile_stats()
    step = NormalizeVector(stats, fields=("actions",), method="quantile")
    inverse = step.inverse_for_output()
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

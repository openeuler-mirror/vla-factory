"""Tests for the pi0 adapter (openpi PI0Pytorch, thin composition).

Patches the entry's openpi handle so tests run without openpi/jax installed.
Runnable both via pytest and directly: `python test/test_pi0_model.py`.
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
import torch.nn as nn

import vla_factory.model.registry.entries.pi0 as pi0_mod
from vla_factory.model.registry import get_entry
from vla_factory.model.protocols.observation import Observation


# ── Fake openpi (so _to_openpi_observation works without openpi installed) ──


@dataclass
class _FakeOpenpiObs:
    images: dict
    image_masks: dict
    state: object
    tokenized_prompt: object = None
    tokenized_prompt_mask: object = None


class _FakePI0Pytorch(nn.Module):
    """Minimal PI0Pytorch stand-in: forward/sample_actions accept openpi Obs."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Linear(1, 1)

    def forward(self, observation, actions, noise=None, time=None):
        return torch.zeros(observation.state.shape[0], 1, 1)

    def sample_actions(self, device, observation, noise=None, num_steps=10):
        return torch.zeros(observation.state.shape[0], 1, 1)


# Patch the entry's openpi handle BEFORE building wrappers. The factory caches
# the import result on _try_import_openpi._cached, so patch that directly.
pi0_mod._try_import_openpi._cached = (
    _FakePI0Pytorch,
    types.SimpleNamespace(action_dim=32),
    _FakeOpenpiObs,
)

# The wrapper imports IMAGE_KEYS from openpi.models_pytorch.preprocessing_pytorch
# at call time; stub that module only when openpi is not installed, so an
# installed openpi keeps its real module for the rest of the session.
try:
    import openpi.models_pytorch.preprocessing_pytorch  # noqa: F401
except ImportError:
    _preproc = types.ModuleType("openpi.models_pytorch.preprocessing_pytorch")
    _preproc.IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    sys.modules["openpi.models_pytorch.preprocessing_pytorch"] = _preproc


def _make_wrapper():
    return pi0_mod.PI0ModelWrapper(
        _FakePI0Pytorch(),
        camera_mapping={"base_0_rgb": "front", "left_wrist_0_rgb": "wrist"},
        image_keys=["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
    )


def _make_obs():
    return Observation(
        images={"front": torch.zeros(2, 224, 224, 3), "wrist": torch.zeros(2, 224, 224, 3)},
        image_masks={"front": torch.ones(2, dtype=torch.bool), "wrist": torch.ones(2, dtype=torch.bool)},
        state=torch.zeros(2, 9),
        tokenized_prompt=torch.zeros(2, 48, dtype=torch.long),
        tokenized_prompt_mask=torch.ones(2, 48, dtype=torch.long),
    )


def test_metadata():
    meta = get_entry("pi0").metadata
    assert meta.name == "pi0"
    assert meta.backend == "pytorch"
    assert meta.action_head_type == "flow_matching"
    assert meta.training_paradigm == "pretrained_finetune"
    assert meta.action_dim == 32
    assert meta.support_lora is True
    assert "llm" in meta.components and "action_expert" in meta.components


def test_camera_mapping_translation():
    wrapper = _make_wrapper()
    openpi_obs = wrapper._to_openpi_observation(_make_obs())
    assert "base_0_rgb" in openpi_obs.images, "base_0_rgb mapped from front"
    assert "left_wrist_0_rgb" in openpi_obs.images, "left_wrist_0_rgb mapped from wrist"
    # unmapped role gets a -1 placeholder
    assert "right_wrist_0_rgb" in openpi_obs.images
    assert (openpi_obs.images["right_wrist_0_rgb"] == -1.0).all()
    assert openpi_obs.state is not None
    assert openpi_obs.tokenized_prompt is not None
    assert openpi_obs.tokenized_prompt_mask is not None


def test_loss_and_predict_delegate_to_upstream():
    wrapper = _make_wrapper()
    obs = _make_obs()
    loss, loss_dict = wrapper.compute_loss(obs, torch.zeros(2, 50, 32))
    assert loss.dim() == 0 and "loss" in loss_dict
    pred = wrapper.predict_actions(obs)
    assert isinstance(pred, torch.Tensor)


def test_empty_camera_mapping_all_placeholders():
    # openpi requires the full fixed IMAGE_KEYS set; an empty mapping fills every
    # role with a -1 placeholder + zero mask (no implicit positional fallback that
    # could swap views — real cameras only land via explicit camera_mapping).
    wrapper = pi0_mod.PI0ModelWrapper(_FakePI0Pytorch(), camera_mapping={})
    openpi_obs = wrapper._to_openpi_observation(_make_obs())
    assert set(openpi_obs.images.keys()) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert all((t == -1.0).all() for t in openpi_obs.images.values())
    assert all((m == 0).all() for m in openpi_obs.image_masks.values())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

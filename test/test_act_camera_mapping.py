"""Regression test for the ACT camera-mapping bug.

``adapters/act.py`` used to ``sorted(observation.images.keys())`` and zip with
``self._image_keys`` (in ``schema.cameras`` order). When the observation dict
order differed from the schema order, cameras were silently swapped — a wrist
image could feed the third-person slot. lerobot is NOT required for this test
(ACTModelWrapper imports it lazily inside the factory).
"""

from __future__ import annotations

import pytest
import torch

from vla_factory.model.model_interface import Observation
from vla_factory.model.adapters.act import ACTModelWrapper


class _DummyModel(torch.nn.Module):
    def forward(self, *args, **kwargs):  # pragma: no cover - never called here
        return None


def test_camera_mapping_is_by_name_not_position():
    # Build-time schema.cameras order was ("wrist", "front") — deliberately not
    # dictionary-sorted order, which is exactly what triggered the old bug.
    image_keys = ["observation.images.wrist", "observation.images.front"]
    wrapper = ACTModelWrapper(_DummyModel(), image_keys=image_keys)

    # Observation dict insertion order is front-first (would swap under the old
    # sorted-zip logic).
    front_t = torch.tensor([1.0])
    wrist_t = torch.tensor([2.0])
    obs = Observation(
        images={"front": front_t, "wrist": wrist_t},
        image_masks={"front": torch.ones(1, dtype=torch.bool), "wrist": torch.ones(1, dtype=torch.bool)},
        state=torch.zeros(6),
    )
    batch = wrapper._obs_to_lerobot_batch(obs)

    # Each camera must land under its OWN config key — no swap.
    assert torch.equal(batch["observation.images.front"], front_t)
    assert torch.equal(batch["observation.images.wrist"], wrist_t)


def test_missing_expected_camera_raises():
    image_keys = ["observation.images.wrist", "observation.images.front"]
    wrapper = ACTModelWrapper(_DummyModel(), image_keys=image_keys)
    obs = Observation(images={"front": torch.tensor([1.0])}, image_masks={"front": torch.ones(1, dtype=torch.bool)}, state=torch.zeros(6))
    # wrist is expected by the model but absent → explicit error, not silent skip.
    with pytest.raises(KeyError, match="wrist"):
        wrapper._obs_to_lerobot_batch(obs)


def test_extra_observation_camera_is_ignored():
    image_keys = ["observation.images.front"]
    wrapper = ACTModelWrapper(_DummyModel(), image_keys=image_keys)
    obs = Observation(
        images={"front": torch.tensor([1.0]), "wrist": torch.tensor([2.0])},
        image_masks={"front": torch.ones(1, dtype=torch.bool), "wrist": torch.ones(1, dtype=torch.bool)},
        state=torch.zeros(6),
    )
    batch = wrapper._obs_to_lerobot_batch(obs)
    assert "observation.images.front" in batch
    assert "observation.images.wrist" not in batch  # not in the model's slots

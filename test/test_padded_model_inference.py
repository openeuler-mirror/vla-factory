"""A padded model's two action widths, end to end through ``InferenceEngine``.

pi0 emits 32 dims and the robot takes 8: the model's internal width and the
width that leaves the engine are different numbers, and checking both ends
against one of them made every prediction of every padded model fail its own
contract check — while the LeRobot action adapter refused to start at all
(32 dims vs 8 motor keys). ACT, the only model the end-to-end tests could build,
has no padding step, so nothing noticed.

These build a checkpoint whose assembly is resolved from a *real* pi0-shaped
declaration and serve it with a stub network, so the widths are exercised
without openpi installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import make_norm_stats, make_schema

from vla_factory.assembly.artifact import save_assembly_artifact
from vla_factory.assembly.resolver import resolve_assembly
from vla_factory.model.interfaces.model import ModelMetadata, VisionSlot
from vla_factory.model.registry import registry as registry_mod
from vla_factory.model.registry.registry import ModelEntry
from vla_factory.utils.constants import (
    ASSEMBLY_FILE, FINAL_DIR, INFERENCE_META_DIR, MODEL_WEIGHTS_FILE, RECIPE_FILE,
)

DATA_ACTION_DIM = 8
MODEL_ACTION_DIM = 32
ACTION_HORIZON = 5
STATE_DIM = 6

_PADDED_METADATA = ModelMetadata(
    name="_padded_stub",
    action_dim=MODEL_ACTION_DIM,
    action_horizon=ACTION_HORIZON,
    dim_policy="padded_to_max",
    dim_policy_max=MODEL_ACTION_DIM,
    vector_normalization="mean_std",
    image_input_range=(-1.0, 1.0),
    requires_prompt=False,
    vision_slots=(
        VisionSlot(name="base_0_rgb", semantic_accepts=("third_person",),
                   resolution=(224, 224)),
    ),
    params={"transforms": {"inputs": [
        {"type": "image_to_float"},
        {"type": "image_layout", "to": "CHW"},
        {"type": "normalize_vector", "fields": ["state", "actions"]},
        {"type": "pad_dimensions", "fields": ["state", "actions"]},
    ]}},
)


class _PaddedStubModel(torch.nn.Module):
    """Emits the model-internal width, like a padded policy does."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def predict_actions(self, observation, num_steps=None):
        return torch.zeros(1, ACTION_HORIZON, MODEL_ACTION_DIM)


@pytest.fixture
def padded_checkpoint(tmp_path):
    """A checkpoint dir whose assembly pads 8 → 32 and unpads back."""
    entry = ModelEntry(
        metadata=_PADDED_METADATA,
        factory=lambda recipe, assembly: _PaddedStubModel(),
    )
    registry_mod._REGISTRY[_PADDED_METADATA.name] = entry
    registry_mod._ENTRIES_LOADED = True
    try:
        schema = make_schema(
            state_dim=STATE_DIM, action_dim=DATA_ACTION_DIM, cameras=("front",),
            image_sizes={"front": (224, 224)},
        )
        assembly = resolve_assembly(
            schema,
            make_norm_stats(state_dim=STATE_DIM, action_dim=DATA_ACTION_DIM),
            _PADDED_METADATA,
        )
        meta_dir = tmp_path / INFERENCE_META_DIR
        meta_dir.mkdir(parents=True)
        save_assembly_artifact(meta_dir / ASSEMBLY_FILE, assembly)
        (meta_dir / RECIPE_FILE).write_text(
            f"model:\n  name: {_PADDED_METADATA.name}\n  config: {{}}\n"
        )
        final_dir = tmp_path / FINAL_DIR
        final_dir.mkdir(parents=True)
        torch.save(_PaddedStubModel().state_dict(), final_dir / MODEL_WEIGHTS_FILE)
        yield tmp_path, assembly
    finally:
        registry_mod._REGISTRY.pop(_PADDED_METADATA.name, None)


def test_the_plan_pads_forward_and_unpads_back(padded_checkpoint):
    _, assembly = padded_checkpoint
    assert assembly.model_io_spec.action_dim == MODEL_ACTION_DIM
    assert [c.type for c in assembly.model_to_robot.calls] == [
        "unpad_action", "unnormalize_action",
    ]


def test_engine_serves_the_command_width_not_the_model_width(padded_checkpoint):
    from vla_factory.inference.infer import InferenceEngine, ObsDict

    checkpoint, _ = padded_checkpoint
    engine = InferenceEngine(checkpoint_path=checkpoint, device="cpu")

    # Two widths, and the public one is what actually leaves the engine.
    assert engine.model_output_dim == MODEL_ACTION_DIM
    assert engine.execution_action_dim == DATA_ACTION_DIM
    assert len(engine.action_keys) == engine.execution_action_dim

    chunk = engine.predict(ObsDict(
        video={"front": np.zeros((224, 224, 3), dtype=np.uint8)},
        state=np.zeros(STATE_DIM, dtype=np.float32),
    ))
    assert chunk.values.shape == (ACTION_HORIZON, DATA_ACTION_DIM)


def test_a_model_that_stops_matching_its_io_spec_is_caught(padded_checkpoint):
    """The raw output is checked before the reverse pipeline touches it, so the
    error names the real problem instead of a post-unpad mismatch."""
    from vla_factory.inference.infer import InferenceEngine, ObsDict

    checkpoint, _ = padded_checkpoint
    engine = InferenceEngine(checkpoint_path=checkpoint, device="cpu")
    engine._model.predict_actions = (
        lambda observation, num_steps=None: torch.zeros(1, ACTION_HORIZON, 7)
    )

    with pytest.raises(ValueError, match="does not match the resolved IO spec"):
        engine.predict(ObsDict(
            video={"front": np.zeros((224, 224, 3), dtype=np.uint8)},
            state=np.zeros(STATE_DIM, dtype=np.float32),
        ))

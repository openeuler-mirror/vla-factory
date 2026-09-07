"""LoRA delta checkpoints served end to end: trainer save → engine → export.

Each piece of the delta path is unit-tested (``VLATrainer._save`` filter,
``validate_delta_load``, ``checkpoint_format``), but the pieces are spliced
together only here — and that wiring is where the non-persistent-buffer bug
lived: every unit passed alone while the assembled path rejected every delta
whose model carries a ``persistent=False`` buffer (transformers rotary
``inv_freq`` is the common one). The stub below declares one on purpose.

Serving a delta must rebuild the declared base from ``model.path`` and apply
only trainable state; exporting must merge that into a self-contained
``bare_full`` model that serves with no base path at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import make_norm_stats, make_schema

from vla_factory.assembly import resolve_from_facts as resolve_assembly
from vla_factory.model.model_interface import ModelMetadata, VisionSlot
from vla_factory.model.registry import ModelEntry, ModelRegistry
from vla_factory.utils.constants import (
    ASSEMBLY_FILE,
    INFERENCE_META_DIR,
    MODEL_WEIGHTS_FILE,
    RECIPE_FILE,
    WEIGHTS_META_FILE,
)

ACTION_DIM = 8
STATE_DIM = 8  # dim_policy="fixed" bounds state and action to the same max
ACTION_HORIZON = 3
FINAL_STEP = 7


def _peft_available() -> bool:
    try:
        import peft  # noqa: F401

        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _peft_available(), reason="peft not installed")


_DELTA_METADATA = ModelMetadata(
    name="_delta_stub",
    action_dim=ACTION_DIM,
    action_horizon=ACTION_HORIZON,
    dim_policy="fixed",
    dim_policy_max=ACTION_DIM,
    vector_normalization="mean_std",
    image_input_range=(-1.0, 1.0),
    image_layout="CHW",
    image_resize_mode="pad",
    requires_prompt=False,
    support_lora=True,
    vision_slots=(
        VisionSlot(name="base_0_rgb", semantic_accepts=("third_person",),
                   resolution=(224, 224)),
    ),
    components={"trunk": ["trunk."]},
)


class _DeltaStubModel(torch.nn.Module):
    """A trunk subtree (LoRA'd) plus a head outside it (full-FT)."""

    def __init__(self) -> None:
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)
        )
        self.head = torch.nn.Linear(4, ACTION_DIM)
        # persistent: rides along in every delta; nonpersistent: never does.
        self.register_buffer("persistent", torch.zeros(1))
        self.register_buffer("nonpersistent", torch.zeros(1), persistent=False)

    def predict_actions(self, observation, num_steps=None):
        return torch.zeros(1, ACTION_HORIZON, ACTION_DIM)


def _factory(recipe, assembly):
    model = _DeltaStubModel()
    if recipe.model.path:
        state = torch.load(recipe.model.path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    return model


@pytest.fixture
def delta_run(tmp_path):
    """A run whose final checkpoint is a LoRA delta, built the real way."""
    entry = ModelEntry(metadata=_DELTA_METADATA, factory=_factory)
    previous_entry = ModelRegistry._entries.get(_DELTA_METADATA.name)
    previous_loaded = ModelRegistry._builtins_loaded
    ModelRegistry._entries[_DELTA_METADATA.name] = entry
    ModelRegistry._builtins_loaded = True
    try:
        schema = make_schema(
            state_dim=STATE_DIM, action_dim=ACTION_DIM, cameras=("front",),
            image_sizes={"front": (224, 224)},
        )
        assembly = resolve_assembly(
            schema,
            make_norm_stats(state_dim=STATE_DIM, action_dim=ACTION_DIM),
            _DELTA_METADATA,
        )

        base_weights = tmp_path / "base" / MODEL_WEIGHTS_FILE
        base_weights.parent.mkdir()
        torch.save(_DeltaStubModel().state_dict(), base_weights)

        meta_dir = tmp_path / INFERENCE_META_DIR
        meta_dir.mkdir()
        assembly.save(meta_dir / ASSEMBLY_FILE)
        (meta_dir / RECIPE_FILE).write_text(
            f"model:\n"
            f"  name: {_DELTA_METADATA.name}\n"
            f"  path: {base_weights}\n"
            f"  config: {{}}\n"
            f"finetuning:\n"
            f"  strategy: lora\n"
            f"  config: {{}}\n"
        )

        # Train-side wiring, production code throughout: wrap exactly as
        # train() does, save exactly as VLATrainer._save does.
        from transformers import TrainingArguments

        from vla_factory.training.strategies import get_strategy
        from vla_factory.training.trainer import VLATrainer
        from vla_factory.user_interface import parse_recipe

        recipe = parse_recipe(meta_dir / RECIPE_FILE)
        strategy = get_strategy(recipe.finetuning.strategy)
        wrapped = strategy.prepare_model(
            _factory(recipe, assembly),
            strategy.parse_config(recipe.finetuning.config),
            _DELTA_METADATA,
        )
        checkpoint = tmp_path / f"checkpoint-{FINAL_STEP}"
        checkpoint.mkdir()
        trainer = VLATrainer(
            model=wrapped,
            args=TrainingArguments(output_dir=str(tmp_path), report_to=[]),
            save_delta_only=True,
            checkpoint_format="lora_delta",
        )
        trainer._save(str(checkpoint))
        (checkpoint / "trainer_state.json").write_text("{}")
        yield tmp_path
    finally:
        if previous_entry is None:
            ModelRegistry._entries.pop(_DELTA_METADATA.name, None)
        else:
            ModelRegistry._entries[_DELTA_METADATA.name] = previous_entry
        ModelRegistry._builtins_loaded = previous_loaded


def _observe():
    from vla_factory.inference.inference_engine import ObsDict

    return ObsDict(
        video={"front": np.zeros((224, 224, 3), dtype=np.uint8)},
        state=np.zeros(STATE_DIM, dtype=np.float32),
    )


def test_delta_holds_trainable_state_and_persistent_buffers_only(delta_run):
    from safetensors.torch import load_file

    delta = load_file(delta_run / f"checkpoint-{FINAL_STEP}" / "model.safetensors")
    keys = set(delta)
    assert any("lora_A" in key for key in keys), "LoRA adapters are the delta"
    assert any("lora_B" in key for key in keys)
    assert "head.weight" in keys and "head.bias" in keys, "full-FT outside the subtree"
    assert "persistent" in keys, "persistent buffers ride along"
    assert "nonpersistent" not in keys, "non-persistent buffers never persist"
    assert not any(key.endswith("trunk.0.weight") for key in keys), "frozen base stays out"


def test_engine_serves_a_delta_by_rebuilding_its_base(delta_run):
    from vla_factory.inference.inference_engine import InferenceEngine

    engine = InferenceEngine(checkpoint_path=delta_run, device="cpu")

    chunk = engine.predict(_observe())
    assert chunk.values.shape == (ACTION_HORIZON, ACTION_DIM)


def test_export_merges_a_delta_into_a_self_contained_model(delta_run):
    import json

    from vla_factory.inference.inference_engine import InferenceEngine
    from vla_factory.training.export import export_checkpoint

    export_dir = delta_run / "export"
    returned = export_checkpoint(delta_run, export_dir)
    assert returned == export_dir
    assert json.loads((export_dir / WEIGHTS_META_FILE).read_text()) == {
        "format": "bare_full"
    }
    assert (export_dir / INFERENCE_META_DIR / ASSEMBLY_FILE).is_file()

    # The export needs no base path: it is a full bare model on its own.
    chunk = InferenceEngine(checkpoint_path=export_dir, device="cpu").predict(_observe())
    assert chunk.values.shape == (ACTION_HORIZON, ACTION_DIM)

"""Tests for the OpenVLA adapter (metadata, Prismatic contract, LoRA target_modules).

OpenVLA's upstream Prismatic stack is not required — these tests cover the
framework-side pieces (registration, Prismatic checkpoint config, per-model LoRA
target_modules, adapter input construction) with no upstream import.

Runnable via pytest or directly: `python test/test_openvla_model.py`.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
import torch.nn as nn

from vla_factory.assembly import resolve_from_facts as resolve_assembly
from vla_factory.data.data_schema import NormStats
from vla_factory.model.checkpoint_validation import extract_checkpoint_observations
from vla_factory.model.model_interface import Observation
from vla_factory.model.registry import get_entry, list_entries
from vla_factory.user_interface import AssemblyOverrides

from helpers import make_schema


# ── Metadata / registration ────────────────────────────────────────


def test_metadata():
    meta = get_entry("openvla-7b").metadata
    assert meta.name == "openvla-7b"
    assert meta.backend == "pytorch"
    assert meta.action_head_type == "autoregressive"
    assert meta.action_dim == 0  # flexible: dataset supplies 7-DoF; adapter normalises internally
    assert meta.action_horizon == 1
    assert meta.training_paradigm == "pretrained_finetune"
    assert meta.support_lora is True
    assert meta.support_full is True
    assert meta.support_freeze is True
    assert "llm" in meta.components and "vision_encoder" in meta.components
    # Contract visibility: the checkpoint's instruction format, declared
    # read-only (no pipeline step consumes it for this prompt-free model).
    assert meta.language_template == "What action should the robot take to {task}?"
    assert meta.params.get("dtype") == "bfloat16"
    assert meta.params.get("num_inference_steps") == 1


def test_oft_registered_and_shares_adapter():
    assert "openvla-7b-oft" in list_entries()
    oft = get_entry("openvla-7b-oft").metadata
    assert oft.name == "openvla-7b-oft"
    # OFT shares the full contract with openvla-7b (same adapter/loader):
    # flexible action width (dataset supplies 7-DoF) and no framework prompt
    # pipeline. requires_prompt=True/action_dim!=0 would make assembly
    # resolution fail (see test_oft_assembly_resolves).
    assert oft.action_dim == 0
    assert oft.requires_prompt is False
    assert oft.dim_policy == "flexible"
    assert oft.components == get_entry("openvla-7b").metadata.components
    assert oft.language_template == get_entry("openvla-7b").metadata.language_template


def test_oft_and_base_assembly_resolves():
    # Regression for review finding: openvla-7b-oft declared requires_prompt=True
    # without tokenizer_max_length (and a fixed action_dim=7 with
    # dim_policy='flexible'), so assembly resolution raised ValueError and the
    # model entry was unusable. Both OpenVLA entries must resolve against a
    # plain 7-DoF non-language schema.
    schema = make_schema(
        state_dim=6, action_dim=7, cameras=("front",), fps=30, has_language=False,
        state_keys=("a", "b", "c", "d", "e", "f"),
        action_keys=("x1", "x2", "x3", "x4", "x5", "x6", "x7"),
    )
    for name in ("openvla-7b", "openvla-7b-oft"):
        meta = get_entry(name).metadata
        assembly = resolve_assembly(schema, NormStats(), meta)
        assert assembly.model_io_spec.action_dim == 7
        assert assembly.model_io_spec.requires_language is False


# ── Prismatic checkpoint config ─────────────────────────────────────


# Mirrors the real openvla/openvla-7b config.json (model_type, image_sizes,
# norm_stats with per-dataset action q01/q99).
PRISMATIC_CONFIG = {
    "model_type": "openvla",
    "architectures": ["OpenVLAForActionPrediction"],
    "image_sizes": [224, 224],
    "n_action_bins": 256,
    "norm_stats": {
        "some_dataset": {
            "action": {
                "q01": [-1.0, -0.96, -0.87, 0.0, 0.0, 0.0, 0.0],
                "q99": [1.0, 0.86, 1.0, 0.0, 0.0, 0.0, 1.0],
            }
        }
    },
}


def _write_config(cfg) -> str:
    d = tempfile.mkdtemp()
    Path(d, "config.json").write_text(json.dumps(cfg))
    return d


def test_prismatic_contract():
    obs = extract_checkpoint_observations(PRISMATIC_CONFIG)
    assert obs["camera_roles"] == {"primary": (3, 224, 224)}
    assert obs["action_dim"] == 7  # derived from norm_stats q01 length
    assert obs["state_dim"] is None
    assert obs["image_resolution"] == (224, 224)


def test_prismatic_action_dim_falls_back_to_none():
    cfg = {k: v for k, v in PRISMATIC_CONFIG.items() if k != "norm_stats"}
    obs = extract_checkpoint_observations(cfg)
    assert obs["action_dim"] is None


def test_lerobot_contract_not_mistaken_for_prismatic():
    # A lerobot-style config must still take the input_features path.
    cfg = {"type": "pi0", "input_features": {}, "output_features": {}}
    obs = extract_checkpoint_observations(cfg)
    assert obs["camera_roles"] is None
    assert obs["action_dim"] is None


# ── LoRA target_modules forwarding (upstream LoraConfig field) ──────


def test_lora_forwards_target_modules_from_config(monkeypatch):
    import peft  # framework hard dependency (peft>=0.11)

    captured = {}

    class _FakeLoraConfig:
        def __init__(self, **kw):
            captured["kw"] = kw

    def _fake_get_peft_model(module, cfg):
        captured["module"] = module
        return module

    monkeypatch.setattr(peft, "LoraConfig", _FakeLoraConfig)
    monkeypatch.setattr(peft, "get_peft_model", _fake_get_peft_model)

    from vla_factory.model.model_interface import ModelMetadata
    from vla_factory.training.strategies.lora import apply_lora

    class LLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(4, 4)
            self.v_proj = nn.Linear(4, 4)
            self.down_proj = nn.Linear(4, 4)

    class Body(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = LLM()

    meta = ModelMetadata(
        name="openvla-7b",
        support_lora=True,
        components={"llm": ["language_model."]},
    )

    class FakeConfig:
        r = 8
        lora_alpha = 8
        lora_dropout = 0.0
        use_rslora = False
        init_lora_weights = "gaussian"
        components = ["llm"]
        freeze_components = []
        target_modules = "all-linear"

    body = Body()
    apply_lora(body, FakeConfig(), meta)
    # The recipe-level tunable is forwarded to peft verbatim, and only the
    # declared subtree gets wrapped (never the whole model).
    assert captured["kw"]["target_modules"] == "all-linear"
    assert captured["module"] is body.language_model


# ── Adapter input construction (no upstream model needed) ───────────


def test_adapter_normalize_actions():
    from vla_factory.model.adapters.openvla import (
        _FINETUNE_STATS_KEY,
        OpenVLAModelWrapper,
    )

    class FakeModel:
        def get_action_stats(self, stats_key):
            assert stats_key == _FINETUNE_STATS_KEY
            return {"q01": [0.0] * 7, "q99": [1.0] * 7}

        def parameters(self):
            return iter([torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))])

    class _IP:
        apply_transform = None

    class FakeProcessor:
        tokenizer = None
        image_processor = _IP()

    w = OpenVLAModelWrapper(FakeModel(), FakeProcessor(), None, None, None)
    actions = torch.tensor([[[0.0, 0.5, 1.0, 0.25, 0.75, 0.0, 1.0]]])  # [1,1,7]
    norm = w._normalize_actions(actions)
    # BOUNDS_Q99: 2*(a - q01)/(q99 - q01) - 1 = 2a - 1 over [0,1]
    expected = torch.tensor([[-1.0, 0.0, 1.0, -0.5, 0.5, -1.0, 1.0]])
    assert torch.allclose(norm, expected, atol=1e-6)


def test_adapter_build_training_instance():
    from vla_factory.model.model_interface import Observation
    from vla_factory.model.adapters.openvla import IGNORE_INDEX, OpenVLAModelWrapper

    class FakePromptBuilder:
        def __init__(self, family):
            assert family == "openvla"
            self.turns = []

        def add_turn(self, role, value):
            self.turns.append((role, value))

        def get_prompt(self):
            return "In: ...\nOut: "

    class _TokOut:
        input_ids = list(range(10))

    class FakeTokenizer:
        def __call__(self, text, add_special_tokens=True):
            return _TokOut()

    class FakeImageTransform:
        def __call__(self, pil):
            return torch.zeros(3, 224, 224)

    class _IP:
        apply_transform = FakeImageTransform()

    class FakeProcessor:
        tokenizer = FakeTokenizer()
        image_processor = _IP()

    class FakeActionTokenizer:
        def __init__(self):
            self.calls = []

        def __call__(self, action):
            self.calls.append(action)
            return "<act>"

    class _LossOut:
        def __init__(self):
            self.loss = torch.tensor(1.0)

    class FakeModel:
        def __init__(self):
            # Patch through the collator'd batch to avoid touching upstream,
            # but record what _build_training_instance produced.
            self.last_batch = None

        def get_action_stats(self, k):
            return {"q01": [0.0] * 7, "q99": [1.0] * 7}

        def parameters(self):
            return iter([torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))])

        def forward(self, **batch):
            self.last_batch = batch
            return _LossOut()

        def __call__(self, **batch):
            return self.forward(**batch)

    at = FakeActionTokenizer()
    w = OpenVLAModelWrapper(FakeModel(), FakeProcessor(), at, FakePromptBuilder, None)
    obs = Observation(
        images={"front": torch.randint(0, 255, (1, 224, 224, 3), dtype=torch.uint8)},
        image_masks={"front": torch.ones(1, dtype=torch.bool)},
        task=["pick apple"],
    )
    inst = w._build_training_instance("pick apple", torch.zeros(7), obs, 0)

    assert inst["input_ids"].tolist() == list(range(10))
    # Only the last (len(action) + 1) = 8 positions carry a loss; prompt masked.
    assert inst["labels"][:2].tolist() == [IGNORE_INDEX, IGNORE_INDEX]
    assert inst["labels"][2:].tolist() == list(range(2, 10))
    assert tuple(inst["pixel_values"].shape) == (3, 224, 224)
    assert len(at.calls) == 1 and torch.allclose(at.calls[0], torch.zeros(7))


def test_camera_mapping_selects_primary_camera():
    # Review fix #2: _image_to_pixel_values must honour the resolved
    # camera_mapping (primary slot -> data camera) instead of silently taking
    # the first camera. Build a real assembly over a two-camera schema with
    # overrides.camera_mapping.primary: front and check the wrapper resolves
    # the front image.
    from vla_factory.model.adapters.openvla import OpenVLAModelWrapper

    entry = get_entry("openvla-7b")
    schema = make_schema(
        state_dim=6, action_dim=7, cameras=("wrist", "front"), fps=30,
        has_language=False,
        state_keys=("a", "b", "c", "d", "e", "f"),
        action_keys=("x1", "x2", "x3", "x4", "x5", "x6", "x7"),
    )
    assembly = resolve_assembly(
        schema, NormStats(), entry.metadata,
        overrides=AssemblyOverrides(camera_mapping={"primary": "front"}),
    )
    assert (assembly.camera_mapping.entries[0]["model_slot"],
            assembly.camera_mapping.entries[0]["data_source"]) == ("primary", "front")

    class FakeImageTransform:
        def __call__(self, pil):
            return torch.zeros(3, 224, 224)

    class _IP:
        apply_transform = FakeImageTransform()

    class FakeProcessor:
        tokenizer = None
        image_processor = _IP()

    class FakeModel:
        def parameters(self):
            return iter([torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))])

    w = OpenVLAModelWrapper(
        FakeModel(), FakeProcessor(), None, None, None,
        camera_key="front",
    )
    obs = Observation(
        images={
            "wrist": torch.randint(0, 255, (1, 224, 224, 3), dtype=torch.uint8),
            "front": torch.randint(0, 255, (1, 224, 224, 3), dtype=torch.uint8),
        },
        image_masks={k: torch.ones(1, dtype=torch.bool) for k in ("wrist", "front")},
        task=["pick apple"],
    )
    # Wrong camera key -> loud KeyError, not a silent first-camera pick.
    w._camera_key = "wrist"
    px = w._image_to_pixel_values(obs, 0)
    assert tuple(px.shape) == (3, 224, 224)

    # Missing declared camera -> KeyError.
    w._camera_key = "missing"
    try:
        w._image_to_pixel_values(obs, 0)
        raise AssertionError("expected KeyError for missing camera")
    except KeyError:
        pass

    # Unmapped + multi-camera -> ValueError (ambiguity must fail loudly).
    w._camera_key = None
    try:
        w._image_to_pixel_values(obs, 0)
        raise AssertionError("expected ValueError for ambiguous cameras")
    except ValueError:
        pass

    # Unmapped + single camera -> unambiguous fallback still works.
    single = Observation(
        images={"front": torch.randint(0, 255, (1, 224, 224, 3), dtype=torch.uint8)},
        image_masks={"front": torch.ones(1, dtype=torch.bool)},
        task=["pick apple"],
    )
    w._camera_key = None
    assert tuple(w._image_to_pixel_values(single, 0).shape) == (3, 224, 224)


def test_dataset_quantiles_fail_fast():
    # Normalization binds to the DATASET's own q01/q99 (upstream fine-tuning
    # semantics). Older lerobot stats.json files carry min/max/mean/std only —
    # such datasets must fail with an actionable error, not silently train
    # against someone else's statistics.
    from vla_factory.model.adapters.openvla import _require_dataset_quantiles

    class _Missing:
        class norm_stats:
            class action:
                q01, q99 = [], []

    with pytest.raises(ValueError, match="q01/q99"):
        _require_dataset_quantiles(_Missing(), "openvla-7b")

    class _Present:
        class norm_stats:
            class action:
                q01, q99 = [0.1, 0.2], [0.3, 0.4]

    assert _require_dataset_quantiles(_Present(), "openvla-7b") == ([0.1, 0.2], [0.3, 0.4])


def test_finetune_stats_injection():
    # The dataset's stats are mounted in the loaded checkpoint's norm_stats
    # under one key, so upstream get_action_stats (training encode) and
    # predict_action (inference decode) share exactly the stats this
    # fine-tune used — the round-trip rule, by construction.
    from vla_factory.model.adapters.openvla import (
        _FINETUNE_STATS_KEY,
        _inject_finetune_stats,
    )

    class FakeModel:
        norm_stats = {"bridge_orig": {"action": {"q01": [0.0], "q99": [1.0]}}}

    _inject_finetune_stats(FakeModel(), [0.1, 0.2], [0.3, 0.4])
    mounted = FakeModel.norm_stats[_FINETUNE_STATS_KEY]["action"]
    assert mounted["q01"] == [0.1, 0.2]
    assert mounted["q99"] == [0.3, 0.4]


def test_task_text_is_pure_transport():
    # Review round 2: the wrapper used to mirror the framework's
    # sample["task"] > default_task > "" chain — a second answer that could
    # drift from the framework's. The chain now lives only framework-side
    # (task_tokenize for prompt models, inject_default_task for prompt-free
    # ones); the adapter reads Observation.task as final transport. Going
    # through compute_loss (the real path) verifies the call-site read.
    from vla_factory.model.adapters.openvla import OpenVLAModelWrapper

    class FakePromptBuilder:
        def __init__(self, family):
            assert family == "openvla"
            self.turns = []

        def add_turn(self, role, value):
            self.turns.append((role, value))

        def get_prompt(self):
            return "In: ...\nOut: "

    class _TokOut:
        input_ids = list(range(10))

    class FakeTokenizer:
        def __call__(self, text, add_special_tokens=True):
            return _TokOut()

    class FakeImageTransform:
        def __call__(self, pil):
            return torch.zeros(3, 224, 224)

    class _IP:
        apply_transform = FakeImageTransform()

    class FakeProcessor:
        tokenizer = FakeTokenizer()
        image_processor = _IP()

    class FakeActionTokenizer:
        def __init__(self):
            self.calls = []

        def __call__(self, action):
            self.calls.append(action)
            return "<act>"

    class _LossOut:
        def __init__(self):
            self.loss = torch.tensor(1.0)

    class FakeModel:
        def __init__(self):
            # Patch through the collator'd batch to avoid touching upstream,
            # but record what _build_training_instance produced.
            self.last_batch = None

        def get_action_stats(self, k):
            return {"q01": [0.0] * 7, "q99": [1.0] * 7}

        def parameters(self):
            return iter([torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))])

        def forward(self, **batch):
            self.last_batch = batch
            return _LossOut()

        def __call__(self, **batch):
            return self.forward(**batch)

    class RecordingPromptBuilder(FakePromptBuilder):
        human_turns = []

        def add_turn(self, role, value):
            if role == "human":
                RecordingPromptBuilder.human_turns.append(value)
            super().add_turn(role, value)

    def _identity_collator(instances):
        return {"instances": instances}

    model = FakeModel()
    w = OpenVLAModelWrapper(
        model, FakeProcessor(), FakeActionTokenizer(),
        RecordingPromptBuilder, _identity_collator,
    )
    actions = torch.zeros(1, 1, 7)

    # default_task no longer lives on the adapter: the wrapper is a pure
    # transport reader, the chain is framework-side (inject_default_task).
    assert not hasattr(w, "_default_task")

    # Transport entry present -> it is the prompt text, verbatim.
    obs2 = Observation(
        images={"front": torch.randint(0, 255, (1, 224, 224, 3), dtype=torch.uint8)},
        image_masks={"front": torch.ones(1, dtype=torch.bool)},
        task=["push red block"],
    )
    RecordingPromptBuilder.human_turns = []
    w.compute_loss(obs2, actions)
    assert any("push red block" in t for t in RecordingPromptBuilder.human_turns)

    # Transport absent -> the chain's terminal "": empty instruction.
    obs = Observation(
        images={"front": torch.randint(0, 255, (1, 224, 224, 3), dtype=torch.uint8)},
        image_masks={"front": torch.ones(1, dtype=torch.bool)},
        task=None,
    )
    RecordingPromptBuilder.human_turns = []
    w.compute_loss(obs, actions)
    assert any("take to ?" in t for t in RecordingPromptBuilder.human_turns) or any(
        "take to  ?" in t for t in RecordingPromptBuilder.human_turns
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

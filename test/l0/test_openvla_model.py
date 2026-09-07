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
import numpy as np
import torch
import torch.nn as nn

from vla_factory.assembly import resolve_from_facts as resolve_assembly
from vla_factory.data.data_schema import FeatureStats, NormStats
from vla_factory.model.checkpoint_validation import extract_checkpoint_observations
from vla_factory.model.model_interface import Observation
from vla_factory.model.registry import get_entry, list_entries
from vla_factory.user_interface import AssemblyOverrides


def _quantile_stats(state_dim: int = 6, action_dim: int = 7) -> NormStats:
    # vector_normalization="quantile" requires per-dim q01/q99 for state and
    # action at resolve time — real datasets must ship them in stats.json.
    return NormStats(
        state=FeatureStats(q01=[0.0] * state_dim, q99=[1.0] * state_dim),
        action=FeatureStats(q01=[0.0] * action_dim, q99=[1.0] * action_dim),
    )

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
    # Contract visibility: the checkpoint's instruction format, consumed by
    # the plan's assemble_token_action_sequence step.
    assert meta.language_template == "What action should the robot take to {task}?"
    # Action normalization runs in the plan (normalize_vector, quantile over
    # the dataset's own q01/q99) — same semantics as upstream fine-tuning.
    assert meta.vector_normalization == "quantile"
    assert meta.vector_normalization_eps == 1e-6
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
    assert oft.vector_normalization == get_entry("openvla-7b").metadata.vector_normalization


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
        assembly = resolve_assembly(
            schema, _quantile_stats(), meta,
            model_path="/ckpts/openvla-7b",  # tokenizer source for the assemble step
        )
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




def test_checkpoint_image_transform_planning_and_step():
    # The checkpoint's own processor owns the image contract plan-side: the
    # plan addresses the resolved primary camera, and the step converts that
    # camera's raw HWC uint8 through the processor. Wrong/missing camera
    # mapping fails at resolve time, not silently.
    from vla_factory.assembly.transform.checkpoint_image import (
        CheckpointImageTransform,
    )

    entry = get_entry("openvla-7b")
    schema = make_schema(
        state_dim=6, action_dim=7, cameras=("wrist", "front"), fps=30,
        has_language=False,
        state_keys=("a", "b", "c", "d", "e", "f"),
        action_keys=("x1", "x2", "x3", "x4", "x5", "x6", "x7"),
    )
    assembly = resolve_assembly(
        schema, _quantile_stats(), entry.metadata,
        overrides=AssemblyOverrides(camera_mapping={"primary": "front"}),
        model_path="/ckpts/openvla-7b",
    )
    args = {c.type: c.args for c in assembly.data_to_model.calls}
    assert "resize_images" not in args  # the processor owns geometry
    assert args["checkpoint_image_transform"]["source_key"] == "images.front"
    assert args["checkpoint_image_transform"]["repo"] == "/ckpts/openvla-7b"

    # Unresolved primary on a multi-camera dataset -> resolution fails loudly
    # (the planner demands an unambiguous primary camera).
    from vla_factory.assembly.resolve import ResolutionError

    with pytest.raises((ResolutionError, ValueError)):
        resolve_assembly(
            schema, _quantile_stats(), entry.metadata,
            model_path="/ckpts/openvla-7b",
        )

    # Step: reads the addressed camera key and runs the checkpoint processor.
    recorded = {}

    class FakeProcessor:
        @staticmethod
        def apply_transform(pil):
            recorded["size"] = pil.size
            return torch.zeros(6, 224, 224)

    import vla_factory.assembly.transform.checkpoint_image as ci

    original = ci._registered_image_processor
    ci._registered_image_processor = lambda repo: FakeProcessor()
    try:
        step = CheckpointImageTransform(source_key="images.front", repo="/ckpts")
        out = step({"images.front": np.full((224, 224, 3), 200, dtype=np.uint8)})
    finally:
        ci._registered_image_processor = original

    assert recorded["size"] == (224, 224)  # HWC uint8 -> PIL, processor called
    assert tuple(out["pixel_values"].shape) == (6, 224, 224)


def test_identity_stats_injection():
    # predict_action's decode affine (0.5*(a+1)*(q99-q01)+q01) becomes the
    # identity for q01=-1/q99=+1, so it returns actions in the NORMALIZED
    # space; the plan's model_to_robot inverse (real assembly statistics)
    # finishes the round trip — pi0's division of labor.
    from vla_factory.model.adapters.openvla import (
        _FINETUNE_STATS_KEY,
        _inject_identity_action_stats,
    )

    class FakeModel:
        norm_stats = {"bridge_orig": {"action": {"q01": [0.0], "q99": [1.0]}}}

    _inject_identity_action_stats(FakeModel(), action_dim=7)
    mounted = FakeModel.norm_stats[_FINETUNE_STATS_KEY]["action"]
    assert mounted["q01"] == [-1.0] * 7
    assert mounted["q99"] == [1.0] * 7


def test_compute_loss_derives_labels_from_plan():
    # The plan's assemble step delivers input ids + masks via Observation;
    # the adapter derives HF labels from the supervision mask and forwards
    # the assembled batch to the upstream model.
    from vla_factory.model.adapters.openvla import IGNORE_INDEX, OpenVLAModelWrapper

    class _LossOut:
        def __init__(self):
            self.loss = torch.tensor(1.0)

    class FakeModel:
        def __init__(self):
            self.last_batch = None

        def parameters(self):
            return iter([torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))])

        def forward(self, **batch):
            self.last_batch = batch
            return _LossOut()

        def __call__(self, **batch):
            return self.forward(**batch)

    model = FakeModel()
    w = OpenVLAModelWrapper(model)
    obs = Observation(
        images={"front": torch.randint(0, 255, (1, 224, 224, 3), dtype=torch.uint8)},
        image_masks={"front": torch.ones(1, dtype=torch.bool)},
        tokenized_prompt=torch.tensor([[5, 6, 7, 8, 0, 0]]),
        tokenized_prompt_mask=torch.tensor([[True, True, True, True, False, False]]),
        token_loss_mask=torch.tensor([[False, False, False, True, False, False]]),
        pixel_values=torch.zeros(1, 3, 224, 224),
    )
    loss, logs = w.compute_loss(obs, torch.zeros(1, 1, 7))

    assert loss.item() == 1.0 and logs["loss"] == 1.0
    b = model.last_batch
    # HF labels: real tokens where supervised, IGNORE_INDEX elsewhere.
    assert b["labels"].tolist() == [[-100, -100, -100, 8, -100, -100]]
    assert torch.equal(b["attention_mask"], obs.tokenized_prompt_mask)
    assert tuple(b["pixel_values"].shape) == (1, 3, 224, 224)
    assert IGNORE_INDEX == -100


def test_predict_selects_unpadded_tokens():
    # The assemble step pads to the declared budget; upstream predict_action
    # expects the unpadded input, so the adapter selects real tokens via the
    # attention mask before delegating.
    from vla_factory.model.adapters.openvla import (
        _FINETUNE_STATS_KEY,
        OpenVLAModelWrapper,
    )

    recorded = {}

    class FakeModel:
        def parameters(self):
            return iter([torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))])

        def predict_action(self, input_ids, unnorm_key, pixel_values=None):
            recorded["input_ids"] = input_ids.clone()
            recorded["key"] = unnorm_key
            return torch.tensor([0.5, -0.5, 0.0, 0.25, 0.75, 0.1, 0.3])

    w = OpenVLAModelWrapper(FakeModel())
    obs = Observation(
        images={"front": torch.randint(0, 255, (1, 224, 224, 3), dtype=torch.uint8)},
        image_masks={"front": torch.ones(1, dtype=torch.bool)},
        tokenized_prompt=torch.tensor([[5, 6, 7, 8, 0, 0]]),
        tokenized_prompt_mask=torch.tensor([[True, True, True, True, False, False]]),
        pixel_values=torch.zeros(3, 224, 224),
    )
    actions = w.predict_actions(obs)

    # Real tokens only — padding stripped before the upstream call; the key
    # addresses the identity stats the factory mounted.
    assert recorded["input_ids"].tolist() == [[5, 6, 7, 8]]
    assert recorded["key"] == _FINETUNE_STATS_KEY
    assert tuple(actions.shape) == (1, 1, 7)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

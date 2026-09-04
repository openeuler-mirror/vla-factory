"""OpenVLA adapter — wraps the upstream Prismatic/OpenVLA model (HF AutoClasses).

OpenVLA (``openvla/openvla-7b``) is a pretrained vision-language-action model:
Llama-2 7B LLM + fused DINOv2/SigLIP vision encoder + a multimodal projector.
Actions are discretised to the last 256 tokens of the LLM vocabulary and
predicted as ordinary next-token output (autoregressive head = the LM head over
the action-token sub-vocabulary); there is no separate action-head MLP.

Because OpenVLA's input contract (chat prompt + action tokens interleaved,
PIL-based image transform, per-dataset q01/q99 normalisation) does not match
the framework's generic transforms, this adapter owns the OpenVLA-specific input
construction, using the upstream ``PurePromptBuilder`` / ``ActionTokenizer`` /
processor image transform — the same code path as upstream's
``RLDSBatchTransform`` + ``PaddedCollatorForActionPrediction``.

OpenVLA-oft is the SAME architecture as a *different base checkpoint*
(pre-fine-tuned upstream with OFT) — ``adapters/openvla_oft.py`` reuses this
wrapper/loader and only changes the metadata name.

Requires the vendored OpenVLA/Prismatic stack:
``bash scripts/install.sh --model openvla``.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from vla_factory.model.model_interface import ModelMetadata, Observation, VisionSlot
from vla_factory.model.registry import register_vla
from vla_factory.user_interface import TrainRecipe
from vla_factory.utils.tracked_config import TrackedConfig

logger = logging.getLogger(__name__)

# Llama-2 label ignore index (matches HF / upstream OpenVLA).
IGNORE_INDEX = -100


# ── Detect upstream OpenVLA / Prismatic availability ───────────────


def try_import_openvla():
    """Return the upstream OpenVLA classes/functions, or None.

    Called lazily from the factory — never at module import — so ``list_entries()``
    keeps working without the prismatic stack. Cached on the function object.
    Returns::

        (OpenVLAConfig, OpenVLAForActionPrediction, PrismaticImageProcessor,
         PrismaticProcessor, ActionTokenizer, PurePromptBuilder,
         PaddedCollatorForActionPrediction)
    """
    if getattr(try_import_openvla, "_cached", None) is not None:
        return try_import_openvla._cached  # type: ignore[attr-defined]
    try:
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.extern.hf.processing_prismatic import (
            PrismaticImageProcessor,
            PrismaticProcessor,
        )
        from prismatic.models.backbones.llm.prompting import PurePromptBuilder
        from prismatic.util.data_utils import PaddedCollatorForActionPrediction
        from prismatic.vla.action_tokenizer import ActionTokenizer

        cached = (
            OpenVLAConfig,
            OpenVLAForActionPrediction,
            PrismaticImageProcessor,
            PrismaticProcessor,
            ActionTokenizer,
            PurePromptBuilder,
            PaddedCollatorForActionPrediction,
        )
    except Exception as e:  # noqa: BLE001
        logger.info("openvla not available (%s: %s)", type(e).__name__, e)
        cached = None
    try_import_openvla._cached = cached  # type: ignore[attr-defined]
    return cached


# ── Wrapper (nn.Module, satisfies VLAModelPyTorch; composition) ─────


class OpenVLAModelWrapper(nn.Module):
    """Thin adapter: vla ``Observation`` → upstream OpenVLA input → loss / actions.

    ``self.model`` is the upstream ``OpenVLAForActionPrediction``; ``nn.Module``
    auto-registers it as a submodule, so ``parameters()`` / ``train()`` / ``to()``
    recurse automatically. The adapter holds the processor (tokenizer + image
    transform), the upstream ``ActionTokenizer``, and the prompt-builder class.
    """

    def __init__(
        self,
        model,
        processor,
        action_tokenizer,
        prompt_builder_fn,
        collator,
        unnorm_key=None,
        camera_key=None,
    ):
        super().__init__()
        self.model = model
        self._backend = "prismatic"
        self._tokenizer = processor.tokenizer
        self._image_transform = processor.image_processor.apply_transform
        self._action_tokenizer = action_tokenizer
        self._prompt_builder_fn = prompt_builder_fn
        self._collator = collator
        self._unnorm_key = unnorm_key
        # Dataset camera feeding the single primary visual slot, from the
        # resolved assembly camera_mapping ("primary" -> data camera). None
        # means "not declared": single-camera observations are unambiguous and
        # fall back to the only image; multi-camera ones fail loudly.
        self._camera_key = camera_key

    @property
    def _device(self):
        return next(self.model.parameters()).device

    @property
    def _dtype(self):
        return next(self.model.parameters()).dtype

    def forward(self, observation, actions, action_is_pad=None):
        return self.compute_loss(observation, actions, action_is_pad=action_is_pad)

    def gradient_checkpointing_enable(self, *args, **kwargs):
        # Delegate to the wrapped OpenVLA model — HF Trainer calls this when
        # gradient_checkpointing is enabled. The upstream model's signature
        # may not accept HF's gradient_checkpointing_kwargs, so fall back.
        try:
            return self.model.gradient_checkpointing_enable(*args, **kwargs)
        except TypeError:
            return self.model.gradient_checkpointing_enable()

    # ── Training ──────────────────────────────────────────────────

    def compute_loss(self, observation, actions, action_is_pad=None):
        # actions: [B, 1, D] raw continuous. Normalise to [-1, 1] via the
        # model's per-dataset q01/q99 stats (BOUNDS_Q99), then discretise.
        norm_actions = self._normalize_actions(actions)

        instances = []
        for i in range(actions.shape[0]):
            # Pure transport read: the fallback chain (sample["task"] >
            # default_task > "") is resolved framework-side (task_tokenize for
            # prompt models, inject_default_task for prompt-free ones), so the
            # entries here are final. An absent entry is the chain's terminal
            # "" — no adapter-side fallback lives here.
            task = (
                observation.task[i]
                if observation.task and i < len(observation.task)
                else ""
            )
            # ActionTokenizer uses np.clip internally; must pass CPU numpy.
            instances.append(
                self._build_training_instance(task, norm_actions[i].cpu().numpy(), observation, i)
            )

        batch = self._collator(instances)
        batch = {
            k: (v.to(self._device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

        out = self.model(**batch)
        return out.loss, {"loss": out.loss.item()}

    def _build_training_instance(self, task, normalized_action, observation, index):
        """Mirror upstream ``RLDSBatchTransform``: chat prompt + action tokens."""
        prompt_builder = self._prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": _task_query(task)},
            {"from": "gpt", "value": self._action_tokenizer(normalized_action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = self._tokenizer(
            prompt_builder.get_prompt(), add_special_tokens=True
        ).input_ids
        labels = torch.tensor(input_ids, dtype=torch.long)
        # Only the action tokens (+ stop token) carry a loss; the prompt is masked.
        labels[: -(len(normalized_action) + 1)] = IGNORE_INDEX

        return {
            "input_ids": torch.as_tensor(input_ids, dtype=torch.long),
            "labels": labels,
            "pixel_values": self._image_to_pixel_values(observation, index),
        }

    # ── Inference ─────────────────────────────────────────────────

    def predict_actions(self, observation, **kwargs):
        task = (
            observation.task[0]
            if observation.task and observation.task[0]
            else ""
        )
        prompt_builder = self._prompt_builder_fn("openvla")
        prompt_builder.add_turn("human", _task_query(task))
        input_ids = self._tokenizer(
            prompt_builder.get_prompt(), add_special_tokens=True
        ).input_ids
        input_ids = torch.as_tensor(
            input_ids, dtype=torch.long, device=self._device
        ).unsqueeze(0)

        pixel_values = self._image_to_pixel_values(observation, 0).unsqueeze(0).to(self._device)

        # OpenVLAForActionPrediction.predict_action → unnormalized [D] continuous action.
        actions = self.model.predict_action(
            input_ids, self._unnorm_key, pixel_values=pixel_values
        )
        return torch.as_tensor(actions, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # ── Helpers ───────────────────────────────────────────────────

    def _normalize_actions(self, actions):
        stats = self.model.get_action_stats(self._unnorm_key)
        q01 = torch.as_tensor(stats["q01"], dtype=actions.dtype, device=actions.device)
        q99 = torch.as_tensor(stats["q99"], dtype=actions.dtype, device=actions.device)
        # BOUNDS_Q99: [q01, q99] → [-1, 1]
        return 2.0 * (actions[:, 0, :] - q01) / (q99 - q01 + 1e-8) - 1.0

    def _image_to_pixel_values(self, observation, index):
        # observation.images: {camera: [B, H, W, C] uint8} — raw, no framework
        # image transforms (OpenVLA's processor does Resize/CenterCrop/Normalize).
        images = observation.images
        if self._camera_key is not None:
            if self._camera_key not in images:
                raise KeyError(
                    f"OpenVLA primary camera {self._camera_key!r} missing from "
                    f"observation; available cameras: {sorted(images)}."
                )
            img = images[self._camera_key][index]
        else:
            # No resolved camera mapping: unambiguous only for a single camera.
            if len(images) != 1:
                raise ValueError(
                    "OpenVLA expects one primary camera but observation has "
                    f"{sorted(images)} and no camera_mapping resolved a "
                    "primary slot. Set overrides.camera_mapping.primary."
                )
            img = next(iter(images.values()))[index]
        arr = img.detach().cpu().numpy().astype(np.uint8)
        # image_transform yields float32; cast to the model's dtype (bf16) so the
        # vision backbone (bf16 weights) doesn't hit a float/bf16 conv mismatch.
        return self._image_transform(Image.fromarray(arr)).to(dtype=self._dtype)


# ── Registration ───────────────────────────────────────────────────

# The instruction format OpenVLA was pretrained with (mirrored from upstream
# RLDSBatchTransform). Declared read-only in ModelMetadata.language_template
# (contract visibility); formatted at the two construction sites below. The
# task text is lowercased before formatting, matching upstream.
_OPENVLA_TASK_TEMPLATE = "What action should the robot take to {task}?"


def _task_query(task: str) -> str:
    return _OPENVLA_TASK_TEMPLATE.format(task=task.lower())


_OPENVLA_PARAMS: dict = {
    "dtype": "bfloat16",
    "num_inference_steps": 1,
    # The dataset key whose q01/q99 stats normalise actions (BOUNDS_Q99).
    # OpenVLA-7b was trained on many OXE datasets; set this to a key present
    # in the base model's norm_stats. None (the default) means use the model's
    # only dataset key if there is exactly one.
    "unnorm_key": None,
}


_OPENVLA_METADATA = ModelMetadata(
    name="openvla-7b",
    backend="pytorch",
    action_dim=0,                  # flexible: dataset supplies 7-DoF; adapter normalises internally
    action_horizon=1,              # one 7-DoF action per inference step
    action_head_type="autoregressive",
    training_paradigm="pretrained_finetune",
    requires_prompt=False,          # OpenVLA builds its own prompt internally (PurePromptBuilder + Llama-2 tokenizer); does not use the framework's task_tokenize pipeline.
    # Declared read-only for the contract (interface_dict → assembly.json): no
    # pipeline step consumes it — the prompt is assembled adapter-side from
    # upstream primitives, and requires_prompt=False keeps task_tokenize out
    # of the plan.
    language_template=_OPENVLA_TASK_TEMPLATE,
    support_lora=True,
    support_full=True,
    support_freeze=True,
    install_hint="bash scripts/install.sh --model openvla",
    # ── Interface contract ──
    # The saved final/model.pt holds merged weights only; structure + processor
    # must be reconstructed from the base checkpoint at inference time.
    inference_needs_base_checkpoint=True,
    dim_policy="flexible",         # OpenVLA adapts to the dataset's action width
    # Framework-side images are kept raw (HWC uint8) and handed to the adapter,
    # which runs Prismatic's own processor transform. Declaring "stretch" makes
    # the framework's resize step match this checkpoint's resize-naive strategy
    # (stretch to 224x224, ignoring aspect ratio): for non-224 sources the step
    # runs first and the processor's resize then sees a square input — same
    # geometry as upstream's own transform, only redundant interpolation. A
    # checkpoint shipped with letterbox/resize-crop would need the framework's
    # resize vocabulary to grow before it can be declared honestly.
    image_resize_mode="stretch",
    vision_slots=(
        VisionSlot(
            name="primary",
            semantic_accepts=(
                "third_person", "third_person_front", "third_person_top",
                "wrist_left", "wrist_right", "wrist",
            ),
            resolution=(224, 224),
        ),
    ),
    components={
        "vision_encoder": ["vision_backbone.", "projector."],
        "llm": ["language_model."],
    },
    params=_OPENVLA_PARAMS,
)


@register_vla(_OPENVLA_METADATA)
def load_openvla(recipe: TrainRecipe, assembly) -> OpenVLAModelWrapper:
    """Factory: construct the upstream OpenVLA model and wrap it.

    Raises ImportError if the OpenVLA/Prismatic stack is not vendored.
    """
    upstream = try_import_openvla()
    if upstream is None:
        raise ImportError(
            "openvla-7b requires the vendored OpenVLA/Prismatic stack. "
            f"Install: {_OPENVLA_METADATA.install_hint}"
        )
    return _load_openvla(recipe, assembly, upstream, model_name="openvla-7b")


def _load_openvla(recipe, assembly, upstream, model_name="openvla-7b"):
    (
        OpenVLAConfig,
        OpenVLAForActionPrediction,
        PrismaticImageProcessor,
        PrismaticProcessor,
        ActionTokenizer,
        PurePromptBuilder,
        PaddedCollatorForActionPrediction,
    ) = upstream

    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModelForVision2Seq,
        AutoProcessor,
    )

    # Register OpenVLA to HF AutoClasses (the checkpoint's auto_map would need
    # remote code; we register from the vendored prismatic package instead).
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    if not recipe.model.path:
        raise ValueError(
            f"{model_name} is finetune-only (training_paradigm=pretrained_finetune): "
            "model.path must point to the base checkpoint."
        )

    cfg = TrackedConfig(recipe.model.config or {})
    dtype = cfg.get("dtype", "bfloat16")
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    processor = AutoProcessor.from_pretrained(recipe.model.path)
    model = AutoModelForVision2Seq.from_pretrained(
        recipe.model.path, torch_dtype=dtype, low_cpu_mem_usage=True
    )

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    collator = PaddedCollatorForActionPrediction(
        model_max_length=getattr(model.config, "llm_max_length", 2048),
        pad_token_id=getattr(model.config, "pad_token_id", 0) or 0,
    )
    unnorm_key = _resolve_unnorm_key(cfg.get("unnorm_key"), model, model_name)

    cfg.assert_all_consumed(model_name)

    return OpenVLAModelWrapper(
        model, processor, action_tokenizer, PurePromptBuilder, collator,
        unnorm_key=unnorm_key,
        camera_key=_resolve_primary_camera(assembly.camera_mapping),
    )


def _resolve_primary_camera(camera_mapping) -> str | None:
    """Dataset camera feeding the model's ``primary`` visual slot, if declared.

    OpenVLA is a single-camera model: assembly resolution maps the ``primary``
    slot to a data camera when the recipe provides ``camera_mapping.primary``
    (or when inference is unambiguous). Returns None when the mapping left the
    slot unmapped — the wrapper then falls back to a lone image or fails loudly
    on multi-camera observations.
    """
    for entry in camera_mapping.entries:
        if entry["model_slot"] == "primary":
            return entry["data_source"]
    return None


def _resolve_unnorm_key(cfg_value, model, model_name) -> str:
    """Resolve the dataset q01/q99 stats key, failing fast with candidates.

    Upstream's ``_check_unnorm_key`` asserts ``len(norm_stats) == 1`` when
    ``unnorm_key`` is None — on a multi-dataset base checkpoint
    (openvla/openvla-7b has ~25 OXE keys) that assert fires only on the first
    training/inference step, after loading and scanning. We validate here, at
    load time, and raise an actionable error listing the available keys so the
    user can set ``model.config.unnorm_key``.
    """
    if cfg_value is not None:
        keys = list(getattr(model, "norm_stats", {}) or {})
        if cfg_value not in keys:
            raise ValueError(
                f"{model_name}: model.config.unnorm_key={cfg_value!r} is not in "
                f"the checkpoint's norm_stats; choose from: {sorted(keys)}"
            )
        return cfg_value
    keys = list(getattr(model, "norm_stats", {}) or {})
    if len(keys) == 1:
        return keys[0]
    raise ValueError(
        f"{model_name}: checkpoint norm_stats has {len(keys)} dataset key(s) "
        f"{sorted(keys) if keys else '(none)'}; action normalization needs "
        "exactly one dataset's q01/q99 stats. Set model.config.unnorm_key to "
        "one of the listed keys."
    )

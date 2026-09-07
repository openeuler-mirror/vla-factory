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

import torch
import torch.nn as nn

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


# Key under which IDENTITY action statistics (q01=-1, q99=+1) are mounted in
# the loaded checkpoint's ``norm_stats`` — see ``_inject_identity_action_stats``.
_FINETUNE_STATS_KEY = "finetune_dataset"


class OpenVLAModelWrapper(nn.Module):
    """Thin adapter: vla ``Observation`` → upstream OpenVLA input → loss / actions.

    ``self.model`` is the upstream ``OpenVLAForActionPrediction``; ``nn.Module``
    auto-registers it as a submodule, so ``parameters()`` / ``train()`` / ``to()``
    recurse automatically. The token sequence, the checkpoint processor's
    image tensors and the dataset statistics all arrive plan-side via
    ``Observation`` / the mounted norm_stats — the adapter only derives the
    HF training target and delegates.
    """

    def __init__(
        self,
        model,
        stats_key: str = _FINETUNE_STATS_KEY,
    ):
        super().__init__()
        self.model = model
        self._backend = "prismatic"
        # Both call sites below read norm_stats under this key; the factory
        # mounts IDENTITY stats there (see _inject_identity_action_stats), so
        # predict_action decodes in the normalized space the plan expects.
        self._stats_key = stats_key

    @property
    def _device(self):
        return next(self.model.parameters()).device

    @property
    def _dtype(self):
        return next(self.model.parameters()).dtype

    @property
    def _dtype(self):
        return next(self.model.parameters()).dtype

    # ── Protocol surface (framework vocabulary) ───────────────────

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

    def compute_loss(self, observation, actions, action_is_pad=None):
        return self._compute_loss_openvla(observation, actions, action_is_pad)

    def predict_actions(self, observation, **kwargs):
        return self._predict_openvla(observation)

    # ── OpenVLA-specific translation + delegation ─────────────────

    def _compute_loss_openvla(self, observation, actions, action_is_pad=None):
        # The plan's assemble_token_action_sequence step produced the full
        # training sequence: tokenized_prompt holds input_ids (padded to the
        # declared tokenizer_max_length), tokenized_prompt_mask the attention
        # mask, token_loss_mask the supervision positions (action tokens +
        # stop), pixel_values the checkpoint processor's image tensors.
        # Labels — HF's training target — are derived from the mask; the
        # pixel dtype is cast to the weights' (bf16) here.
        input_ids = observation.tokenized_prompt
        labels = input_ids.masked_fill(~observation.token_loss_mask, IGNORE_INDEX)

        batch = {
            "input_ids": input_ids.to(self._device),
            "labels": labels.to(self._device),
            "attention_mask": observation.tokenized_prompt_mask.to(self._device),
            "pixel_values": observation.pixel_values.to(self._device, dtype=self._dtype),
        }
        out = self.model(**batch)
        return out.loss, {"loss": out.loss.item()}

    def _predict_openvla(self, observation):
        # The plan's assemble step tokenized the (answer-less) prompt; select
        # the real tokens out of the padded sequence — upstream's
        # predict_action expects the unpadded input.
        input_ids = observation.tokenized_prompt[observation.tokenized_prompt_mask]
        input_ids = input_ids.to(self._device).unsqueeze(0)

        pixel_values = observation.pixel_values.to(self._device, dtype=self._dtype)

        # predict_action decodes tokens → bin centers with IDENTITY stats
        # mounted by the factory (q01=-1, q99=+1), so it returns actions in
        # the NORMALIZED space; the plan's model_to_robot inverse
        # (unnormalize_action, real assembly statistics) finishes the round
        # trip — same division of labor as pi0.
        actions = self.model.predict_action(
            input_ids, self._stats_key, pixel_values=pixel_values
        )
        return torch.as_tensor(actions, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


# ── Registration ───────────────────────────────────────────────────

# The instruction format OpenVLA was pretrained with (mirrored from upstream
# RLDSBatchTransform). Declared read-only in ModelMetadata.language_template —
# the plan's assemble_token_action_sequence step consumes it (task lowercased
# before formatting, matching upstream).
_OPENVLA_TASK_TEMPLATE = "What action should the robot take to {task}?"


_OPENVLA_PARAMS: dict = {
    "dtype": "bfloat16",
    "num_inference_steps": 1,
    # Action normalization binds to the DATASET's own q01/q99 (assembly
    # norm_stats — same semantics as upstream fine-tuning, which computes
    # dataset statistics at train time). No tunable: there is nothing to
    # select, so the former unnorm_key is retired.
}


_OPENVLA_METADATA = ModelMetadata(
    name="openvla-7b",
    backend="pytorch",
    action_dim=0,                  # flexible: dataset supplies 7-DoF; normalized by the plan's normalize_vector
    action_horizon=1,              # one 7-DoF action per inference step
    vector_normalization="quantile",   # dataset's own q01/q99 — same semantics as upstream fine-tuning
    vector_normalization_eps=1e-6,
    action_head_type="autoregressive",
    training_paradigm="pretrained_finetune",
    requires_prompt=False,          # OpenVLA builds its own prompt internally (PurePromptBuilder + Llama-2 tokenizer); does not use the framework's task_tokenize pipeline.
    # Declared read-only for the contract (interface_dict → assembly.json).
    # Consumed by the plan's assemble_token_action_sequence step, which
    # tokenizes the instruction (+ answer, at training time) with the
    # checkpoint's own tokenizer; requires_prompt=False keeps the framework's
    # task_tokenize out of the plan.
    language_template=_OPENVLA_TASK_TEMPLATE,
    # Fixed sequence budget for the assembled training/inference tokens
    # (prompt ≲ 32 + 7 action tokens + stop; 48 pads the rest).
    tokenizer_max_length=48,
    support_lora=True,
    support_full=True,
    support_freeze=True,
    install_hint="bash scripts/install.sh --model openvla",
    # ── Interface contract ──
    # The saved final/model.pt holds merged weights only; structure + processor
    # must be reconstructed from the base checkpoint at inference time.
    inference_needs_base_checkpoint=True,
    dim_policy="flexible",         # OpenVLA adapts to the dataset's action width
    # The checkpoint's own processor owns the image contract end-to-end
    # (resize + per-tower normalize, channel-stacked for the fused
    # DINOv2+SigLIP backbone) and runs as the plan's checkpoint_image_transform
    # step on the primary camera. The framework's generic image vocabulary
    # does not touch these pixels.
    image_normalize_mode="checkpoint_processor",
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

    model = AutoModelForVision2Seq.from_pretrained(
        recipe.model.path, torch_dtype=dtype, low_cpu_mem_usage=True
    )

    _inject_identity_action_stats(model, len(assembly.norm_stats.action.q01))

    cfg.assert_all_consumed(model_name)

    return OpenVLAModelWrapper(model)

def _inject_identity_action_stats(model, action_dim: int) -> None:
    """Mount identity stats so upstream's predict_action decode is the
    identity map and returns actions in the NORMALIZED space.

    Training-side normalization runs in the plan (normalize_vector, real
    assembly statistics); at inference the plan's model_to_robot inverse
    (unnormalize_action, same statistics) finishes the round trip. Upstream's
    decode affine ``0.5*(a+1)*(q99-q01)+q01`` becomes the identity for
    q01=-1/q99=+1 — the same model-emits-normalized division of labor as pi0.
    The key itself is pure internal addressing: predict_action requires a
    norm_stats entry to exist and be named.
    """
    model.norm_stats[_FINETUNE_STATS_KEY] = {
        "action": {"q01": [-1.0] * action_dim, "q99": [1.0] * action_dim},
    }

"""OpenVLA-oft adapter — the OFT pre-fine-tuned variant of OpenVLA-7b.

OpenVLA-oft is a *separate base checkpoint* with the SAME architecture as
OpenVLA-7b (OFT = Orthogonal Fine-Tuning was applied upstream to produce the
checkpoint; it does not change the model structure). This entry reuses the
OpenVLA adapter (wrapper + loader) and only changes the metadata name.

Requires the same vendored OpenVLA/Prismatic stack:
``bash scripts/install.sh --model openvla``.
"""

from __future__ import annotations

from dataclasses import replace

from vla_factory.model.model_interface import ModelMetadata, VisionSlot
from vla_factory.model.registry import register_vla

from .openvla import (
    _OPENVLA_PARAMS,
    _OPENVLA_TASK_TEMPLATE,
    _load_openvla,
    try_import_openvla,
)

_OPENVLA_OFT_METADATA = ModelMetadata(
    name="openvla-7b-oft",
    backend="pytorch",
    # Same flexible contract as openvla-7b: DataSchema supplies the action
    # width (7-DoF), the adapter normalises internally. A fixed action_dim
    # would conflict with dim_policy='flexible' at assembly resolution.
    action_dim=0,
    action_horizon=1,
    vector_normalization="quantile",   # same as openvla-7b: dataset's own q01/q99
    vector_normalization_eps=1e-6,
    action_head_type="autoregressive",
    training_paradigm="pretrained_finetune",
    # Same adapter as openvla-7b: the prompt is built internally via
    # observation.task (PurePromptBuilder + Llama-2 tokenizer); the framework's
    # task_tokenize pipeline is not used, so requires_prompt must stay False
    # (True would trip the assembly resolver's tokenizer_max_length check).
    requires_prompt=False,
    # Same instruction format as openvla-7b (shared adapter), consumed by the
    # plan's assemble_token_action_sequence step.
    language_template=_OPENVLA_TASK_TEMPLATE,
    tokenizer_max_length=48,
    support_lora=True,
    support_full=True,
    support_freeze=True,
    install_hint="bash scripts/install.sh --model openvla",
    inference_needs_base_checkpoint=True,
    dim_policy="flexible",
    # Same image contract as openvla-7b: the checkpoint's own processor runs
    # plan-side (checkpoint_image_transform).
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


@register_vla(_OPENVLA_OFT_METADATA)
def load_openvla_oft(recipe, assembly):
    """Factory: construct the OpenVLA-oft checkpoint and wrap it (same adapter)."""
    upstream = try_import_openvla()
    if upstream is None:
        raise ImportError(
            "openvla-7b-oft requires the vendored OpenVLA/Prismatic stack. "
            f"Install: {_OPENVLA_OFT_METADATA.install_hint}"
        )
    return _load_openvla(recipe, assembly, upstream, model_name="openvla-7b-oft")

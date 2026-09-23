"""PI0 model declaration and factory (lerobot 0.5 ``PI0Policy``).

Upstream is lerobot 0.5's port of openpi's ``PI0Pytorch`` (flow-matching
action expert); the shared wrapper lives in
:mod:`vla_factory.model.adapters.lerobot_pi`. Pretrained start:
``lerobot/pi0_base`` (``model.path``). Checkpoints trained through the
former openpi route are not loadable here (state_dict nesting + saved image
range both differ — loud strict-load failure by design; see lerobot_pi).

Prompt: the task text + the trailing ``\\n`` start-of-answer token
(``language_template="{task}"``, max_length 48, state stays continuous).
Normalization: mean_std for state/actions, eps 1e-6 (checkpoint lineage —
lerobot's own normalizer uses 1e-8; guarded by test_normalize_parity).

Requires the shared pi-family environment (lerobot 0.5 + transformers 5.3)::
bash scripts/install.sh --model pi0
"""

from __future__ import annotations

from vla_factory.model.model_interface import ModelMetadata, VisionSlot
from vla_factory.model.registry import register_vla

from .lerobot_pi import (
    LEROBOT_PI_PARAMS,
    PILerobotModelWrapper,
    _try_import_lerobot_pi0,
    load_lerobot_pi,
)


_PI0_METADATA = ModelMetadata(
    name="pi0",
    backend="pytorch",
    action_dim=32,
    action_horizon=50,
    action_head_type="flow_matching",
    training_paradigm="pretrained_finetune",
    requires_prompt=True,
    support_lora=True,
    support_full=True,
    support_freeze=True,
    install_hint="bash scripts/install.sh --model pi0",
    dim_policy="padded_to_max",
    dim_policy_max=32,
    # lerobot's _preprocess_images takes [0,1] and maps to [-1,1] for SigLIP
    # itself (pad black-0 before the map ≡ black-(-1) after, matching the
    # openpi-era semantics).
    image_input_range=(0.0, 1.0),
    image_layout="CHW",
    image_resize_mode="pad",
    vector_normalization="mean_std",
    vector_normalization_eps=1e-6,
    language_template="{task}",
    tokenizer_repo="google/paligemma-3b-pt-224",
    tokenizer_max_length=48,
    control_mode_pref=("joint_pos",),
    expected_hz=50,
    vision_slots=(
        VisionSlot(
            name="base_0_rgb",
            semantic_accepts=(
                "third_person", "third_person_front", "third_person_top",
            ),
            resolution=(224, 224),
        ),
        VisionSlot(
            name="left_wrist_0_rgb",
            semantic_accepts=("wrist_left", "wrist"),
            resolution=(224, 224),
        ),
        VisionSlot(
            name="right_wrist_0_rgb",
            semantic_accepts=("wrist_right", "wrist"),
            resolution=(224, 224),
        ),
    ),
    components={
        # wrapper.model (PI0Policy) → .model (PI0Pytorch) →
        # .paligemma_with_expert → .paligemma (VLM) / .gemma_expert (action
        # expert). The finer-grained vision/language keys target the PaliGemma
        # sub-backbones alone, mirroring the pi0fast declaration.
        "llm": ["model.model.paligemma_with_expert.paligemma."],
        "action_expert": ["model.model.paligemma_with_expert.gemma_expert."],
        "vision_encoder": [
            "model.model.paligemma_with_expert.paligemma.model.vision_tower.",
        ],
        "language_model": [
            "model.model.paligemma_with_expert.paligemma.model.language_model.",
        ],
    },
    params=LEROBOT_PI_PARAMS,
)


@register_vla(_PI0_METADATA)
def load_pi0(recipe, assembly) -> PILerobotModelWrapper:
    """Construct lerobot's ``PI0Policy`` and wrap its framework boundary."""
    lerobot_pi0 = _try_import_lerobot_pi0()
    if lerobot_pi0 is None:
        raise ImportError(
            "pi0 requires lerobot>=0.5 (upstream PI0Policy). "
            f"Install: {_PI0_METADATA.install_hint}"
        )
    PI0Policy, PI0Config = lerobot_pi0
    return load_lerobot_pi(recipe, assembly, PI0Policy, PI0Config, "pi0")

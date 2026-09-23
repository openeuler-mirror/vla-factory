"""PI05 model declaration and factory (lerobot 0.5 ``PI05Policy``).

pi05 is an **independent** lerobot class (unlike the openpi era, where a
``pi05=True`` flag switched the same ``PI0Pytorch``): same
``paligemma_with_expert`` block layout, with two model-side differences —

  * the state input is part of the discrete language tokens (digitized into
    the prompt) rather than a continuous ``observation.state`` tensor —
    ``PI05Policy`` never reads that key, so the wrapper omits it
    (``include_state=False``);
  * the action expert uses adaRMSNorm to inject the flow-matching timestep,
    and ``tokenizer_max_length`` defaults to 200 (vs 48 for pi0).

Data-side differences live in this declaration's interface facts: quantile
normalization for state/actions, a 200-token prompt, and state digitization
before vector padding. The shared wrapper/loader lives in
:mod:`vla_factory.model.adapters.lerobot_pi`. Pretrained start:
``lerobot/pi05_base``. Checkpoints trained through the former openpi route
are not loadable here (see lerobot_pi).

Requires the shared pi-family environment (lerobot 0.5 + transformers 5.3)::
bash scripts/install.sh --model pi05
"""

from __future__ import annotations

from vla_factory.model.model_interface import ModelMetadata, VisionSlot
from vla_factory.model.registry import register_vla

from .lerobot_pi import (
    LEROBOT_PI_PARAMS,
    PILerobotModelWrapper,
    _try_import_lerobot_pi05,
    load_lerobot_pi,
)


_PI05_METADATA = ModelMetadata(
    name="pi05",
    backend="pytorch",
    action_dim=32,                  # openpi max_action_dim (pad target)
    action_horizon=50,              # chunk_size
    action_head_type="flow_matching",
    training_paradigm="pretrained_finetune",
    requires_prompt=True,
    support_lora=True,
    support_full=True,
    support_freeze=True,
    install_hint="bash scripts/install.sh --model pi05",
    # ── Interface contract (model-module §4.3) ──
    # Same vision/dim contract as pi0; differs in vector normalization: pi05
    # uses quantile (q01/q99 → [-1,1]) normalization (openpi use_quantile_norm).
    dim_policy="padded_to_max",
    dim_policy_max=32,
    image_input_range=(0.0, 1.0),
    image_layout="CHW",
    image_resize_mode="pad",
    vector_normalization="quantile",
    vector_normalization_eps=1e-6,
    language_template="{task}",
    tokenizer_repo="google/paligemma-3b-pt-224",
    tokenizer_max_length=200,
    prompt_includes_state=True,
    # openpi PaligemmaTokenizer lineage: the discrete-state prompt carries the
    # "Action: " answer marker itself (the default; FAST models differ).
    prompt_action_marker=True,
    control_mode_pref=("joint_pos",),
    expected_hz=50,
    vision_slots=(
        VisionSlot(name="base_0_rgb",
                   semantic_accepts=("third_person", "third_person_front", "third_person_top"),
                   resolution=(224, 224)),
        VisionSlot(name="left_wrist_0_rgb",
                   semantic_accepts=("wrist_left", "wrist"), resolution=(224, 224)),
        VisionSlot(name="right_wrist_0_rgb",
                   semantic_accepts=("wrist_right", "wrist"), resolution=(224, 224)),
    ),
    components={
        # Same PI0Pytorch block layout as pi0 (two classes, one structure).
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


@register_vla(_PI05_METADATA)
def load_pi05(recipe, assembly) -> PILerobotModelWrapper:
    """Factory: construct lerobot's ``PI05Policy`` and wrap it.

    Raises ImportError if lerobot (>=0.5, the ``pi05`` policy) is absent.
    """
    lerobot_pi05 = _try_import_lerobot_pi05()
    if lerobot_pi05 is None:
        raise ImportError(
            "pi05 requires lerobot>=0.5 (upstream PI05Policy). "
            f"Install: {_PI05_METADATA.install_hint}"
        )
    PI05Policy, PI05Config = lerobot_pi05
    return load_lerobot_pi(recipe, assembly, PI05Policy, PI05Config, "pi05")

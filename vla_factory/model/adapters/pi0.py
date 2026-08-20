"""PI0 model declaration and factory."""

from __future__ import annotations

from vla_factory.model.model_interface import ModelMetadata, VisionSlot
from vla_factory.model.registry import register_vla

from .openpi import OPENPI_PARAMS, PI0ModelWrapper, load_openpi, try_import_openpi


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
    install_hint="bash scripts/install.sh .venv pi0",
    dim_policy="padded_to_max",
    dim_policy_max=32,
    image_input_range=(-1.0, 1.0),
    image_layout="CHW",
    image_resize_mode="pad",
    vector_normalization="mean_std",
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
        "llm": ["paligemma_with_expert.paligemma."],
        "action_expert": ["paligemma_with_expert.gemma_expert."],
    },
    params=OPENPI_PARAMS,
)


@register_vla(_PI0_METADATA)
def load_pi0(recipe, assembly) -> PI0ModelWrapper:
    """Construct OpenPI's ``PI0Pytorch`` and wrap its framework boundary."""
    openpi = try_import_openpi()
    if openpi is None:
        raise ImportError(
            "pi0 requires openpi (upstream PI0Pytorch). "
            f"Install: {_PI0_METADATA.install_hint}"
        )
    PI0Pytorch, Pi0Config, _OpenpiObservation = openpi
    return load_openpi(recipe, assembly, PI0Pytorch, Pi0Config)

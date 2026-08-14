"""PI05 adapter — the pi05 variant of openpi's ``PI0Pytorch``.

pi05 is NOT a separate upstream class: openpi's ``Pi0Config(pi05=True)``
switches the same ``PI0Pytorch`` into the pi05 mode, with two model-side
differences (openpi pi0_config.py):

  * the state input is part of the discrete language tokens (digitized into
    the prompt) rather than a continuous input in the action-expert suffix;
  * the action expert uses adaRMSNorm to inject the flow-matching timestep,
    and ``max_token_len`` defaults to 200 (vs 48 for pi0).

Data-side differences live in this module's interface facts: quantile
normalisation for state/actions (openpi ``use_quantile_norm``), a 200-token
prompt, and state digitization before vector padding.

Everything else (thin composition wrapper, camera_mapping translation, weight
loading from a pytorch safetensors port such as ``lerobot/pi05_base``) is
shared by :mod:`vla_factory.model.adapters.openpi`.

Requires openpi (uv install):: bash scripts/install.sh .venv pi05
"""

from __future__ import annotations

from vla_factory.model.model_interface import ModelMetadata, VisionSlot
from vla_factory.model.registry import register_vla

from .openpi import OPENPI_PARAMS, PI0ModelWrapper, load_openpi, try_import_openpi

# pi05 shares PI0's upstream tunables. Its preprocessing differences are named
# interface facts below, not recipe-overridable config.
_PI05_PARAMS: dict = dict(OPENPI_PARAMS)

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
    install_hint="bash scripts/install.sh .venv pi05",
    # ── Interface contract (model-module §4.3) ──
    # Same vision/dim contract as pi0; differs in vector normalization: pi05
    # uses quantile (q01/q99 → [-1,1]) normalization (openpi use_quantile_norm).
    dim_policy="padded_to_max",
    dim_policy_max=32,
    image_input_range=(-1.0, 1.0),
    image_layout="CHW",
    image_resize_mode="pad",
    vector_normalization="quantile",
    language_template="{task}",
    tokenizer_repo="google/paligemma-3b-pt-224",
    tokenizer_max_length=200,
    prompt_includes_state=True,
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
        # Same PI0Pytorch top-level blocks as pi0 (one class, pi05 flag).
        "llm": ["paligemma_with_expert.paligemma."],
        "action_expert": ["paligemma_with_expert.gemma_expert."],
    },
    params=_PI05_PARAMS,
)


@register_vla(_PI05_METADATA)
def load_pi05(recipe, assembly) -> PI0ModelWrapper:
    """Factory: construct openpi ``PI0Pytorch`` in pi05 mode and wrap it.

    Raises ImportError if openpi is not installed.
    """
    openpi = try_import_openpi()
    if openpi is None:
        raise ImportError(
            "pi05 requires openpi (upstream PI0Pytorch). "
            f"Install: {_PI05_METADATA.install_hint}"
        )
    PI0Pytorch, Pi0Config, _OpenpiObservation = openpi  # noqa: F841
    return load_openpi(
        recipe, assembly, PI0Pytorch, Pi0Config, model_name="pi05", pi05=True,
    )

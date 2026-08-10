"""PI05 adapter — the pi05 variant of openpi's ``PI0Pytorch``.

pi05 is NOT a separate upstream class: openpi's ``Pi0Config(pi05=True)``
switches the same ``PI0Pytorch`` into the pi05 mode, with two model-side
differences (openpi pi0_config.py):

  * the state input is part of the discrete language tokens (digitized into
    the prompt) rather than a continuous input in the action-expert suffix;
  * the action expert uses adaRMSNorm to inject the flow-matching timestep,
    and ``max_token_len`` defaults to 200 (vs 48 for pi0).

Data-side differences live in this module's declaration: quantile normalisation
for state/actions (openpi ``use_quantile_norm``) and ``task_tokenize`` with
``discrete_state: true`` running BEFORE ``pad_dimensions`` (state is digitized
at its native dimension).

Everything else (thin composition wrapper, camera_mapping translation, weight
loading from a pytorch safetensors port such as ``lerobot/pi05_base``) is
shared with the pi0 entry — see ``entries/pi0.py``.

Requires openpi (uv install):: bash scripts/install.sh .venv pi05
"""

from __future__ import annotations

from vla_factory.model.interfaces.model import ModelMetadata, VisionSlot
from vla_factory.model.registry.registry import register_vla

from .pi0 import _PI0_PARAMS, PI0ModelWrapper, _load_pi0, _try_import_openpi

# pi05 starts from the pi0 family defaults and overrides only what openpi
# changes for the pi05 flag: the prompt carries the digitized state, so the
# tokenizer runs at max_token_len 200 (vs 48) with ``discrete_state`` on and
# BEFORE pad_dimensions — the state must be digitized at its native dimension.
_PI05_PARAMS: dict = {
    **_PI0_PARAMS,
    "transforms": {
        "inputs": [
            {"type": "image_to_float"},
            {"type": "image_layout", "to": "CHW"},
            {"type": "resize_images", "height": 224, "width": 224, "mode": "pad"},
            {"type": "normalize_vector", "fields": ["state", "actions"]},
            {"type": "task_tokenize", "max_length": 200, "discrete_state": True,
             "tokenizer_repo": "google/paligemma-3b-pt-224"},
            {"type": "pad_dimensions", "fields": ["state", "actions"]},
        ],
    },
}

_PI05_METADATA = ModelMetadata(
    name="pi05",
    backend="pytorch",
    action_dim=32,                  # openpi max_action_dim (pad target)
    action_horizon=50,              # chunk_size
    action_head_type="flow_matching",
    training_paradigm="pretrained_finetune",
    requires_prompt=True,
    requires_augmentation=False,
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
    vector_normalization="quantile",
    language_template="{task}",
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
def load_pi05(recipe, schema) -> PI0ModelWrapper:
    """Factory: construct openpi ``PI0Pytorch`` in pi05 mode and wrap it.

    Raises ImportError if openpi is not installed.
    """
    openpi = _try_import_openpi()
    if openpi is None:
        raise ImportError(
            "pi05 requires openpi (upstream PI0Pytorch). "
            f"Install: {_PI05_METADATA.install_hint}"
        )
    PI0Pytorch, Pi0Config, _OpenpiObservation = openpi  # noqa: F841
    return _load_pi0(
        recipe, schema, PI0Pytorch, Pi0Config, model_name="pi05", pi05=True,
        metadata=_PI05_METADATA,
    )

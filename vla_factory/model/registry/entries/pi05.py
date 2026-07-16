"""PI05 adapter — the pi05 variant of openpi's ``PI0Pytorch``.

pi05 is NOT a separate upstream class: openpi's ``Pi0Config(pi05=True)``
switches the same ``PI0Pytorch`` into the pi05 mode, with two model-side
differences (openpi pi0_config.py):

  * the state input is part of the discrete language tokens (digitized into
    the prompt) rather than a continuous input in the action-expert suffix;
  * the action expert uses adaRMSNorm to inject the flow-matching timestep,
    and ``max_token_len`` defaults to 200 (vs 48 for pi0).

Data-side differences live in ``config/model/pi05.yaml``: quantile
normalisation for state/actions (openpi ``use_quantile_norm``) and
``task_tokenize`` with ``discrete_state: true`` running BEFORE
``pad_dimensions`` (state is digitized at its native dimension).

Everything else (thin composition wrapper, camera_mapping translation, weight
loading from a pytorch safetensors port such as ``lerobot/pi05_base``) is
shared with the pi0 entry — see ``entries/pi0.py``.

Requires openpi (uv install):: bash scripts/install.sh .venv pi05
"""

from __future__ import annotations

from vla_factory.model.protocols.model import ModelMetadata
from vla_factory.model.registry.registry import register_vla

from .pi0 import PI0ModelWrapper, _load_pi0, _try_import_openpi

_PI05_METADATA = ModelMetadata(
    name="pi05",
    backend="pytorch",
    action_dim=32,                  # openpi max_action_dim (pad target)
    action_horizon=50,              # chunk_size
    action_head_type="flow_matching",
    architecture="decomposed",
    training_paradigm="pretrained_finetune",
    requires_prompt=True,
    requires_augmentation=False,
    support_lora=True,
    support_full=True,
    support_freeze=True,
    inference_num_steps=10,
    install_hint="bash scripts/install.sh .venv pi05",
    components={
        # Same PI0Pytorch top-level blocks as pi0 (one class, pi05 flag).
        "llm": ["paligemma_with_expert.paligemma."],
        "action_expert": ["paligemma_with_expert.gemma_expert."],
    },
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
        recipe, schema, PI0Pytorch, Pi0Config, model_name="pi05", pi05=True
    )

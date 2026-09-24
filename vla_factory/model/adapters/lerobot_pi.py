"""Shared lerobot 0.5 adapter used by the PI0 and PI0.5 declarations.

Thin **composition** adapter (NOT inheritance): holds lerobot 0.5's
``PI0Policy`` / ``PI05Policy`` (independent classes, structurally the same
``PI0Pytorch`` port — ``wrapper.model`` is the policy, ``.model`` the inner
network), translates ``vla_factory.Observation`` ↔ lerobot's batch dict, and
delegates ``forward`` / ``predict_action_chunk``. It does not rewrite the
upstream model.

Why lerobot rather than openpi (verified 2026-09): the PyTorch checkpoints
this framework actually loads (``lerobot/pi0_base`` / ``lerobot/pi05_base``)
are lerobot's ports in the first place — openpi's own checkpoints are JAX and
its PyTorch path needs an in-place transformers patch behind strict uv pins.
Wrapping lerobot aligns the upstream implementation with the weight source.
The openpi wrapper remains at :mod:`vla_factory.model.adapters.openpi` as a
reference for a future JAX engine. Old checkpoints trained through that
openpi path are NOT loadable here: their state_dict nesting differs by one
``model.`` level and their saved assemblies normalize images to [-1,1] —
both surface as loud strict-load / interface failures, by design.

lerobot batch contract (verified against lerobot 0.5.1):

* ``observation.images.<slot>`` — only resolved slots. The factory declares
  **every** declared vision slot in the config's ``input_features`` (mapped or
  not), which is what makes the policy's ``_preprocess_images`` fill absent
  slots with -1 images + zero masks; it emits present images in config order
  and appends the placeholders at the sequence tail, so the factory only
  accepts unmapped slots that trail every mapped one (see ``load_lerobot_pi``).
* ``observation.language.tokens`` + ``observation.language.attention_mask``
  — the framework-tokenized prompt, mask cast to **bool**.
* ``observation.state`` — **pi0 only**: ``prepare_state`` pads it to
  ``max_state_dim``. ``PI05Policy`` never reads state; the state lives in
  the discrete prompt the framework's ``task_tokenize`` already built.
* ``action`` — training only; the policy pads it to ``max_action_dim``
  idempotently.
* images are **[0,1] CHW**: the policy resizes with pad and maps to [-1,1]
  for SigLIP internally, so the resolved pipeline delivers [0,1] (pad
  semantics match the old [-1,1] path: black-0 pad before ×2−1 ≡ black-(-1)
  pad after).

``forward(batch) → (loss_tensor, dict)``;
``predict_action_chunk(batch, **kwargs) → (B, chunk, true_dim)`` (unpadded
to ``output_features['action'].shape[0]``, i.e. the padded 32-wide model
interface — the framework's ``model_to_robot`` plan slices it back down).

Requires the lerobot 0.5 environment (shared by the pi family)::
bash scripts/install.sh --model pi0
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from vla_factory.user_interface import TrainRecipe
from vla_factory.model.adapters.lerobot_version import (
    LEROBOT_MIN,
    lerobot_version_supported,
)
from vla_factory.model.model_interface import Observation
from vla_factory.utils.tracked_config import TrackedConfig

logger = logging.getLogger(__name__)


# lerobot batch keys (lerobot.utils.constants). Spelled out as literals — the
# wrapper must stay importable without lerobot for registry scans and tests.
_LANGUAGE_TOKENS = "observation.language.tokens"
_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"
_STATE_KEY = "observation.state"
_ACTION_KEY = "action"
_IMAGE_KEY_PREFIX = "observation.images."


# ── Detect lerobot availability ──────────────────────────────────────


def _try_import_lerobot_pi0():
    """Return ``(PI0Policy, PI0Config)`` or None.

    Called lazily from the factory, NOT at module top-level — importing this
    entry (registry scan / ``list_entries()``) must not pull lerobot. Cached on
    the function object after the first call.
    """
    if getattr(_try_import_lerobot_pi0, "_cached", None) is not None:
        return _try_import_lerobot_pi0._cached  # type: ignore[attr-defined]
    if not lerobot_version_supported():
        logger.info(
            "lerobot missing or older than %s — reinstall with "
            '`pip install -e ".[pi0]"` or `bash scripts/install.sh --model pi0`',
            LEROBOT_MIN,
        )
        _try_import_lerobot_pi0._cached = None  # type: ignore[attr-defined]
        return None
    try:
        from lerobot.policies.pi0.configuration_pi0 import PI0Config
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy

        cached = (PI0Policy, PI0Config)
    except Exception as e:  # noqa: BLE001
        logger.info("lerobot pi0 not available (%s: %s)", type(e).__name__, e)
        cached = None
    _try_import_lerobot_pi0._cached = cached
    return cached


def _try_import_lerobot_pi05():
    """Return ``(PI05Policy, PI05Config)`` or None (see _try_import_lerobot_pi0)."""
    if getattr(_try_import_lerobot_pi05, "_cached", None) is not None:
        return _try_import_lerobot_pi05._cached  # type: ignore[attr-defined]
    if not lerobot_version_supported():
        logger.info(
            "lerobot missing or older than %s — reinstall with "
            '`pip install -e ".[pi0]"` or `bash scripts/install.sh --model pi0`',
            LEROBOT_MIN,
        )
        _try_import_lerobot_pi05._cached = None  # type: ignore[attr-defined]
        return None
    try:
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        cached = (PI05Policy, PI05Config)
    except Exception as e:  # noqa: BLE001
        logger.info("lerobot pi05 not available (%s: %s)", type(e).__name__, e)
        cached = None
    _try_import_lerobot_pi05._cached = cached
    return cached


# ── Wrapper (nn.Module, satisfies VLAModelPyTorch; composition, not inheritance) ──


class PILerobotModelWrapper(nn.Module):
    """Thin adapter: vla ``Observation`` → lerobot batch → ``PI0(5)Policy``.

    ``self.model`` is lerobot's ``PI0Policy`` / ``PI05Policy``. 0.5's policies
    embed **no** dataset-stats normalization — that lives in lerobot's
    external processor pipeline, which the framework's own transform pipeline
    replaces — so the wrapper is the only translation layer and nothing is
    normalized twice.

    ``camera_mapping`` is the resolved ``{model_slot: data_camera}``
    correspondence.  Only mapped slots are placed in the batch; unmapped ones
    are omitted and the policy substitutes its -1 image + zero mask placeholder
    for every declared slot missing from the batch — the factory must have
    declared all slots in the config's ``input_features`` for that substitution
    to exist (a slot absent from ``config.image_features`` is simply never
    padded, shrinking the image-token sequence against the checkpoint).

    ``include_state`` selects the pi0/pi05 split: pi0 feeds the continuous
    ``observation.state`` vector, pi05 carries the state inside the discrete
    prompt (``prompt_includes_state``) and never receives the tensor.
    """

    def __init__(
        self,
        model: nn.Module,
        camera_mapping: dict[str, str] | None = None,
        include_state: bool = True,
    ) -> None:
        super().__init__()
        self.model = model  # lerobot PI0Policy / PI05Policy
        self._backend = "lerobot"
        self._camera_mapping = dict(camera_mapping or {})
        self._image_keys = [
            _IMAGE_KEY_PREFIX + slot for slot in self._camera_mapping
        ]
        self._include_state = include_state

    def forward(self, observation, actions, action_is_pad=None):
        return self.compute_loss(observation, actions, action_is_pad=action_is_pad)

    def gradient_checkpointing_enable(self, *args, **kwargs):
        # Enable checkpointing on the HF sub-backbones by name suffix, not by
        # attribute path: the LoRA strategy wraps component subtrees with
        # peft, which inserts LoraModel levels into the chain and breaks
        # PI0Pytorch's hardcoded paligemma/gemma_expert lookups. Three leaves:
        # the PaliGemma vision tower + language model and the flow-matching
        # action expert (its GemmaModel is what upstream's manual layer-loop
        # gate reads). The flag mirrors upstream so its own checkpoint wrapper
        # also stays active.
        enabled = False
        for name, module in self.model.named_modules():
            leaf = name.rsplit(".", 1)[-1]
            if leaf in ("vision_tower", "language_model", "gemma_expert") and hasattr(
                module, "gradient_checkpointing_enable"
            ):
                module.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                enabled = True
        if hasattr(self.model, "model"):
            self.model.model.gradient_checkpointing_enabled = enabled
        return None

    def compute_loss(self, observation, actions, action_is_pad=None):
        batch = self._to_lerobot_batch(observation, actions)
        loss, loss_dict = self.model.forward(batch)
        # NB: no ``dict.get(key, <expr>)`` here — the default expression would
        # evaluate on every call (float(loss) on a grad tensor warns).
        loss_value = loss_dict["loss"] if "loss" in loss_dict else loss.detach()
        return loss, {"loss": float(loss_value)}

    def predict_actions(self, observation, **kwargs):
        # The engine passes num_steps=; lerobot's flow-matching step count is
        # config.num_inference_steps (a declared tunable), so swallow the kwarg.
        batch = self._to_lerobot_batch(observation)
        return self.model.predict_action_chunk(batch)

    # ── vla Observation → lerobot batch ─────────────────────────────

    def _to_lerobot_batch(self, observation: Observation, actions=None):
        batch: dict = {}

        # Images: [0,1] CHW from the framework pipeline. The policy internally
        # resizes/pads to its image_resolution and maps to [-1,1] for SigLIP.
        for key in self._image_keys:
            slot = key[len(_IMAGE_KEY_PREFIX):]
            cam = self._camera_mapping.get(slot)
            img = observation.images.get(cam) if cam else None
            if img is not None:
                batch[key] = img

        # Language: the framework's task_tokenize already produced the prompt
        # (pi0: task + "\n"; pi05: the discrete-state "Task: ..., State: ..."
        # prefix). Pass the tokens through untouched.
        if observation.tokenized_prompt is None:
            raise ValueError(
                "pi0/pi05 require tokenized_prompt (the framework's "
                "task_tokenize step); got None."
            )
        batch[_LANGUAGE_TOKENS] = observation.tokenized_prompt
        if observation.tokenized_prompt_mask is not None:
            # lerobot's policies consume a bool attention mask (their own
            # processor casts with .to(torch.bool)); the framework's
            # tokenized_prompt_mask is long.
            batch[_LANGUAGE_ATTENTION_MASK] = observation.tokenized_prompt_mask.to(torch.bool)

        # State: pi0 only — PI05Policy never reads observation.state (the
        # state rides inside the discrete prompt instead).
        if self._include_state and observation.state is not None:
            batch[_STATE_KEY] = observation.state

        # Actions: the normalized, framework-padded chunk (training only). The
        # policy pads to max_action_dim again, idempotently.
        if actions is not None:
            batch[_ACTION_KEY] = actions

        return batch


# ── Strict pretrained loading ────────────────────────────────────────


def load_pretrained_strict(policy: nn.Module, path: str, model_name: str) -> None:
    """Resolve ``path`` and load its weights into ``policy`` strictly.

    Why not ``PolicyCls.from_pretrained``: the pi-family overrides (lerobot
    0.5.1, PI0/PI05/PI0Fast all three) wrap the whole download + read +
    ``load_state_dict`` in ``try/except`` blocks that *print* and return the
    partially-initialized or fully-random model — a wrong path, a corrupt
    file, an offline cache miss or a key mismatch (old openpi-layout
    checkpoint) all silently "succeed". This loader performs the same steps
    (lerobot's own key remap included) with every failure raised:

    * ``cached_file`` raises when the checkpoint cannot be resolved;
    * ``load_file`` raises when the safetensors file is unreadable;
    * ``load_state_dict(..., strict=True)`` raises on any key mismatch —
      the openpi-ported checkpoints carry every weight explicitly (no
      tied-weight gaps), so torch's strict check holds for them.
    """
    from safetensors.torch import load_file
    from transformers.utils import cached_file

    resolved = cached_file(path, "model.safetensors")
    state_dict = load_file(resolved)
    # lerobot's own openpi→pytorch key fixes (the same method the upstream
    # overrides call); fall back to identity if a future rename removes it —
    # the strict load below then fails loudly on the unmapped keys instead.
    fix_keys = getattr(policy, "_fix_pytorch_state_dict_keys", None)
    if fix_keys is not None:
        state_dict = fix_keys(state_dict, policy.config)
    remapped = {
        key if key.startswith("model.") else f"model.{key}": value
        for key, value in state_dict.items()
    }
    policy.load_state_dict(remapped, strict=True)


# ── Shared family defaults ───────────────────────────────────────────


# Tunable defaults shared by the lerobot PI0 family declarations. Recipes
# override any of these through ``model.config``; the named ModelMetadata
# fields are facts and are not overridable.
LEROBOT_PI_PARAMS: dict = {
    "dtype": "bfloat16",
    "paligemma_variant": "gemma_2b",
    "action_expert_variant": "gemma_300m",
    # Flow-matching denoising steps at inference (also read by the inference
    # engine's num_steps plumbing; the config value is what the policy uses).
    "num_inference_steps": 10,
    # torch.compile for forward + sampling (lerobot compile_model). Off by
    # default: compile benchmarking on the first call needs significant VRAM;
    # enable only on large cards.
    "compile_model": False,
}


def load_lerobot_pi(
    recipe: TrainRecipe, assembly, PolicyCls, ConfigCls, model_name: str,
) -> PILerobotModelWrapper:
    """Shared loader for lerobot 0.5's ``PI0Policy`` / ``PI05Policy``.

    ``model_name`` (``"pi0"`` / ``"pi05"``) only picks the state-split: pi0
    gets ``include_state=True``. Everything upstream-side differs through the
    two already-resolved classes.
    """
    from lerobot.configs.types import FeatureType, PolicyFeature

    # Every shape is the resolved composition's answer (architecture §4.2.6):
    # widths come from model_io_spec, the slot→camera correspondence from the
    # resolved CameraMapping — never from the schema or the recipe here.
    io_spec = assembly.model_io_spec
    action_horizon = int(io_spec.action_horizon)
    # Declaration order = the pretrained checkpoint's image-token order: the
    # mapping keeps one entry per declared vision slot, mapped or not.
    all_slots = [entry["model_slot"] for entry in assembly.camera_mapping.entries]
    camera_mapping = {
        entry["model_slot"]: entry["data_source"]
        for entry in assembly.camera_mapping.entries
        if entry.get("data_source")
    }
    if not camera_mapping:
        raise ValueError(
            f"{model_name} needs at least one camera, but the resolved camera "
            "mapping has no data source for any vision slot."
        )
    # lerobot's placeholder ordering constraint: ``_preprocess_images`` emits
    # present images in config order and appends missing-slot placeholders at
    # the END of the sequence. An unmapped slot that is not trailing would
    # shift every later real camera into the wrong pretrained position — the
    # images would silently swap identities. Refuse instead.
    unmapped = [slot for slot in all_slots if slot not in camera_mapping]
    if unmapped and unmapped != all_slots[-len(unmapped):]:
        raise ValueError(
            f"{model_name}: unmapped vision slots {unmapped} do not trail the "
            f"mapped ones in the declared slot order {all_slots}. The policy "
            "appends missing-slot placeholder images at the end of the image "
            "sequence, so a non-trailing unmapped slot would shift later real "
            "cameras into the wrong pretrained position. Map the slots in "
            "declaration order (or map all of them)."
        )

    cfg = TrackedConfig(
        OmegaConf.to_container(
            OmegaConf.create(recipe.model.config or {}), resolve=True
        )
    )

    # Declare ALL vision slots, not just the mapped ones: a slot absent from
    # input_features never reaches config.image_features, so the policy would
    # not generate its -1 image + zero mask placeholder and the image-token
    # sequence would shrink against the pretrained checkpoint (a 3-camera
    # checkpoint run on 2 tokens' worth of images). Placeholder ordering is
    # why unmapped slots must be trailing (checked above). State/action
    # features are auto-completed by validate_features at max_state_dim /
    # max_action_dim — the same 32 the model_io_spec pads to
    # (predict_action_chunk unpads against that output width).
    input_features = {
        f"{_IMAGE_KEY_PREFIX}{slot}": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 224, 224),
        )
        for slot in all_slots
    }

    config_kwargs = dict(
        chunk_size=action_horizon,
        n_action_steps=action_horizon,
        input_features=input_features,
        dtype=cfg.get("dtype", "bfloat16"),
        paligemma_variant=cfg.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=cfg.get("action_expert_variant", "gemma_300m"),
        num_inference_steps=int(cfg.get("num_inference_steps", 10)),
        compile_model=bool(cfg.get("compile_model", False)),
    )
    # Every declared key must have been read; a leftover means the declaration
    # carries a knob nothing consumes.
    cfg.assert_all_consumed(model_name)

    config = ConfigCls(**config_kwargs)

    if recipe.model.path:
        # Construct from our resolved config, then load weights through the
        # strict helper — never through the family's from_pretrained, whose
        # overrides print-and-continue on every load failure (see
        # ``load_pretrained_strict``). The composition stays in charge of
        # shapes regardless of what the checkpoint shipped.
        policy = PolicyCls(config)
        load_pretrained_strict(policy, recipe.model.path, model_name)
        logger.info("Loaded %s weights from %s (strict)", model_name, recipe.model.path)
    else:
        # finetune-only; inference path constructs structure, weights via
        # load_state_dict.
        logger.warning(
            "%s: model.path is None — constructing untrained structure.", model_name
        )
        policy = PolicyCls(config)

    return PILerobotModelWrapper(
        policy, camera_mapping=camera_mapping, include_state=(model_name == "pi0")
    )

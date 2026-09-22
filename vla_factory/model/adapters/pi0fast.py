"""PI0-FAST model declaration and factory (lerobot 0.5 ``PI0FastPolicy``).

π0-FAST replaces pi0's flow-matching action expert with a **FAST tokenizer**
(DCT compression + BPE): the action chunk is discretized into token ids and
predicted autoregressively by the (single) PaliGemma backbone — no expert, no
denoising steps.

Upstream selection (verified 2026-09): openpi's pinned commit ships π0-FAST as
**JAX-only** (``models/pi0_fast.py``; no ``models_pytorch/`` counterpart), so
the PyTorch path is lerobot 0.5's ``pi0_fast`` — an openpi-style port
(``PI0FastPytorch`` with the same ``paligemma_with_expert`` block layout, built
on transformers' PiGemma) plus a processor layer that owns the model-side data
preparation. This adapter wraps **lerobot's ``PI0FastPolicy``** by composition
and reuses that processor layer instead of reimplementing it.

Pretrained starts (probed by measuring the loaded weights, not by repo name):
``lerobot/pi0fast-libero`` is the only FAST-trained PyTorch checkpoint on the
Hub and loads directly via ``model.path``; ``lerobot/pi0fast-base`` is
structure-only (a PaliGemma initialization re-shelved in the pi0_fast layout —
it does not know the FAST ``Task/State/Action`` convention).

Division of labour (the resolved assembly decides, this adapter translates):

* language/state — the framework pipeline runs the **pi05-style discrete-state
  prompt** (``task_tokenize`` with ``discrete_state=True``): the quantile-
  normalized state is digitized into 256 bins over [-1, 1] and embedded as
  ``Task: <task>, State: <bins>;\n`` — the same prefix lerobot's
  ``Pi0FastPrepareStateAndLanguageTokenizerProcessorStep`` produces (both trace
  to openpi ``FASTTokenizer.tokenize``). Unlike pi05 there is **no
  ``Action: `` marker** at the prompt's end (``prompt_action_marker=False``):
  lerobot's action tokenizer puts ``<bos>Action: `` at the head of the action
  segment itself. The wrapper forwards the resulting ``tokenized_prompt``
  tensors untouched.
* actions — FAST encoding is *model-side* by design (it couples to the
  tokenizer's vocabulary mapping and to the decoding path), so the wrapper
  calls lerobot's ``ActionTokenizerProcessorStep._tokenize_action`` on the
  normalized, framework-padded action chunk at training time and lerobot's
  ``detokenize_actions`` (via ``predict_action_chunk``) at inference.
* images — [0, 1] CHW from the framework pipeline; the policy converts to
  [-1, 1] and pads geometry internally. Unmapped vision slots are simply left
  out of the batch — the policy fills them with -1 images + zero masks.

Normalization contract mirrors pi05: **quantile** (q01/q99 → [-1, 1]) for
state/actions. The 256-bin digitization and lerobot's normalizer both assume
that value range; mean_std would overflow it and get clipped.

Requires the ``[pi0fast]`` extra::

    pip install -e ".[pi0fast]"
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from vla_factory.model.model_interface import ModelMetadata, Observation, VisionSlot
from vla_factory.model.registry import register_vla
from vla_factory.user_interface import TrainRecipe
from vla_factory.utils.tracked_config import TrackedConfig

logger = logging.getLogger(__name__)


# lerobot batch keys (lerobot.utils.constants). Spelled out as literals — the
# wrapper must stay importable without lerobot for registry scans and tests.
_LANGUAGE_TOKENS = "observation.language.tokens"
_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"
_ACTION_TOKENS = "action.tokens"
_ACTION_TOKEN_MASK = "action.token_mask"
_IMAGE_KEY_PREFIX = "observation.images."


# ── Detect lerobot availability ──────────────────────────────────────


def _try_import_lerobot_pi0fast():
    """Return ``(PI0FastPolicy, PI0FastConfig, ActionTokenizerProcessorStep)`` or None.

    Called lazily from the factory, NOT at module top-level — importing this
    entry (registry scan / ``list_entries()``) must not pull lerobot. Cached on
    the function object after the first call.
    """
    if getattr(_try_import_lerobot_pi0fast, "_cached", None) is not None:
        return _try_import_lerobot_pi0fast._cached  # type: ignore[attr-defined]
    try:
        from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig
        from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy
        from lerobot.processor.tokenizer_processor import ActionTokenizerProcessorStep

        cached = (PI0FastPolicy, PI0FastConfig, ActionTokenizerProcessorStep)
    except Exception as e:  # noqa: BLE001
        logger.info("lerobot pi0_fast not available (%s: %s)", type(e).__name__, e)
        cached = None
    _try_import_lerobot_pi0fast._cached = cached
    return cached


# ── Wrapper (nn.Module, satisfies VLAModelPyTorch; composition, not inheritance) ──


class PI0FASTModelWrapper(nn.Module):
    """Thin adapter: vla ``Observation`` → lerobot batch → ``PI0FastPolicy``.

    ``self.model`` is lerobot's ``PI0FastPolicy`` (tokenizer + ``PI0FastPytorch``
    + FAST detokenization). 0.5's policy embeds **no** dataset-stats
    normalization — that lives in lerobot's external processor pipeline, which
    the framework's own transform pipeline replaces — so the wrapper is the
    only translation layer and nothing is normalized twice.

    ``camera_mapping`` is the resolved ``{model_slot: data_camera}``
    correspondence.  Only mapped slots are placed in the batch; unmapped ones
    are omitted and the policy substitutes its -1 image + zero mask placeholder.
    """

    def __init__(
        self,
        model: nn.Module,
        camera_mapping: dict[str, str] | None = None,
        action_tokenizer_step: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.model = model  # lerobot PI0FastPolicy
        self._backend = "lerobot"
        self._camera_mapping = dict(camera_mapping or {})
        self._image_keys = [
            _IMAGE_KEY_PREFIX + slot for slot in self._camera_mapping
        ]
        # ActionTokenizerProcessorStep: FAST-encodes the normalized action
        # chunk into paligemma-vocab token ids + padding mask (training only).
        self._action_step = action_tokenizer_step

    def forward(self, observation, actions, action_is_pad=None):
        return self.compute_loss(observation, actions, action_is_pad=action_is_pad)

    def gradient_checkpointing_enable(self, *args, **kwargs):
        # Enable checkpointing on the two HF sub-backbones by name suffix,
        # not by attribute path: the LoRA strategy wraps component subtrees
        # with peft, which inserts LoraModel levels into the chain and breaks
        # PI0FastPytorch's hardcoded paligemma.model.{vision_tower,
        # language_model} lookups. The flag mirrors upstream so its
        # _apply_checkpoint wrapper also stays active.
        enabled = False
        for name, module in self.model.named_modules():
            leaf = name.rsplit(".", 1)[-1]
            if leaf in ("vision_tower", "language_model") and hasattr(
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
        normalized = {"loss": float(loss_value)}
        if "ce_loss" in loss_dict:
            normalized["ce_loss"] = float(loss_dict["ce_loss"])
        return loss, normalized

    def predict_actions(self, observation, **kwargs):
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

        # Language: the framework's task_tokenize(discrete_state=True) already
        # produced the "Task: ..., State: <bins>;" paligemma token sequence.
        if observation.tokenized_prompt is None:
            raise ValueError(
                "pi0fast requires tokenized_prompt (the framework's "
                "task_tokenize discrete-state step); got None."
            )
        batch[_LANGUAGE_TOKENS] = observation.tokenized_prompt
        if observation.tokenized_prompt_mask is not None:
            # lerobot's pi0_fast consumes a bool attention mask (its own
            # processor casts with .to(torch.bool)); the framework's
            # tokenized_prompt_mask is long.
            batch[_LANGUAGE_ATTENTION_MASK] = observation.tokenized_prompt_mask.to(torch.bool)

        # Actions: FAST-encode the normalized, padded chunk (training only —
        # at inference the policy generates the tokens itself).
        if actions is not None:
            if self._action_step is None:
                raise ValueError(
                    "pi0fast training requires the FAST action tokenizer "
                    "(ActionTokenizerProcessorStep); wrapper built without one."
                )
            tokens, mask = self._action_step._tokenize_action(actions)
            batch[_ACTION_TOKENS] = tokens
            batch[_ACTION_TOKEN_MASK] = mask

        return batch


# ── Tunable defaults ─────────────────────────────────────────────────


# Tunables the recipe's ``model.config`` may override. FAST decoding is
# autoregressive — there are no flow-matching denoising steps to configure;
# ``max_decoding_steps`` bounds the token generation loop instead.
_PI0FAST_PARAMS: dict = {
    "dtype": "bfloat16",
    "max_decoding_steps": 256,
    "temperature": 0.0,
    "use_kv_cache": True,
    # torch.compile for forward + sampling (lerobot compile_model). Off by
    # default: lerobot's "max-autotune" compile mode needs >=24 GB VRAM to
    # benchmark on first call; enable only on large cards.
    "compile_model": False,
}


_PI0FAST_METADATA = ModelMetadata(
    name="pi0fast",
    backend="pytorch",
    action_dim=32,                  # max_action_dim (pad target)
    action_horizon=50,              # chunk_size (openpi pi0 family fact)
    action_head_type="autoregressive",
    training_paradigm="pretrained_finetune",
    requires_prompt=True,           # framework tokenizes the Task+State prompt
    support_lora=True,
    support_full=True,
    support_freeze=True,
    install_hint='pip install -e ".[pi0fast]"',
    # ── Interface contract (model-module §4.3) ──
    # Same vision/dim contract as pi0/pi05; state enters the discrete language
    # tokens (pi05-style discrete_state prompt) and quantile normalization
    # keeps state/actions in [-1, 1], which the 256-bin digitization assumes.
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
    # FAST prefix ends at ";\n" (openpi FASTTokenizer): the "Action: " marker
    # belongs to the action segment — lerobot's _tokenize_action prepends
    # "<bos>Action: " itself, so a marker in the prompt would duplicate it
    # out of distribution (verified against lerobot/pi0fast-base).
    prompt_action_marker=False,
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
        # wrapper.model (PI0FastPolicy) → .model (PI0FastPytorch) →
        # .paligemma_with_expert (PI0FastPaliGemma) → .paligemma. No action
        # expert in the FAST family — the backbone predicts action tokens.
        "llm": ["model.model.paligemma_with_expert.paligemma."],
        "vision_encoder": [
            "model.model.paligemma_with_expert.paligemma.model.vision_tower.",
        ],
        "language_model": [
            "model.model.paligemma_with_expert.paligemma.model.language_model.",
        ],
    },
    params=_PI0FAST_PARAMS,
)


@register_vla(_PI0FAST_METADATA)
def load_pi0fast(recipe: TrainRecipe, assembly) -> PI0FASTModelWrapper:
    """Factory: construct lerobot's ``PI0FastPolicy`` and wrap its boundary.

    Raises ImportError if lerobot (>=0.5, the ``pi0_fast`` policy) is absent.
    """
    lerobot_pi0fast = _try_import_lerobot_pi0fast()
    if lerobot_pi0fast is None:
        raise ImportError(
            "pi0fast requires lerobot>=0.5 (upstream pi0_fast policy). "
            f"Install: {_PI0FAST_METADATA.install_hint}"
        )
    PI0FastPolicy, PI0FastConfig, ActionTokenizerProcessorStep = lerobot_pi0fast
    from lerobot.configs.types import FeatureType, PolicyFeature

    # Every shape is the resolved composition's answer (architecture §4.2.6):
    # widths come from model_io_spec, the slot→camera correspondence from the
    # resolved CameraMapping — never from the schema or the recipe here.
    io_spec = assembly.model_io_spec
    action_horizon = int(io_spec.action_horizon)
    camera_mapping = {
        entry["model_slot"]: entry["data_source"]
        for entry in assembly.camera_mapping.entries
        if entry.get("data_source")
    }
    if not camera_mapping:
        raise ValueError(
            "pi0fast needs at least one camera, but the resolved camera "
            "mapping has no data source for any vision slot."
        )

    cfg = TrackedConfig(
        OmegaConf.to_container(
            OmegaConf.create(recipe.model.config or {}), resolve=True
        )
    )

    # Image features follow the resolved slots; state/action features are
    # auto-completed by PI0FastConfig.validate_features at max_state_dim /
    # max_action_dim — the same 32 the model_io_spec pads to.
    input_features = {}
    for slot in camera_mapping:
        input_features[f"{_IMAGE_KEY_PREFIX}{slot}"] = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 224, 224),
        )

    config_kwargs = dict(
        chunk_size=action_horizon,
        n_action_steps=action_horizon,
        input_features=input_features,
        dtype=cfg.get("dtype", "bfloat16"),
        max_decoding_steps=int(cfg.get("max_decoding_steps", 256)),
        temperature=float(cfg.get("temperature", 0.0)),
        use_kv_cache=bool(cfg.get("use_kv_cache", True)),
        compile_model=bool(cfg.get("compile_model", False)),
    )
    # Every declared key must have been read; a leftover means the declaration
    # carries a knob nothing consumes.
    cfg.assert_all_consumed("pi0fast")

    config = PI0FastConfig(**config_kwargs)

    if recipe.model.path:
        # lerobot's from_pretrained handles the openpi→pytorch key fixes and
        # the "model." prefix remap; our resolved config keeps the composition
        # in charge of shapes regardless of what the checkpoint shipped.
        policy = PI0FastPolicy.from_pretrained(recipe.model.path, config=config)
        logger.info("Loaded pi0fast weights from %s", recipe.model.path)
    else:
        # finetune-only; inference path constructs structure, weights via
        # load_state_dict.
        logger.warning(
            "pi0fast: model.path is None — constructing untrained structure."
        )
        policy = PI0FastPolicy(config)

    action_step = ActionTokenizerProcessorStep(
        action_tokenizer_name=config.action_tokenizer_name,
        max_action_tokens=config.max_action_tokens,
        fast_skip_tokens=config.fast_skip_tokens,
        paligemma_tokenizer_name=config.text_tokenizer_name,
    )

    return PI0FASTModelWrapper(
        policy, camera_mapping=camera_mapping, action_tokenizer_step=action_step
    )

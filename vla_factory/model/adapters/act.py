"""ACT adapter — pure glue wrapping lerobot's official ``ACTPolicy``.

VLA Factory owns *no* model architecture code. ACT's network (CVAE encoder /
ResNet18 backbone / transformer encoder-decoder) lives upstream in
``lerobot`` and arrives as an installed package — this entry is just the
adapter that translates between VLA Factory's :class:`Observation` dataclass
and lerobot's batch-dict format.

Requires the ``[act]`` extra::

    pip install -e ".[act]"

If lerobot is not importable, the *factory* (not the module) raises a clear
``ImportError`` so ``list_entries()`` keeps working for users who never touch
ACT (see ``registry.py``'s ``RegistryLoadError`` rationale).
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from omegaconf import OmegaConf

import torch
import torch.nn as nn

from vla_factory.model.model_interface import ModelMetadata, Observation
from vla_factory.model.registry import register_vla
from vla_factory.user_interface import TrainRecipe
from vla_factory.utils.tracked_config import TrackedConfig

logger = logging.getLogger(__name__)


# ── Detect lerobot availability ──────────────────────────────────────


def _patch_lerobot_groot():
    """Auto-patch lerobot's groot_n1.py dataclass bug.

    All lerobot versions (0.4.0 ~ 0.5.1) have a bug in GR00TN15Config:
    ``field(init=False)`` without ``default`` precedes fields with defaults,
    causing ``TypeError`` on Python >=3.12.

    This function detects the bug and patches the file in-place (one-time).
    Safe to call repeatedly — no-ops if already patched.
    """

    # Use find_spec on the leaf package only — avoid triggering parent __init__.
    # We locate the file by finding the lerobot package root first.
    try:
        pkg_spec = importlib.util.find_spec("lerobot")
    except (ModuleNotFoundError, ValueError):
        return

    if pkg_spec is None or pkg_spec.submodule_search_locations is None:
        return

    # Construct path directly: lerobot/policies/groot/groot_n1.py
    pkg_root = Path(list(pkg_spec.submodule_search_locations)[0])
    groot_file = pkg_root / "policies" / "groot" / "groot_n1.py"

    if not groot_file.exists():
        return

    try:
        src = groot_file.read_text()
    except OSError:
        return

    # Check if the bug is present (unpatched)
    BUG_PATTERN = "field(init=False, metadata="
    if BUG_PATTERN not in src:
        return  # already patched or different code

    patched = src.replace(
        "field(init=False, metadata=",
        "field(init=False, default=None, metadata=",
    )
    try:
        groot_file.write_text(patched)
        logger.info("Auto-patched lerobot groot_n1.py dataclass bug (%s)", groot_file)
    except OSError:
        pass  # no write permission — can't patch; ACT will be unavailable


def _try_import_lerobot():
    """Try importing lerobot's ACT classes. Returns (ACTPolicy, ACTConfig) or None.

    Called lazily from the factory, NOT at module top-level — so merely
    importing this entry (registry scan / list_entries) triggers no lerobot
    import. Cached on the function object after the first call.
    """
    if getattr(_try_import_lerobot, "_cached", None) is not None:
        return _try_import_lerobot._cached  # type: ignore[attr-defined]
    try:
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.act.configuration_act import ACTConfig

        cached = (ACTPolicy, ACTConfig)
    except Exception:
        pass
    else:
        _try_import_lerobot._cached = cached  # type: ignore[attr-defined]
        return cached

    # Import failed — try patching known bugs and retry once
    _patch_lerobot_groot()

    # Clear cached broken modules so Python re-imports from patched files
    to_remove = [k for k in sys.modules if k.startswith("lerobot")]
    for k in to_remove:
        del sys.modules[k]

    try:
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.act.configuration_act import ACTConfig

        cached = (ACTPolicy, ACTConfig)
    except Exception as e:
        logger.info("lerobot not available (%s: %s)", type(e).__name__, e)
        cached = None
    _try_import_lerobot._cached = cached  # type: ignore[attr-defined]
    return cached
# ── Checkpoint loading (continued training / fine-tuning) ────────────


def _load_state_dict_file(path: str | Path) -> dict:
    """Load a state dict from a ``.pt`` or ``.safetensors`` checkpoint.

    ``final/model.pt`` (written by ``torch.save`` in ``train.py``) and
    ``checkpoint-*/model.safetensors`` (written by the HF Trainer every
    ``save_steps``) are both supported as ``model.path`` sources.
    """
    p = Path(path)
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(p))
    return torch.load(str(p), map_location="cpu", weights_only=True)


def _adapt_state_dict(state_dict: dict, target: nn.Module) -> dict:
    """Normalise a checkpoint's key prefix to match *target*'s state_dict.

    A checkpoint saved from ``ACTModelWrapper.state_dict()`` carries an extra
    ``model.`` prefix relative to the inner policy, while a native lerobot
    checkpoint has one fewer. Rather than hard-coding a prefix, this inspects
    both key sets and shifts by exactly one ``model.`` segment when that alone
    reconciles them. Anything else is returned unchanged so a genuine
    architectural mismatch surfaces as a strict ``load_state_dict`` error.
    """
    target_keys = set(target.state_dict().keys())
    incoming = set(state_dict.keys())
    if not target_keys or incoming == target_keys:
        return state_dict
    # incoming keys = target keys with one extra leading "model."
    if {"model." + k for k in target_keys} == incoming:
        return {k[len("model."):]: v for k, v in state_dict.items()}
    # target keys = incoming keys with one extra leading "model."
    if {"model." + k for k in incoming} == target_keys:
        return {"model." + k: v for k, v in state_dict.items()}
    return state_dict


def _load_pretrained_weights(target: nn.Module, path: str | Path) -> None:
    """Load checkpoint *path* into *target* (an ``ACTModelWrapper``).

    Loading into the wrapper — not the inner policy — means a checkpoint saved
    by ``train.py``'s ``torch.save(model.state_dict(), ...)`` round-trips
    exactly, so a model trained on dataset A can be used as the starting point
    for dataset B simply by setting ``model.path`` in the recipe.
    """
    state_dict = _load_state_dict_file(path)
    state_dict = _adapt_state_dict(state_dict, target)
    target.load_state_dict(state_dict, strict=True)


# ── Wrapper (nn.Module, satisfies VLAModelPyTorch protocol) ──────────


class ACTModelWrapper(nn.Module):
    """Thin adapter: Observation → lerobot batch → (loss, loss_dict) / actions.

    ``self.model`` is a lerobot ``ACTPolicy``; ``nn.Module`` auto-registers it
    as a submodule, so ``parameters()`` / ``train()`` / ``to()`` all recurse
    automatically. ``self._backend`` is fixed to ``"lerobot"`` and kept only so
    callers that introspect the backend (e.g. ``inference_engine.py``) and tests
    asserting ``== "lerobot"`` still work.
    """

    def __init__(self, model: nn.Module, image_keys: list[str] | None = None):
        super().__init__()
        self.model = model
        self._backend = "lerobot"
        # lerobot config's image feature keys (e.g. ["observation.images.top"])
        # used to map Observation camera names → lerobot expected keys
        self._image_keys = image_keys

    def forward(self, observation, actions, action_is_pad=None):
        return self.compute_loss(observation, actions, action_is_pad=action_is_pad)

    def compute_loss(self, observation, actions, action_is_pad=None):
        return self._compute_loss_lerobot(observation, actions, action_is_pad)

    def predict_actions(self, observation, **kwargs):
        return self._predict_lerobot(observation)

    # ── lerobot-specific translation ──────────────────────────────

    def _obs_to_lerobot_batch(self, observation: Observation, actions=None, action_is_pad=None):
        """Translate Observation → lerobot batch dict format.

        Maps Observation camera names to the keys registered in lerobot's config.
        For example, if config has "observation.images.top" and Observation has
        {"front": tensor}, the result will have {"observation.images.top": tensor}.
        """
        batch = {"observation.state": observation.state}

        # Map observation cameras → lerobot config keys by NAME, not position.
        # ``self._image_keys`` is in the canonical ``schema.cameras`` order the
        # model was built with (factory builds input_features from schema.cameras
        # in order). Sorting ``observation.images.keys()`` here would reorder by
        # dict order and silently swap cameras when that differs from the schema
        # order (architecture §4.2.3 forbids dictionary-order guessing) — so we
        # iterate the build-time correspondence directly and require each camera
        # to be present.
        if self._image_keys:
            for config_key in self._image_keys:
                cam = config_key.split("observation.images.", 1)[-1]
                if cam not in observation.images:
                    raise KeyError(
                        f"observation is missing expected camera {cam!r}; model expects "
                        f"{self._image_keys}, got {sorted(observation.images.keys())}."
                    )
                batch[config_key] = observation.images[cam]
        else:
            # Fallback for direct construction without config keys: use the
            # observation's own camera names as the config keys.
            for cam in observation.images:
                batch[f"observation.images.{cam}"] = observation.images[cam]

        if actions is not None:
            batch["action"] = actions
            if action_is_pad is not None:
                batch["action_is_pad"] = action_is_pad
            else:
                batch["action_is_pad"] = torch.zeros(
                    actions.shape[:2], dtype=torch.bool, device=actions.device
                )
        return batch

    def _compute_loss_lerobot(self, observation, actions, action_is_pad=None):
        batch = self._obs_to_lerobot_batch(observation, actions, action_is_pad)
        loss, loss_dict = self.model.forward(batch)
        # Normalise lerobot's loss_dict keys to a canonical set
        # (lerobot emits ``kld_loss``; we surface it as ``kl_loss``).
        normalized = {}
        for k, v in loss_dict.items():
            if k == "kld_loss":
                normalized["kl_loss"] = torch.tensor(v) if not isinstance(v, torch.Tensor) else v
            else:
                normalized[k] = torch.tensor(v) if not isinstance(v, torch.Tensor) else v
        normalized["total_loss"] = loss
        return loss, normalized

    def _predict_lerobot(self, observation):
        batch = self._obs_to_lerobot_batch(observation)
        return self.model.predict_action_chunk(batch)


# ── Registration ─────────────────────────────────────────────────────


_ACT_METADATA = ModelMetadata(
    name="act",
    backend="pytorch",
    action_head_type="regression",
    training_paradigm="from_scratch",
    requires_prompt=False,
    support_lora=False,
    support_full=True,
    support_freeze=True,
    install_hint='pip install -e ".[act]"',
    # ── Interface contract (model-module §4.3) ──
    # ACT trains its own projection layers from scratch → flexible dim; vision
    # slots follow the dataset; images are ImageNet-normalized in [0,1].
    dim_policy="flexible",
    image_input_range=(0.0, 1.0),
    image_normalize_mode="imagenet",
    image_layout="CHW",
    image_resize_mode="stretch",
    vector_normalization="mean_std",
    vector_normalization_eps=1e-8,
    control_mode_pref=("joint_pos",),
    # Trainable-component name patterns.  The wrapper holds the lerobot policy
    # as ``self.model`` and the policy holds the ACT network as ``self.model``,
    # so every parameter is prefixed ``model.model.<component>.``.
    components={
        "backbone": ["model.model.backbone."],
        "transformer": ["model.model.encoder.", "model.model.decoder."],
        "cvae": ["model.model.vae_encoder."],
    },
    # ── Tunable defaults (recipe ``model.config`` overrides these) ──
    # The framework's recommended baseline for running ACT — what VLA Factory
    # ships and evolves, not a mechanical copy of lerobot's ACTConfig defaults.
    # Only pure upstream hyperparameters live here; data-derived fields
    # (chunk_size / n_action_steps / input_features / output_features) are
    # computed by the factory from the recipe + schema, and anything omitted
    # falls back to lerobot's own ACTConfig defaults.
    params={
        # ACT trains from scratch, so its chunk length is the user's choice, not
        # a pretrained family fact — which is exactly why it is declared here
        # rather than on ModelMetadata.action_horizon (the resolver enforces
        # that split by training_paradigm). The composition resolver reads it and
        # reports it as ModelIOSpec.action_horizon.
        "action_horizon": 100,
        # ── Transformer / CVAE ──
        "dim_model": 512,
        "n_heads": 8,
        "dim_feedforward": 3200,
        "n_encoder_layers": 4,
        "n_decoder_layers": 1,
        "latent_dim": 32,
        "n_vae_encoder_layers": 4,
        "kl_weight": 10.0,
        "dropout": 0.1,
        # ACT's regression head is deterministic — one forward, no denoising.
        "num_inference_steps": 1,
        # Optional model input size for this from-scratch family. ``None``
        # means build ACT around the dataset's native camera resolutions.
        "input_image_size": None,
    },
)


@register_vla(_ACT_METADATA)
def load_act(recipe, assembly) -> ACTModelWrapper:
    """Factory: create ACT model via lerobot's official ``ACTPolicy``.

    Args:
        recipe: TrainRecipe — checkpoint selection + this model's tunables.
        assembly: ResolvedAssembly — IO spec, camera mapping, dataset schema.

    Returns:
        ACTModelWrapper (nn.Module + VLAModelPyTorch)

    Raises:
        ImportError: if lerobot is not installed. ACT has no in-tree fallback;
            install the upstream model with ``{_ACT_METADATA.install_hint}``.
    """
    if _try_import_lerobot() is None:
        raise ImportError(
            "ACT requires lerobot (upstream model impl). "
            f"Install: {_ACT_METADATA.install_hint}"
        )
    return _load_lerobot(recipe, assembly)


def _resolve_act_config(recipe_or_config) -> TrackedConfig:
    """Resolve ACT config for adapter construction.

        Normal training/deployment passes a ``TrainRecipe`` whose ``model.config``
    already carries the merged defaults — ``merge_model_config()`` folds
    ``ModelMetadata.params`` in once at the entrypoint. The dict fallback is
    retained for direct unit tests and low-level adapter calls that pass an
    authoring-style override dictionary; there the declared params are merged
    the same way.

    Returns a :class:`TrackedConfig` so the factory can assert every declared
    key was actually read (see ``utils/tracked_config.py``).
    """
    if isinstance(recipe_or_config, TrainRecipe):
        merged = OmegaConf.create(recipe_or_config.model.config or {})
    else:
        merged = OmegaConf.merge(_ACT_METADATA.params, recipe_or_config or {})
    return TrackedConfig(OmegaConf.to_container(merged, resolve=True))


def _load_lerobot(recipe, assembly) -> ACTModelWrapper:
    """Create the model from lerobot's official ACTPolicy."""
    ACTPolicy, ACTConfig = _try_import_lerobot()
    from lerobot.configs.types import FeatureType, PolicyFeature

    # Every shape below is the resolved composition's answer, not this adapter's:
    # The IO spec is resolved from model/data facts before the pipeline is
    # planned, and the pipeline consumes the same targets. Where the composition
    # says "nothing", this fails rather than
    # inventing a default — a model built around a guessed interface trains
    # happily and wrongly.
    io_spec = assembly.model_io_spec
    action_dim = io_spec.action_dim
    action_horizon = io_spec.action_horizon
    state_dim = io_spec.state_dim
    if not state_dim:
        raise ValueError(
            "ACT requires a state input, but the resolved composition has "
            "state_dim=0 (the dataset provides no proprioceptive vector)."
        )
    # ACT declares no vision slots, so its visual inputs *are* the dataset
    # cameras — the camera mapping says so explicitly (identity entries) rather
    # than this adapter reading schema.cameras and hoping the resolver agreed.
    camera_names = [
        entry["data_source"] for entry in assembly.camera_mapping.entries
        if entry.get("data_source")
    ]
    if not camera_names:
        raise ValueError(
            "ACT needs at least one camera, but the resolved camera mapping has "
            "no data source for any slot."
        )

    # Configuration: ACT's declared ``ModelMetadata.params`` are the baseline and
    # the recipe's per-run model.config deep-merges on top (recipe wins), folded
    # in once by merge_model_config(). Unknown keys are rejected there
    # against the declared key set, and anything reaching ACTConfig that it does
    # not know raises a TypeError there — no silent typo failures either way.
    cfg = _resolve_act_config(recipe)

    # Framework-managed keys are computed from the composition and must not pass
    # through to ACTConfig.
    for fw_key in ("chunk_size", "n_action_steps", "input_features", "output_features",
                   "num_inference_steps", "action_horizon",
                   "input_image_size"):
        cfg.pop(fw_key, None)

    # Build input_features from the resolved model interface. The transform plan
    # was compiled from these same sizes.
    input_features = {
        "observation.state": PolicyFeature(
            type=FeatureType.STATE,
            shape=(state_dim,),
        ),
    }
    for cam in camera_names:
        image_size = io_spec.camera_shapes.get(cam)
        if image_size is None:
            raise ValueError(
                f"No image size resolved for camera {cam!r}: the dataset "
                "declares no resolution for it and the model declares no "
                "input_image_size. Set model.config.input_image_size, or use "
                "a reader that reports camera resolutions."
            )
        input_features[f"observation.images.{cam}"] = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, *image_size),
        )

    # NOTE: PolicyFeature.type must be FeatureType enum (not string) — lerobot
    # uses `is` identity comparison in PreTrainedConfig.image_features.
    # Remaining cfg keys are pure model hyperparameters; any the declaration and
    # recipe both omit fall back to ACTConfig's own defaults.
    config = ACTConfig(
        chunk_size=action_horizon,
        n_action_steps=action_horizon,
        input_features=input_features,
        output_features={
            "action": PolicyFeature(
                type=FeatureType.ACTION,
                shape=(action_dim,),
            ),
        },
        **cfg,
    )
    # Every declared key must have been read by now (``**cfg`` counts as a read
    # for the pass-through hyperparameters); a leftover means the declaration
    # carries a knob nothing consumes.
    cfg.assert_all_consumed("act")

    policy = ACTPolicy(config)

    # Collect config image keys for camera name mapping
    image_keys = [k for k in config.input_features if k.startswith("observation.images.")]

    wrapper = ACTModelWrapper(policy, image_keys=image_keys)
    if recipe.model.path:
        # Load into the wrapper so a checkpoint saved by train.py's
        # torch.save(model.state_dict()) round-trips exactly (continued
        # training: use a model trained on dataset A as the init for B).
        _load_pretrained_weights(wrapper, recipe.model.path)

    return wrapper

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

from vla_factory.model.protocols.model import ModelMetadata
from vla_factory.model.protocols.observation import Observation
from vla_factory.model.registry.registry import register_vla
from vla_factory.config.defaults import load_model_defaults
from vla_factory.config.recipe import TrainRecipe

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
    """Try importing lerobot's ACT classes. Returns (ACTPolicy, ACTConfig) or None."""
    try:
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.act.configuration_act import ACTConfig

        return ACTPolicy, ACTConfig
    except Exception:
        pass

    # Import failed — try patching known bugs and retry once
    _patch_lerobot_groot()

    # Clear cached broken modules so Python re-imports from patched files
    to_remove = [k for k in sys.modules if k.startswith("lerobot")]
    for k in to_remove:
        del sys.modules[k]

    try:
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.act.configuration_act import ACTConfig

        return ACTPolicy, ACTConfig
    except Exception as e:
        logger.info("lerobot not available (%s: %s)", type(e).__name__, e)
        return None


_LEROBOT_ACT = _try_import_lerobot()


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
    callers that introspect the backend (e.g. ``deploy/infer.py``) and tests
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

        # Map observation cameras → config image keys by position
        obs_cameras = sorted(observation.images.keys())
        config_keys = self._image_keys or [
            f"observation.images.{cam}" for cam in obs_cameras
        ]
        for obs_cam, config_key in zip(obs_cameras, config_keys):
            batch[config_key] = observation.images[obs_cam]

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
    requires_augmentation=True,
    support_lora=False,
    support_full=True,
    support_freeze=True,
    inference_num_steps=1,
    install_hint='pip install -e ".[act]"',
    # Trainable-component name patterns.  The wrapper holds the lerobot policy
    # as ``self.model`` and the policy holds the ACT network as ``self.model``,
    # so every parameter is prefixed ``model.model.<component>.``.
    components={
        "backbone": ["model.model.backbone."],
        "transformer": ["model.model.encoder.", "model.model.decoder."],
        "cvae": ["model.model.vae_encoder."],
    },
)


@register_vla(_ACT_METADATA)
def load_act(recipe, schema) -> ACTModelWrapper:
    """Factory: create ACT model via lerobot's official ``ACTPolicy``.

    Args:
        recipe: TrainRecipe with all training configuration.
        schema: DataSchema with dataset metadata (cameras, state_dim, etc.).

    Returns:
        ACTModelWrapper (nn.Module + VLAModelPyTorch)

    Raises:
        ImportError: if lerobot is not installed. ACT has no in-tree fallback;
            install the upstream model with ``{_ACT_METADATA.install_hint}``.
    """
    if _LEROBOT_ACT is None:
        raise ImportError(
            "ACT requires lerobot (upstream model impl). "
            f"Install: {_ACT_METADATA.install_hint}"
        )
    return _load_lerobot(recipe, schema)


def _resolve_act_config(recipe_or_config) -> dict:
    """Resolve ACT config for adapter construction.

    Normal training/deployment passes a ``TrainRecipe`` whose ``model_config``
    has already been prepared by the entrypoint or loaded from checkpoint
    metadata. The dict fallback is retained for direct unit tests and low-level
    adapter calls that pass an authoring-style override dictionary.
    """
    if isinstance(recipe_or_config, TrainRecipe):
        return OmegaConf.to_container(
            OmegaConf.create(recipe_or_config.model_config or {}),
            resolve=True,
        )
    else:
        model_config = recipe_or_config or {}

    merged = OmegaConf.merge(load_model_defaults("act"), model_config)
    return OmegaConf.to_container(merged, resolve=True)


def _resolve_resize_image_size(cfg: dict) -> tuple[int, int] | None:
    """Return the configured resize target, or None for dataset-native size."""
    transform_inputs = (cfg.get("transforms") or {}).get("inputs") or []
    for item in transform_inputs:
        if not isinstance(item, dict) or item.get("type") != "resize_images":
            continue
        height = item.get("height")
        width = item.get("width")
        if height is None and width is None:
            return None
        if height is None or width is None:
            raise ValueError("resize_images requires both `height` and `width`.")
        height = int(height)
        width = int(width)
        if height <= 0 or width <= 0:
            raise ValueError("resize_images target height and width must be positive.")
        return height, width
    return None


def _schema_image_size(schema, camera_name: str) -> tuple[int, int]:
    image_sizes = getattr(schema, "image_sizes", {}) or {}
    size = image_sizes.get(camera_name)
    if size is None and len(image_sizes) == 1:
        size = next(iter(image_sizes.values()))
    if size is None:
        raise ValueError(
            f"Cannot infer image size for camera `{camera_name}`. "
            "Set `height` and `width` on the resize_images transform, or use "
            "a dataset reader that provides DataSchema.image_sizes."
        )
    if len(size) != 2:
        raise ValueError(f"DataSchema.image_sizes[{camera_name!r}] must be (height, width).")
    return int(size[0]), int(size[1])


def _load_lerobot(recipe, schema) -> ACTModelWrapper:
    """Create the model from lerobot's official ACTPolicy."""
    ACTPolicy, ACTConfig = _LEROBOT_ACT
    from lerobot.configs.types import FeatureType, PolicyFeature

    action_spec = recipe.action_spec
    state_dim = schema.state_dim or action_spec.action_dim
    camera_names = list(schema.cameras) if schema.cameras else ["top"]

    # Configuration: the framework default profile (vla_factory/config/model/act.yaml) is
    # the baseline; the recipe's per-run model.config deep-merges on top
    # (recipe wins). Unknown keys surface as a TypeError from ACTConfig rather
    # than being silently dropped — the upstream config object is the schema.
    cfg = _resolve_act_config(recipe)

    # Framework-managed keys are computed from the recipe/schema and must not
    # pass through to ACTConfig. Image feature sizes come from resize_images
    # when configured, otherwise from the dataset schema's native image size.
    resize_image_size = _resolve_resize_image_size(cfg)
    for fw_key in ("chunk_size", "n_action_steps", "input_features", "output_features", "transforms"):
        cfg.pop(fw_key, None)

    # Build input_features dynamically from dataset cameras
    input_features = {
        "observation.state": PolicyFeature(
            type=FeatureType.STATE,
            shape=(state_dim,),
        ),
    }
    for cam in camera_names:
        image_size = resize_image_size or _schema_image_size(schema, cam)
        input_features[f"observation.images.{cam}"] = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, *image_size),
        )

    # NOTE: PolicyFeature.type must be FeatureType enum (not string) — lerobot
    # uses `is` identity comparison in PreTrainedConfig.image_features.
    # Remaining cfg keys are pure model hyperparameters; any the profile/recipe
    # omits fall back to ACTConfig's own defaults.
    config = ACTConfig(
        chunk_size=action_spec.action_horizon,
        n_action_steps=action_spec.action_horizon,
        input_features=input_features,
        output_features={
            "action": PolicyFeature(
                type=FeatureType.ACTION,
                shape=(action_spec.action_dim,),
            ),
        },
        **cfg,
    )

    policy = ACTPolicy(config)

    # Collect config image keys for camera name mapping
    image_keys = [k for k in config.input_features if k.startswith("observation.images.")]

    wrapper = ACTModelWrapper(policy, image_keys=image_keys)
    if recipe.model_path:
        # Load into the wrapper so a checkpoint saved by train.py's
        # torch.save(model.state_dict()) round-trips exactly (continued
        # training: use a model trained on dataset A as the init for B).
        _load_pretrained_weights(wrapper, recipe.model_path)

    return wrapper

"""Diffusion Policy adapter — wraps real-stanford's ``DiffusionUnetHybridImagePolicy``.

VLA Factory owns *no* model architecture code. The diffusion UNet noise head,
the robomimic-built ResNet18 observation encoder and the DDPM training loop
all live upstream in `real-stanford/diffusion_policy
<https://github.com/real-stanford/diffusion_policy>`__ — this entry only
translates between VLA Factory's :class:`Observation` and upstream's
``batch['obs']`` / ``batch['action']`` dicts.

Contract highlights (verified against upstream @ ``5ba07ac``):

- **Observation history**: upstream conditions on the last ``To`` frames
  (``history_frames=2`` in the paper). The framework's ``n_obs_steps``
  plumbing supplies them as a time axis directly after the batch axis —
  images ``(B, To, 3, H, W)``, state ``(B, To, D)``.
- **Normalization** lives framework-side (decision D1): the model's internal
  ``LinearNormalizer`` is installed as an identity so upstream's own
  normalize/unnormalize are pass-throughs, and the ``min_max`` pipeline
  (assembly.json) is the single source of statistics.
- **Horizon**: the model predicts the full ``horizon`` actions; upstream's
  receding-horizon slicing (``n_action_steps``) belongs to the framework's
  execution layer, which already implements it.
- Upstream checkpoints (dill/Lightning ``.ckpt``) are **not** loadable in v1;
  ``model.path`` accepts checkpoints this framework produced (decision D5).

Requires the ``[diffusion_policy]`` extra — the upstream repo has no
``__init__.py`` anywhere, so a plain ``pip install git+…`` produces an empty
package. Use the installer, which downloads a pinned commit, patches its
``setup.py`` to namespace packages and installs it from that source tree::

    bash scripts/install.sh --model diffusion_policy

If the upstream is not importable, the *factory* (not the module) raises a
clear ``ImportError`` so ``list_entries()`` keeps working for users who never
touch diffusion policy (see ``registry.py``'s ``RegistryLoadError`` rationale).
"""

from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import OmegaConf

import torch
import torch.nn as nn

from vla_factory.model.model_interface import ModelMetadata, Observation
from vla_factory.model.registry import register_vla
from vla_factory.user_interface import TrainRecipe
from vla_factory.utils.tracked_config import TrackedConfig

logger = logging.getLogger(__name__)


# ── Detect upstream availability ─────────────────────────────────────


def _try_import_upstream():
    """Try importing the upstream policy classes, lazily.

    Called from the factory only, never at module import, so a registry scan
    stays robust on machines without robomimic/diffusers installed (see
    ``registry.py``'s ``RegistryLoadError`` rationale). Cached on the
    function object after the first call.
    """
    if getattr(_try_import_upstream, "_cached", None) is not None:
        return _try_import_upstream._cached  # type: ignore[attr-defined]
    try:
        from diffusers import DDPMScheduler
        from diffusion_policy.model.common.normalizer import (
            LinearNormalizer,
            SingleFieldLinearNormalizer,
        )
        from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
            DiffusionUnetHybridImagePolicy,
        )

        cached = (DiffusionUnetHybridImagePolicy, LinearNormalizer,
                  SingleFieldLinearNormalizer, DDPMScheduler)
    except Exception as e:
        logger.info(
            "diffusion_policy upstream not available (%s: %s)",
            type(e).__name__, e,
        )
        cached = None
    _try_import_upstream._cached = cached  # type: ignore[attr-defined]
    return cached


# ── Checkpoint loading (continued training) ──────────────────────────


def _load_state_dict_file(path: str | Path) -> dict:
    """Load a state dict from a ``.pt`` or ``.safetensors`` checkpoint."""
    p = Path(path)
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(p))
    return torch.load(str(p), map_location="cpu", weights_only=True)


def _adapt_state_dict(state_dict: dict, target: nn.Module) -> dict:
    """Shift a checkpoint's key prefix by exactly one ``policy.`` segment
    when that alone reconciles it with *target*'s state_dict.

    Same semantics as ``act.py``'s helper: a checkpoint saved from the
    wrapper carries the extra ``policy.`` prefix relative to the bare
    upstream policy, and anything that a one-segment shift cannot reconcile
    is returned unchanged so a genuine mismatch surfaces as a strict
    ``load_state_dict`` error.
    """
    target_keys = set(target.state_dict().keys())
    incoming = set(state_dict.keys())
    if not target_keys or incoming == target_keys:
        return state_dict
    if {"policy." + k for k in target_keys} == incoming:
        return {k[len("policy."):]: v for k, v in state_dict.items()}
    if {"policy." + k for k in incoming} == target_keys:
        return {"policy." + k: v for k, v in state_dict.items()}
    return state_dict


# ── Wrapper (nn.Module, satisfies VLAModelPyTorch protocol) ──────────


class DiffusionPolicyModelWrapper(nn.Module):
    """Thin adapter: Observation → upstream batch → loss / denoised actions.

    ``self.policy`` is upstream's ``DiffusionUnetHybridImagePolicy``;
    ``nn.Module`` auto-registers it as a submodule, so ``parameters()`` /
    ``train()`` / ``to()`` all recurse automatically.

    The translation below is a pure field mapping with **no shape
    re-validation**: the tensors arriving here were produced by a pipeline
    planned toward the very same ``model_io_spec`` this wrapper was built
    from, so a wrong shape can only be a framework bug — and the untrusted
    deployment boundary is validated earlier, by the inference engine
    against the checkpoint's DataSchema. The input contract (per-camera
    ``(B, To, 3, H, W)``, state ``(B, To, D)``) is pinned in the L1 test
    ``test_diffusion_policy_parity.py``, which walks the full
    resolve→pipeline→collate path and feeds the result to the real upstream:
    a contract drift goes red there, at the true consumption boundary.
    """

    def __init__(
        self,
        policy: nn.Module,
        *,
        camera_names: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.policy = policy
        self._backend = "diffusion_policy"
        self._camera_names = tuple(camera_names)

    def forward(self, observation, actions, action_is_pad=None):
        return self.compute_loss(observation, actions)

    def compute_loss(self, observation, actions, action_is_pad=None):
        """Upstream ``compute_loss`` → ``(loss, loss_dict)``.

        The upstream returns a bare 0-dim loss tensor; the framework's
        trainer contract unpacks a 2-tuple. Upstream expects *no* padding
        mask in the batch (repeat-last padding is the horizon convention),
        so ``action_is_pad`` is deliberately ignored.
        """
        batch = {
            "obs": self._observation_to_obs_dict(observation),
            "action": actions,
        }
        loss = self.policy.compute_loss(batch)
        return loss, {"diffusion_loss": loss.detach()}

    def predict_actions(self, observation, num_steps=None, **kwargs):
        """Denoise and return the **full** horizon ``(B, horizon, action_dim)``.

        Upstream also slices a receding-horizon ``n_action_steps`` window out
        of its prediction; that slicing belongs to the framework's execution
        layer (``RECEDING_HORIZON``), so the wrapper surfaces the whole
        trajectory. ``num_steps`` overrides the denoising step count for this
        call (the inference engine passes the recipe's
        ``num_inference_steps``).
        """
        if num_steps is not None:
            self.policy.num_inference_steps = int(num_steps)
        result = self.policy.predict_action(self._observation_to_obs_dict(observation))
        return result["action_pred"]

    # ── upstream translation ──────────────────────────────────────

    def _observation_to_obs_dict(self, observation: Observation) -> dict:
        """Translate an :class:`Observation` into upstream's ``obs`` dict.

        Pure mapping, no contract re-checks (see the class docstring): a
        missing camera surfaces as the natural ``KeyError`` on its name, and
        shape mismatches fail inside the upstream encoder — which is exactly
        where the L1 input-contract test anchors them.
        """
        return {
            **{camera: observation.images[camera] for camera in self._camera_names},
            "state": observation.state,
        }


# ── Registration ─────────────────────────────────────────────────────


_DIFFUSION_POLICY_METADATA = ModelMetadata(
    name="diffusion_policy",
    backend="pytorch",
    action_head_type="diffusion",
    training_paradigm="from_scratch",
    requires_prompt=False,
    support_lora=False,
    support_full=True,
    support_freeze=True,
    install_hint="bash scripts/install.sh --model diffusion_policy",
    # ── Interface contract (model-module §4.3) ──
    # The paper's CNN variant: independent ResNet18 encoders (one per camera,
    # ImageNet-pretrained, BatchNorm swapped for GroupNorm) + a
    # ConditionalUnet1D denoising head. Trained from scratch → flexible dims;
    # cameras follow the dataset; images are consumed as float [0,1] CHW with
    # NO ImageNet normalization (upstream feeds raw /255 pixels).
    dim_policy="flexible",
    image_input_range=(0.0, 1.0),
    image_normalize_mode=None,
    image_layout="CHW",
    image_resize_mode="stretch",
    vector_normalization="min_max",
    vector_normalization_eps=1e-4,
    # Diffusion Policy has no control-mode preference (paper: joint position
    # control, but the architecture is mode-agnostic).
    control_mode_pref=(),
    # The paper's condition window: the trailing To=2 frames of images+state.
    # A fact, not a tunable — changing To is a different model.
    history_frames=2,
    # Trainable-component name patterns. The wrapper holds the upstream
    # policy as ``self.policy``, so every parameter is prefixed
    # ``policy.``. The mask generator and the (identity) normalizer carry no
    # trainable parameters.
    components={
        "obs_encoder": ["policy.obs_encoder."],
        "noise_head": ["policy.model."],
    },
    # ── Tunable defaults (recipe ``model.config`` overrides these) ──
    # The paper's PushT/Lift baseline hyperparameters. ``action_horizon`` is
    # the from-scratch chunk length (the resolver reports it as
    # ModelIOSpec.action_horizon); ``n_action_steps`` is deliberately absent
    # — it is an execution-layer parameter, not a model one.
    params={
        "action_horizon": 16,
        # ── DDPM schedule (upstream defaults) ──
        "num_train_timesteps": 100,
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "beta_schedule": "squaredcos_cap_v2",
        "prediction_type": "epsilon",
        "clip_sample": True,
        # Shorter than num_train_timesteps: the DDIM-style stride the
        # scheduler takes at inference.
        "num_inference_steps": 16,
        # ── ConditionalUnet1D ──
        "down_dims": [256, 512, 1024],
        "kernel_size": 5,
        "n_groups": 8,
        "diffusion_step_embed_dim": 256,
        # ── Observation encoder ──
        # Random-crop augmentation off by default: it requires the input to
        # be ≥ crop size, a constraint the resolver does not track.
        "crop_shape": None,
        "obs_encoder_group_norm": True,
        # Optional model input size for this from-scratch family. ``None``
        # means build around the dataset's native camera resolutions.
        "input_image_size": None,
    },
)


@register_vla(_DIFFUSION_POLICY_METADATA)
def load_diffusion_policy(recipe, assembly) -> DiffusionPolicyModelWrapper:
    """Factory: create the model via upstream's ``DiffusionUnetHybridImagePolicy``.

    Args:
        recipe: TrainRecipe — checkpoint selection + this model's tunables.
        assembly: ResolvedAssembly — IO spec, camera mapping, dataset schema.

    Returns:
        DiffusionPolicyModelWrapper (nn.Module + VLAModelPyTorch)

    Raises:
        ImportError: if the diffusion_policy upstream is not installed.
    """
    upstream = _try_import_upstream()
    if upstream is None:
        raise ImportError(
            "diffusion_policy requires the real-stanford upstream package. "
            f"Install: {_DIFFUSION_POLICY_METADATA.install_hint}"
        )
    return _load_upstream(recipe, assembly)


def _resolve_config(recipe_or_config) -> TrackedConfig:
    """Resolve this model's tunables into a tracked config.

    Normal training/deployment passes a ``TrainRecipe`` whose
    ``model.config`` already carries the merged defaults
    (``merge_model_config()`` folds ``ModelMetadata.params`` in at the
    entrypoint). The dict fallback serves direct unit tests the same way.
    """
    if isinstance(recipe_or_config, TrainRecipe):
        merged = OmegaConf.create(recipe_or_config.model.config or {})
    else:
        merged = OmegaConf.merge(
            _DIFFUSION_POLICY_METADATA.params, recipe_or_config or {}
        )
    return TrackedConfig(OmegaConf.to_container(merged, resolve=True))


def _load_upstream(recipe, assembly) -> DiffusionPolicyModelWrapper:
    """Create the upstream policy and wrap it."""
    (DiffusionUnetHybridImagePolicy, LinearNormalizer,
     SingleFieldLinearNormalizer, DDPMScheduler) = _try_import_upstream()

    # Every shape is the resolved composition's answer, never this adapter's
    # guess: the IO spec was resolved from model/data facts before any
    # pipeline was planned, and the transforms consume the same targets.
    io_spec = assembly.model_io_spec
    action_dim = io_spec.action_dim
    action_horizon = io_spec.action_horizon
    state_dim = io_spec.state_dim
    n_obs_steps = io_spec.n_obs_steps
    if not state_dim:
        raise ValueError(
            "DiffusionUnetHybridImagePolicy is a hybrid (image+state) variant "
            "and requires a state input, but the resolved composition has "
            "state_dim=0 (the dataset provides no proprioceptive vector)."
        )
    camera_names = [
        entry["data_source"] for entry in assembly.camera_mapping.entries
        if entry.get("data_source")
    ]
    if not camera_names:
        raise ValueError(
            "diffusion_policy needs at least one camera, but the resolved "
            "camera mapping has no data source for any slot."
        )

    cfg = _resolve_config(recipe)
    # Framework-consumed keys never reach the upstream constructor:
    # action_horizon → io_spec.action_horizon (read above),
    # num_inference_steps → the inference engine passes it to
    # predict_actions per call, input_image_size → io_spec.camera_shapes.
    for fw_key in ("action_horizon", "num_inference_steps", "input_image_size"):
        cfg.pop(fw_key, None)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=int(cfg["num_train_timesteps"]),
        beta_start=float(cfg["beta_start"]),
        beta_end=float(cfg["beta_end"]),
        beta_schedule=str(cfg["beta_schedule"]),
        clip_sample=bool(cfg["clip_sample"]),
        prediction_type=str(cfg["prediction_type"]),
    )
    crop_shape = cfg["crop_shape"]
    if crop_shape is not None:
        crop_shape = (int(crop_shape[0]), int(crop_shape[1]))

    # shape_meta is upstream's interface declaration, built from the same
    # resolved shapes the transform pipeline targets.
    obs_meta: dict = {}
    camera_shapes: dict[str, tuple[int, int]] = {}
    for camera in camera_names:
        size = io_spec.camera_shapes.get(camera)
        if size is None:
            raise ValueError(
                f"No image size resolved for camera {camera!r}: the dataset "
                "declares no resolution for it and the model declares no "
                "input_image_size. Set model.config.input_image_size, or use "
                "a reader that reports camera resolutions."
            )
        camera_shapes[camera] = (int(size[0]), int(size[1]))
        obs_meta[camera] = {"shape": [3, size[0], size[1]], "type": "rgb"}
    obs_meta["state"] = {"shape": [state_dim], "type": "low_dim"}
    shape_meta = {"obs": obs_meta, "action": {"shape": [action_dim]}}

    policy = DiffusionUnetHybridImagePolicy(
        shape_meta=shape_meta,
        noise_scheduler=noise_scheduler,
        horizon=action_horizon,
        # The full horizon leaves the model; receding-horizon slicing is the
        # execution layer's job (see wrapper.predict_actions).
        n_action_steps=action_horizon,
        n_obs_steps=n_obs_steps,
        num_inference_steps=None,  # falls back to the scheduler's train steps
        obs_as_global_cond=True,
        crop_shape=crop_shape,
        diffusion_step_embed_dim=int(cfg["diffusion_step_embed_dim"]),
        down_dims=tuple(int(d) for d in cfg["down_dims"]),
        kernel_size=int(cfg["kernel_size"]),
        n_groups=int(cfg["n_groups"]),
        cond_predict_scale=True,
        obs_encoder_group_norm=bool(cfg["obs_encoder_group_norm"]),
        eval_fixed_crop=False,
    )
    cfg.assert_all_consumed("diffusion_policy")

    # Decision D1: normalization is framework-side. The model's internal
    # LinearNormalizer is installed as an identity, so upstream's
    # normalize/unnormalize are pass-throughs and the min_max pipeline saved
    # in assembly.json stays the single source of statistics (train and
    # inference both consume it; a checkpoint never carries a second copy).
    identity = LinearNormalizer()
    for key in (*camera_names, "state", "action"):
        identity[key] = SingleFieldLinearNormalizer.create_identity()
    policy.set_normalizer(identity)

    wrapper = DiffusionPolicyModelWrapper(
        policy,
        camera_names=tuple(camera_names),
    )
    if recipe.model.path:
        state_dict = _load_state_dict_file(recipe.model.path)
        state_dict = _adapt_state_dict(state_dict, wrapper)
        wrapper.load_state_dict(state_dict, strict=True)

    return wrapper

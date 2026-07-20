"""PI0 adapter — wraps openpi's ``PI0Pytorch`` (official pytorch port).

Thin **composition** adapter (NOT inheritance): holds a ``PI0Pytorch``
instance, translates ``vla_factory.Observation`` ↔ ``openpi.Observation``
(camera_mapping + image contract), and delegates ``forward`` / ``sample_actions``.
It does **not** rewrite ``PI0Pytorch`` — unlike RLinf's
``OpenPi0ForRLActionPrediction`` which inherits + adds RL heads (value/noise)
for its own RL use case. vla_factory only fine-tunes, so per the architecture
doc (2.2 / 5.2.4 / 6.4) a thin wrapper is the correct boundary.

Requires openpi (uv install):: bash scripts/install.sh .venv pi0

openpi contract (verified against openpi main):
  * ``PI0Pytorch.forward(observation, actions) -> loss[B, steps, motors]``
    (per-element MSE; the adapter reduces with ``.mean()``).
  * ``PI0Pytorch.sample_actions(device, observation, ...) -> actions``.
  * openpi ``Observation``: ``images`` (dict, **[-1,1] HWC**), ``image_masks``,
    ``state``, ``tokenized_prompt``, ``tokenized_prompt_mask`` — field-for-field
    ~vla ``Observation``; the only difference is image range/layout.
  * ``Pi0Config`` carries **no image_keys** — the camera contract lives in
    openpi's data transforms, not the model config. So ``camera_mapping`` roles
    are user-declared in the recipe and validated against the dataset cameras.

Image [-1,1] HWC must be produced by the transform pipeline (``image_to_float``
with ``range: [-1,1]``); the adapter does not rescale.
"""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from vla_factory.config.recipe import TrainRecipe
from vla_factory.model.protocols.model import ModelMetadata
from vla_factory.model.protocols.observation import Observation
from vla_factory.model.registry.registry import register_vla

logger = logging.getLogger(__name__)


# ── Detect openpi availability ───────────────────────────────────────


def _try_import_openpi():
    """Return (PI0Pytorch, Pi0Config, openpi.Observation) or None.

    Called lazily from the factory, NOT at module top-level — so merely
    importing this entry (registry scan / list_entries) triggers no openpi
    import. A clear ImportError is raised only when the factory is called.
    Cached on the function object after the first call.
    """
    if getattr(_try_import_openpi, "_cached", None) is not None:
        return _try_import_openpi._cached  # type: ignore[attr-defined]
    try:
        from openpi.models.model import Observation as OpenpiObservation
        from openpi.models.pi0_config import Pi0Config
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

        cached = (PI0Pytorch, Pi0Config, OpenpiObservation)
    except Exception as e:  # noqa: BLE001
        logger.info("openpi not available (%s: %s)", type(e).__name__, e)
        cached = None
    _try_import_openpi._cached = cached  # type: ignore[attr-defined]
    return cached
# ── Wrapper (nn.Module, satisfies VLAModelPyTorch; composition, not inheritance) ──


class PI0ModelWrapper(nn.Module):
    """Thin adapter: vla ``Observation`` → openpi ``Observation`` → ``PI0Pytorch``.

    ``camera_mapping`` ({openpi_role: dataset_camera}) is user-declared in the
    recipe (openpi ``Pi0Config`` has no camera contract). Only mapped roles are
    populated in the openpi Observation; PI0Pytorch consumes whatever images the
    dict holds. Images must already be [-1,1] HWC from the transform pipeline.
    """

    def __init__(
        self,
        model: nn.Module,
        camera_mapping: dict[str, str] | None = None,
        image_keys: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.model = model  # openpi PI0Pytorch
        self._backend = "openpi"
        self._camera_mapping = camera_mapping or {}
        self._image_keys = list(image_keys or list(self._camera_mapping.keys()))

    def forward(self, observation, actions, action_is_pad=None):
        return self.compute_loss(observation, actions, action_is_pad=action_is_pad)

    def gradient_checkpointing_enable(self, *args, **kwargs):
        # Delegate to the wrapped PI0Pytorch — HF Trainer calls this when
        # gradient_checkpointing is enabled. openpi's signature doesn't accept
        # HF's gradient_checkpointing_kwargs, so fall back to a bare call.
        try:
            return self.model.gradient_checkpointing_enable(*args, **kwargs)
        except TypeError:
            return self.model.gradient_checkpointing_enable()

    def compute_loss(self, observation, actions, action_is_pad=None):
        obs = self._to_openpi_observation(observation)
        loss = self.model.forward(obs, actions)  # [B, steps, motors], per-element MSE
        loss = loss.mean()
        return loss, {"loss": loss.item()}

    def predict_actions(self, observation, **kwargs):
        obs = self._to_openpi_observation(observation)
        device = next(self.model.parameters()).device
        return self.model.sample_actions(device, obs, num_steps=kwargs.get("num_steps"))

    # ── vla Observation → openpi Observation ────────────────────────

    def _to_openpi_observation(self, observation: Observation):
        _PI0Pytorch, _Pi0Config, OpenpiObservation = _try_import_openpi()  # noqa: F841
        # openpi expects a FIXED set of camera roles (IMAGE_KEYS); the observation
        # must contain all of them. Roles mapped via camera_mapping take the real
        # image; unmapped roles get an empty placeholder (-1 image + zero mask),
        # which openpi's SigLIP ignores.
        from openpi.models_pytorch.preprocessing_pytorch import IMAGE_KEYS
        # openpi's patched SigLIP casts pixel_values to patch_embedding.dtype
        # internally, so images/state pass through in float32 (transform output).
        images: dict = {}
        image_masks: dict = {}
        ref_img = next(iter(observation.images.values())) if observation.images else None
        for role in IMAGE_KEYS:
            ds_cam = self._camera_mapping.get(role)
            img = observation.images.get(ds_cam) if ds_cam else None
            if img is not None:
                images[role] = img
                m = observation.image_masks.get(ds_cam)
                image_masks[role] = m if m is not None else torch.ones(
                    img.shape[0], dtype=torch.bool, device=img.device)
            elif ref_img is not None:
                images[role] = torch.full_like(ref_img, -1.0)
                image_masks[role] = torch.zeros(
                    ref_img.shape[0], dtype=torch.bool, device=ref_img.device)

        return OpenpiObservation(
            images=images,
            image_masks=image_masks,
            state=observation.state,
            tokenized_prompt=observation.tokenized_prompt,
            tokenized_prompt_mask=observation.tokenized_prompt_mask,
        )


# ── Registration ─────────────────────────────────────────────────────


_PI0_METADATA = ModelMetadata(
    name="pi0",
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
    install_hint="bash scripts/install.sh .venv pi0",
    components={
        # openpi PI0Pytorch top-level blocks (no extra "model." wrapper prefix,
        # unlike lerobot's ported PI0Policy which nests under self.model).
        "llm": ["paligemma_with_expert.paligemma."],
        "action_expert": ["paligemma_with_expert.gemma_expert."],
    },
)


@register_vla(_PI0_METADATA)
def load_pi0(recipe, schema) -> PI0ModelWrapper:
    """Factory: construct openpi ``PI0Pytorch`` and wrap it.

    Raises ImportError if openpi is not installed (install via ``[pi0]`` extra).
    """
    openpi = _try_import_openpi()
    if openpi is None:
        raise ImportError(
            "pi0 requires openpi (upstream PI0Pytorch). "
            f"Install: {_PI0_METADATA.install_hint}"
        )
    PI0Pytorch, Pi0Config, _OpenpiObservation = openpi  # noqa: F841
    return _load_pi0(recipe, schema, PI0Pytorch, Pi0Config)


def _resolve_pi0_config(recipe: TrainRecipe, model_name: str = "pi0") -> dict:
    """Merge the framework default profile with the recipe's model.config."""
    from vla_factory.config.defaults import load_model_defaults

    merged = OmegaConf.merge(
        load_model_defaults(model_name),
        OmegaConf.create(recipe.model_config or {}),
    )
    return OmegaConf.to_container(merged, resolve=True)


def _load_pi0(
    recipe, schema, PI0Pytorch, Pi0Config,
    model_name: str = "pi0", pi05: bool = False,
) -> PI0ModelWrapper:
    """Shared loader for the openpi PI0Pytorch family.

    ``pi05=True`` selects the pi05 variant of the SAME upstream class
    (openpi Pi0Config.pi05: state enters the discrete language tokens instead
    of the continuous suffix; the action expert uses adaRMSNorm for the flow
    timestep; max_token_len defaults to 200 instead of 48).
    """
    cfg = _resolve_pi0_config(recipe, model_name)
    camera_mapping = cfg.get("camera_mapping") or {}
    dtype = cfg.get("dtype", "bfloat16")

    # pytorch_compile_mode: torch.compile mode for openpi's sample_actions
    # (inference path only; training forward is never compiled). Pass-through to
    # Pi0Config — profile default "max-autotune" matches openpi, but a 12 GB
    # card OOMs on the first call's autotune benchmarking, so recipes that just
    # smoke-test set this to null for eager. Pi0Config.__post_init__ asserts the
    # value; cfg.get returns None when the key is absent or explicitly null.
    # Older openpi commits have no such field — pass it only when the installed
    # Pi0Config declares it (there sample_actions never compiles, so dropping
    # the knob preserves behaviour).
    config_kwargs = dict(
        action_dim=cfg.get("action_dim", 32),
        action_horizon=recipe.action_spec.action_horizon,
        dtype=dtype,
        paligemma_variant=cfg.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=cfg.get("action_expert_variant", "gemma_300m"),
    )
    if pi05:
        # only forwarded when requested: openpi commits older than the pi05
        # flag then still construct pi0, and a missing field fails loudly
        # exactly when pi05 was actually asked for.
        config_kwargs["pi05"] = True
    compile_mode = cfg.get("pytorch_compile_mode")
    init_params = inspect.signature(Pi0Config.__init__).parameters
    if "pytorch_compile_mode" in init_params:
        config_kwargs["pytorch_compile_mode"] = compile_mode
    elif compile_mode is not None:
        logger.info(
            "%s: installed openpi Pi0Config has no pytorch_compile_mode field "
            "(older commit, sample_actions is never compiled) — ignoring %r.",
            model_name, compile_mode,
        )

    config = Pi0Config(**config_kwargs)
    model = PI0Pytorch(config)

    if recipe.model_path:
        _load_weights(model, recipe.model_path, model_name=model_name)
    else:
        # finetune-only; inference path constructs structure, weights via load_state_dict.
        logger.warning("%s: model.path is None — constructing untrained structure.", model_name)

    # openpi's selective bf16 (matches RLinf __init__.py): converts selected
    # PaliGemma params — incl. SigLIP layer norms — to bf16 so the
    # mixed-precision forward type-checks. Must run AFTER weight loading.
    if dtype == "bfloat16":
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

    if not camera_mapping:
        logger.warning(
            "%s: camera_mapping not set. Declare {openpi_role: dataset_camera} "
            "in recipe.model.config (openpi Pi0Config carries no camera contract).",
            model_name,
        )

    return PI0ModelWrapper(
        model, camera_mapping=camera_mapping, image_keys=list(camera_mapping.keys())
    )


def _strip_model_prefix(state_dict: dict) -> dict:
    """Strip a leading ``model.`` wrapper if every key carries it.

    openpi ``PI0Pytorch`` expects bare keys (``paligemma_with_expert.*``), but
    the readily-available pytorch checkpoint ``lerobot/pi0_base`` stores them
    under ``model.*`` (lerobot's ``PI0Policy`` nests ``PI0Pytorch`` at
    ``self.model``). openpi and lerobot ``PI0Pytorch`` are structurally
    same-source (lerobot is a port), so stripping one ``model.`` segment aligns
    the keys. Anything else is returned unchanged so a genuine mismatch surfaces.
    """
    keys = list(state_dict.keys())
    if keys and all(k.startswith("model.") for k in keys):
        return {k[len("model."):]: v for k, v in state_dict.items()}
    return state_dict


def _load_weights(model: nn.Module, model_path: str, model_name: str = "pi0") -> None:
    """Load pretrained weights into ``PI0Pytorch``.

    openpi ships jax checkpoints by default; readily-available **pytorch**
    sources are ``lerobot/pi0_base`` / ``lerobot/pi05_base`` (model.safetensors,
    lerobot's ports of the same weights). We load that safetensors, strip the
    ``model.`` wrapper (see :func:`_strip_model_prefix`), and load into openpi
    ``PI0Pytorch``. ``strict=False`` + a missing/unexpected report lets the
    smoke surface any residual key drift to be tightened.
    """
    p = Path(model_path)
    candidates: list[Path] = []
    if p.is_dir():
        candidates = sorted(p.rglob("*.safetensors"))
    elif p.is_file() and p.suffix == ".safetensors":
        candidates = [p]
    else:
        # Treat as a HuggingFace repo id (e.g. "lerobot/pi0_base"); resolve
        # model.safetensors via the HF cache (download on miss). lerobot/pi0_base
        # is a pytorch safetensors port of openpi's weights. Sharded repos ship
        # model.safetensors.index.json instead of a single file — fall back to
        # its weight_map to fetch every shard.
        try:
            from huggingface_hub import hf_hub_download
            try:
                candidates = [Path(hf_hub_download(repo_id=model_path, filename="model.safetensors"))]
            except Exception:  # noqa: BLE001
                index_file = Path(hf_hub_download(
                    repo_id=model_path, filename="model.safetensors.index.json"
                ))
                with open(index_file) as f:
                    weight_map: dict[str, str] = json.load(f)["weight_map"]
                candidates = [
                    Path(hf_hub_download(repo_id=model_path, filename=shard))
                    for shard in sorted(set(weight_map.values()))
                ]
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "hf_hub_download(%s) found neither model.safetensors nor "
                "model.safetensors.index.json: %s", model_path, e,
            )

    if candidates:
        from safetensors.torch import load_file

        # Merge every shard: sharded checkpoints (model-0000X-of-0000Y.safetensors)
        # split one state_dict across files; loading only the first would drop
        # weights silently under load_state_dict(strict=False).
        state_dict: dict[str, torch.Tensor] = {}
        for shard in candidates:
            state_dict.update(load_file(str(shard)))
        state_dict = _strip_model_prefix(state_dict)
        # lerobot's checkpoint stores the tied lm_head.weight; openpi PI0Pytorch
        # expects embed_tokens.weight. Materialize the tied weight (mirrors
        # lerobot's _fix_pytorch_state_dict_keys).
        for k in list(state_dict):
            if k.endswith("paligemma.lm_head.weight"):
                et = k.replace("lm_head.weight", "model.language_model.embed_tokens.weight")
                state_dict.setdefault(et, state_dict[k].clone())
        # lerobot/pi0_base ships a float32 checkpoint (~13 GB); PI0Pytorch is
        # built at Pi0Config.dtype (bf16). Cast weights to the model's dtype so
        # load_state_dict doesn't dtype-mismatch.
        model_dtype = next(model.parameters()).dtype
        if model_dtype != torch.float32:
            state_dict = {k: v.to(model_dtype) for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(
            "Loaded %s weights from %s (%d shard(s), missing=%d unexpected=%d)",
            model_name, candidates[0], len(candidates), len(missing), len(unexpected),
        )
        if missing or unexpected:
            logger.warning(
                "state_dict mismatch after prefix strip — verify key alignment. "
                "(missing[:3]=%s unexpected[:3]=%s)", missing[:3], unexpected[:3]
            )
        return

    raise NotImplementedError(
        f"Loading openpi weights from {model_path!r}: no .safetensors found. "
        "openpi's default jax checkpoint needs openpi.weight_loaders conversion "
        "(not wired here) — provide a pytorch safetensors checkpoint "
        "(e.g. lerobot/pi0_base)."
    )

"""VLAModel protocols and ModelMetadata.

Layering
--------
``VLAModel``          — universal, framework-agnostic: compute_loss + predict_actions
``VLAModelPyTorch``   — PyTorch-specific: extends VLAModel with nn.Module pass-through
``VLAModelJAX``       — JAX/Flax-specific: extends VLAModel with params + apply_fn

Engines only consume their corresponding sub-protocol:
  - ``PyTorchEngine`` → ``VLAModelPyTorch``
  - ``JAXEngine``     → ``VLAModelJAX``

ModelMetadata is a frozen value object that describes a model's capabilities
so the framework can make dispatch decisions without importing the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable


# ── Model metadata (static descriptor) ─────────────────────────────


@dataclass(frozen=True)
class VisionSlot:
    """One expected visual input slot (model-module §4.3 vision).

    ``semantic_accepts`` is the controlled ``CAMERA_SEMANTICS`` vocabulary (one
    definition, referenced from both model and data sides); the generalization
    ``third_person`` accepts any third-person view.
    """

    name: str
    semantic_accepts: tuple[str, ...] = ()
    required: bool = True
    resolution: tuple[int, int] | None = None
    channels: int = 3


@dataclass(frozen=True)
class ModelMetadata:
    """Full description of a model's capabilities and requirements.

    Registered once per model type; used by the engine to decide which
    transforms to apply, which strategies are valid, etc.
    """

    # ── Basic ──
    name: str
    backend: Literal["pytorch", "jax"] = "pytorch"

    # ── Action space ──
    action_dim: int = 0  # model's internal dim (may need padding)
    action_horizon: int = 0
    action_head_type: Literal[
        "flow_matching",    # PI0, PI0.5, GR00T
        "diffusion",        # Octo, TinyVLA
        "autoregressive",   # PI0-FAST, OpenVLA
        "regression",       # SmolVLA (MLP), ACT (Linear + CVAE)
    ] = "regression"


    # ── Training paradigm ──
    training_paradigm: Literal[
        "pretrained_finetune",
        "from_scratch",
    ] = "pretrained_finetune"

    # ── Trainable components (name → parameter-name patterns) ──
    components: dict[str, list[str]] = field(default_factory=dict)

    # ── Capabilities ──
    requires_prompt: bool = True
    requires_augmentation: bool = False

    # ── Fine-tuning support ──
    support_lora: bool = True
    support_full: bool = True
    support_freeze: bool = True



    # ── Dependencies / install ──
    install_hint: str = ""   # e.g. 'pip install -e ".[act]"'; "" = no extra needed

    # ── Interface contract (model-module §4.3) ──
    # Vision contract (CameraMapping model side).
    vision_slots: tuple[VisionSlot, ...] = ()
    missing_slot_policy: str = "zero_pad"       # zero_pad | drop | error
    # Image contract — the source of truth for image_to_float / image_normalize
    # transform params (decision D3); profiles no longer duplicate these.
    image_input_range: tuple[float, float] | None = None    # e.g. (-1.0, 1.0)
    image_normalize_mode: str | None = None                 # "imagenet" | None
    # Language contract.
    language_template: str | None = None                    # e.g. "{task}"
    # Proprio / action dimension + normalization contract.
    dim_policy: str = "flexible"            # fixed | padded_to_max | flexible
    dim_policy_max: int | None = None       # N for fixed / padded_to_max
    vector_normalization: str | None = None  # mean_std | quantile | min_max
    # Action contract.
    control_mode_pref: tuple[str, ...] = ()  # CONTROL_MODES, priority order
    # Temporal contract.
    expected_hz: int | None = None
    history_frames: int = 1

    # ── Tunable defaults (model-module §4.6) ──
    # Everything above is a *fact*: the resolver reads it and a recipe can never
    # override it. ``params`` is the opposite half — this model's own upstream
    # hyperparameters plus its default ``transforms`` step list, each carrying a
    # default value that the recipe's ``model.config`` block may override.
    #
    # The container is the attribute: named field ⇒ fact, ``params`` key ⇒
    # tunable. So a model author never classifies anything — framework facts
    # have names and types, everything else goes in ``params``.
    #
    # The key set doubles as the tunable allow-list: ``resolve_recipe()``
    # rejects a ``model.config`` key that is not declared here, and the factory
    # rejects a declared key that nothing reads (see ``recipe/defaults.py``).
    # ``frozen=True`` freezes the binding, not the dict — treat it as read-only.
    params: dict[str, Any] = field(default_factory=dict)


# ── Universal model protocol (framework-agnostic) ──────────────────


@runtime_checkable
class VLAModel(Protocol):
    """Minimal interface shared by *all* models regardless of framework.

    Only two methods are truly universal: compute the training loss and
    predict actions at inference time.  Everything else (parameter access,
    device management, training mode) is backend-specific and lives in
    the sub-protocols below.
    """

    def compute_loss(self, observation, actions):
        """Training forward pass.

        Returns
        -------
        loss : Tensor or jax.Array
            Scalar loss value.
        loss_dict : dict, optional
            Auxiliary metrics for logging (e.g. L1 loss, KL divergence).
            Implementations may return either ``loss`` or ``(loss, loss_dict)``.
        """
        ...

    def predict_actions(self, observation, **kwargs):
        """Inference: observation → actions [B, horizon, action_dim]."""
        ...


# ── PyTorch sub-protocol ───────────────────────────────────────────


@runtime_checkable
class VLAModelPyTorch(VLAModel, Protocol):
    """PyTorch model wrapper — consumed by ``PyTorchEngine``.

    Adds ``nn.Module`` pass-through methods required by HF Trainer and peft.
    Implement by wrapping a ``torch.nn.Module`` and delegating these calls.
    """

    def parameters(self):
        """``nn.Module.parameters()`` — used by optimizer / peft."""
        ...

    def named_parameters(self):
        """``nn.Module.named_parameters()`` — used by freeze / selective strategies."""
        ...

    def train(self, mode: bool = True):
        """``nn.Module.train()`` — toggle training / eval mode."""
        ...

    def to(self, *args, **kwargs):
        """``nn.Module.to()`` — device / dtype transfer."""
        ...


# ── JAX / Flax sub-protocol ────────────────────────────────────────


@runtime_checkable
class VLAModelJAX(VLAModel, Protocol):
    """JAX / Flax model wrapper — consumed by ``JAXEngine``.

    JAX separates model definition (``Module``) from parameters (``dict``).
    The wrapper holds both and exposes a uniform interface to the engine.
    """

    @property
    def params(self) -> dict:
        """Frozen parameter dict (``flax.core.FrozenDict`` or plain dict)."""
        ...

    @params.setter
    def params(self, value: dict) -> None:
        """Replace parameters — used by LoRA injection, optimizer update, etc."""
        ...

    @property
    def apply_fn(self) -> Any:
        """``flax.linen.Module.apply`` or equivalent stateless forward."""
        ...

    def init_params(self, rng_key: Any, observation, actions) -> dict:
        """Initialize parameters from scratch (``from_scratch`` paradigm)."""
        ...

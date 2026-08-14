"""Unified interface between VLA models and the rest of VLA Factory.

The model interface includes both declarative and runtime contracts:
``ModelMetadata`` describes a model family before construction, ``Observation``
is the format-neutral input value, and the ``VLAModel`` protocols define the
behaviour an adapter exposes.  Upstream-specific construction and translation
remain in :mod:`vla_factory.model.adapters`.

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

import json
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Generic, Literal, Protocol, TypeVar, runtime_checkable


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
    # Image contract. The resolver derives image transforms from these facts;
    # recipes never carry or override a transform step list.
    image_input_range: tuple[float, float] | None = None    # e.g. (-1.0, 1.0)
    image_normalize_mode: str | None = None                 # "imagenet" | None
    image_layout: Literal["CHW", "HWC"] | None = None
    image_resize_mode: Literal["stretch", "pad"] | None = None
    # Language contract.
    language_template: str | None = None                    # e.g. "{task}"
    tokenizer_repo: str | None = None
    tokenizer_max_length: int | None = None
    prompt_includes_state: bool = False
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
    # hyperparameters, each carrying a default value that the recipe's
    # ``model.config`` block may override. Transform operations are not tunables:
    # the resolver derives them from the named interface facts above.
    #
    # The container is the attribute: named field ⇒ fact, ``params`` key ⇒
    # tunable. So a model author never classifies anything — framework facts
    # have names and types, everything else goes in ``params``.
    #
    # The key set doubles as the tunable allow-list: ``merge_model_config()``
    # rejects a ``model.config`` key that is not declared here, and the factory
    # rejects a declared key that nothing reads (see ``recipe/model_config.py``).
    # ``frozen=True`` freezes the binding, not the dict — treat it as read-only.
    params: dict[str, Any] = field(default_factory=dict)

    INTERFACE_FIELDS: ClassVar[tuple[str, ...]] = (
        "name", "action_dim", "action_horizon", "dim_policy", "dim_policy_max",
        "vision_slots", "missing_slot_policy", "image_input_range",
        "image_normalize_mode", "image_layout", "image_resize_mode",
        "vector_normalization", "requires_prompt", "language_template",
        "tokenizer_repo", "tokenizer_max_length", "prompt_includes_state",
        "control_mode_pref", "expected_hz", "history_frames",
    )

    @classmethod
    def interface_fields(cls) -> tuple[str, ...]:
        return cls.INTERFACE_FIELDS

    def interface_dict(self) -> dict[str, Any]:
        """JSON-shaped facts that define the tensors exchanged with the model."""
        values = asdict(self)
        return json.loads(json.dumps({key: values[key] for key in self.INTERFACE_FIELDS}))


# ── Format-neutral runtime input ────────────────────────────────────


T = TypeVar("T")


@dataclass
class Observation(Generic[T]):
    """Unified observation format received by every model adapter."""

    images: dict[str, T]
    image_masks: dict[str, T]
    state: T | None = None
    tokenized_prompt: T | None = None
    tokenized_prompt_mask: T | None = None
    token_ar_mask: T | None = None
    token_loss_mask: T | None = None

    def to(self, *args, **kwargs):
        """Move every present tensor to a device or dtype."""
        return Observation(
            images={key: value.to(*args, **kwargs) for key, value in self.images.items()},
            image_masks={
                key: value.to(*args, **kwargs)
                for key, value in self.image_masks.items()
            },
            state=self.state.to(*args, **kwargs) if self.state is not None else None,
            tokenized_prompt=(
                self.tokenized_prompt.to(*args, **kwargs)
                if self.tokenized_prompt is not None else None
            ),
            tokenized_prompt_mask=(
                self.tokenized_prompt_mask.to(*args, **kwargs)
                if self.tokenized_prompt_mask is not None else None
            ),
            token_ar_mask=(
                self.token_ar_mask.to(*args, **kwargs)
                if self.token_ar_mask is not None else None
            ),
            token_loss_mask=(
                self.token_loss_mask.to(*args, **kwargs)
                if self.token_loss_mask is not None else None
            ),
        )


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

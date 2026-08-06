"""Unified observation and action spec protocols.

These are the data contracts that flow between the data pipeline and model
abstraction layers.  They are deliberately framework-agnostic (generic over T)
so the same definitions work with both PyTorch tensors and NumPy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


# ── Action specification (YAML config side) ────────────────────────


@dataclass(frozen=True)
class ActionSpec:
    """Describes the robot action space.

    This is the *runtime* representation built from the YAML ``action_spec``
    block.  It lives in the protocol layer so both the data pipeline and model
    wrapper can reference it without circular imports.
    """

    action_dim: int
    action_horizon: int
    action_type: str  # "delta_eef" | "delta_joint" | "joint_pos" | "se3" | "tokenized"
    bounds_low: tuple[float, ...] | None = None
    bounds_high: tuple[float, ...] | None = None

    @property
    def model_action_dim(self) -> int:
        """The dimension the model internally expects (may differ from robot dim).

        For MVP this is identical to ``action_dim``; models that require padding
        (e.g. PI0 expects dim=32) override this via their ModelMetadata.
        """
        return self.action_dim


# ── Observation (tensor side) ──────────────────────────────────────


@dataclass
class Observation(Generic[T]):
    """Unified observation format received by every model.

    Different models use different subsets of fields:
      - PI0 / OpenVLA: images + state + tokenized_prompt
      - ACT:           images + state  (no prompt)
    """

    images: dict[str, T]
    image_masks: dict[str, T]
    state: T | None = None
    tokenized_prompt: T | None = None
    tokenized_prompt_mask: T | None = None
    # PI0-FAST specific
    token_ar_mask: T | None = None
    token_loss_mask: T | None = None

    def to(self, *args, **kwargs):
        """Move all tensors to device/dtype (for Trainer/Accelerator compatibility)."""
        return Observation(
            images={k: v.to(*args, **kwargs) for k, v in self.images.items()},
            image_masks={k: v.to(*args, **kwargs) for k, v in self.image_masks.items()},
            state=self.state.to(*args, **kwargs) if self.state is not None else None,
            tokenized_prompt=self.tokenized_prompt.to(*args, **kwargs) if self.tokenized_prompt is not None else None,
            tokenized_prompt_mask=self.tokenized_prompt_mask.to(*args, **kwargs) if self.tokenized_prompt_mask is not None else None,
            token_ar_mask=self.token_ar_mask.to(*args, **kwargs) if self.token_ar_mask is not None else None,
            token_loss_mask=self.token_loss_mask.to(*args, **kwargs) if self.token_loss_mask is not None else None,
        )

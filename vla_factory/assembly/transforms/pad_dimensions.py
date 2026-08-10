"""Padding transform for action dimensions (numpy-based).

Some models (e.g. PI0) expect a larger action dimension than the robot
provides.  This transform zero-pads the action vector to the target size.
"""

from __future__ import annotations

import numpy as np

from .base import TransformStep, reject_fact_override
from .registry import TransformRegistry


@TransformRegistry.register("pad_dimensions")
class PadDimensions(TransformStep):
    """Zero-pad selected vector fields to ``target_dim``.

    If the current action dimension is already >= ``target_dim``, this is
    a no-op.
    """

    def __init__(
        self,
        target_dim: int,
        fields: list[str] | tuple[str, ...] = ("actions",),
    ) -> None:
        self.target_dim = target_dim
        self.fields = tuple(fields)

    @classmethod
    def from_config(cls, cfg: dict, ctx=None) -> "PadDimensions | None":
        # The pad target is the model's dimension policy, not a per-run knob:
        # it comes from ModelMetadata (dim_policy_max / action_dim) via the
        # context. Setting it in a recipe would silently contradict the model.
        reject_fact_override(cfg, "target_dim", "dim_policy_max",
                             "pad_dimensions.target_dim")
        target_dim = int(getattr(ctx, "model_action_dim", 0) or 0) if ctx is not None else 0
        fields = tuple(cfg.get("fields", ("actions",)))
        if target_dim <= 0:
            return None
        dataset_dim = getattr(ctx, "dataset_action_dim", 0) if ctx is not None else 0
        if fields == ("actions",) and dataset_dim and target_dim <= dataset_dim:
            return None
        return cls(target_dim=target_dim, fields=fields)

    def __call__(self, sample: dict) -> dict:
        if self.target_dim <= 0:
            return sample

        for field in self.fields:
            value = sample.get(field)
            if value is None:
                continue
            current_dim = value.shape[-1]
            if current_dim < self.target_dim:
                pad_size = self.target_dim - current_dim
                padding = np.zeros((*value.shape[:-1], pad_size), dtype=value.dtype)
                sample[field] = np.concatenate([value, padding], axis=-1)

        return sample

    def inverse_for_output(self, ctx=None) -> TransformStep | None:
        if "actions" not in self.fields:
            return None
        target_dim = getattr(ctx, "dataset_action_dim", 0) if ctx is not None else 0
        if target_dim <= 0:
            return None
        return UnpadAction(target_dim=target_dim)


@TransformRegistry.register("unpad_action")
class UnpadAction(TransformStep):
    """Crop model action output back to the dataset/action-spec dimension."""

    def __init__(self, target_dim: int) -> None:
        self.target_dim = int(target_dim)

    def __call__(self, sample: dict) -> dict:
        actions = sample.get("actions")
        if actions is not None and self.target_dim > 0:
            sample["actions"] = actions[..., : self.target_dim]
        return sample

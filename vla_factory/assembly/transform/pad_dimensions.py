"""Padding transform for action dimensions (numpy-based).

Some models (e.g. PI0) expect a larger action dimension than the robot
provides.  This transform zero-pads the action vector to the target size.
"""

from __future__ import annotations

import numpy as np

from .base import PlanContext, TransformStep, reject_fact_override
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
    def compile_call(cls, cfg: dict, ctx: PlanContext) -> dict | None:
        # The pad target is the model's dimension policy, not a per-run knob:
        # it comes from ModelMetadata (dim_policy_max / action_dim) via the
        # context. A second caller-supplied value would contradict the model.
        reject_fact_override(cfg, "target_dim", "dim_policy_max",
                             "pad_dimensions.target_dim")
        fields = tuple(cfg.get("fields", ("actions",)))
        targets = {
            "state": int(ctx.target_state_dim or 0),
            "actions": int(ctx.target_action_dim or 0),
        }
        sources = {
            "state": int(ctx.source_state_dim or 0),
            "actions": int(ctx.source_action_dim or 0),
        }
        unknown = sorted(set(fields) - set(targets))
        if unknown:
            raise ValueError(f"pad_dimensions has unknown fields: {unknown}")
        required = [name for name in fields if targets[name] > sources[name]]
        if not required:
            return None
        target_dims = {targets[name] for name in required}
        if len(target_dims) != 1:
            raise ValueError(
                "One pad_dimensions call cannot reconcile fields with different "
                f"target widths: { {name: targets[name] for name in required} }."
            )
        target_dim = target_dims.pop()
        return {"target_dim": target_dim, "fields": list(fields)}

    @classmethod
    def inverse_call(cls, args: dict, ctx: PlanContext) -> tuple[str, dict] | None:
        """Padding's inverse is cropping back to the dataset's own width."""
        if "actions" not in (args.get("fields") or ()):
            return None
        target_dim = int(ctx.source_action_dim or 0)
        if target_dim <= 0:
            return None
        return "unpad_action", {"target_dim": target_dim}

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

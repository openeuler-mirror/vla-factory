"""Padding transform for action dimensions (numpy-based).

Some models (e.g. PI0) expect a larger action dimension than the robot
provides.  This transform zero-pads the action vector to the target size.
"""

from __future__ import annotations

import numpy as np

from .base import TransformStep
from .registry import TransformRegistry


@TransformRegistry.register("pad_dimensions")
class PadDimensions(TransformStep):
    """Zero-pad the ``"actions"`` array to ``target_dim``.

    If the current action dimension is already >= ``target_dim``, this is
    a no-op.
    """

    def __init__(self, target_dim: int) -> None:
        self.target_dim = target_dim

    def __call__(self, sample: dict) -> dict:
        if self.target_dim <= 0:
            return sample

        actions = sample.get("actions")
        if actions is None:
            return sample

        current_dim = actions.shape[-1]
        if current_dim < self.target_dim:
            pad_size = self.target_dim - current_dim
            padding = np.zeros((*actions.shape[:-1], pad_size), dtype=actions.dtype)
            sample["actions"] = np.concatenate([actions, padding], axis=-1)

        return sample

    def state_dict(self) -> dict:
        return {"type": "pad_dimensions", "target_dim": self.target_dim}

    @classmethod
    def from_state_dict(cls, d: dict, **shared) -> "PadDimensions":
        return cls(target_dim=d["target_dim"])

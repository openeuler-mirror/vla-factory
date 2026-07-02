"""Transform base class: sample-level preprocessing/postprocessing step."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class TransformStep(ABC):
    """A single transform step.

    A step takes and returns a ``dict`` with the same flat-key structure used
    by ``VLADataset.__getitem__`` — typically ``"state"``, ``"actions"`` and
    ``"images.<cam>"`` keys.

    ``from_config`` is the normal construction hook used by YAML-driven
    pipelines. ``inverse_for_output`` is optional: input-only steps return
    ``None``; action-affecting steps can return the postprocessor step that
    maps model output back to robot action space.
    """

    @abstractmethod
    def __call__(self, sample: dict) -> dict:
        """Apply this step.  Take a sample dict, return the transformed sample."""
        ...

    @classmethod
    def from_config(cls, cfg: dict, ctx: Any | None = None) -> "TransformStep":
        """Construct from a YAML/config dictionary.

        The default implementation forwards every key except ``type`` to the
        constructor. Steps that need runtime context or skip logic override this
        method.
        """
        return cls(**{k: v for k, v in cfg.items() if k != "type"})

    def inverse_for_output(self, ctx: Any | None = None) -> "TransformStep | None":
        """Return the corresponding output postprocessor step, if any."""
        return None

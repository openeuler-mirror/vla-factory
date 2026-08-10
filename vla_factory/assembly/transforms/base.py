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


def reject_fact_override(cfg: dict, key: str, attr: str, what: str) -> None:
    """Refuse a per-run override of a model fact.

    Image range, normalize mode, vector normalization and the pad target are
    facts: the composition resolver reads them, and changing one per run makes
    the resolved assembly disagree with the model that actually runs. pi0 wants
    images in ``[-1, 1]`` because SigLIP was trained that way — a recipe setting
    ``range: [0, 1]`` used to win silently, training happily on wrong-valued
    pixels. So a fact key present in the step config is an error, not an
    override (architecture §1.7, conservative failure).
    """
    if key in cfg:
        raise ValueError(
            f"{what} is a model fact and cannot be set per run: it belongs to "
            f"ModelMetadata.{attr}, which the composition resolver reads. "
            f"Remove {key!r} from the transform config; to change it for a "
            "model family, change that model's declaration."
        )


def model_fact(cfg: dict, key: str, ctx: Any | None, attr: str, what: str) -> Any:
    """Read a model-side transform fact from ``ctx.metadata``.

    Rejects a per-run override (see :func:`reject_fact_override`). Absence is
    also an error rather than a quiet default: if the declaration does not carry
    the fact, the pipeline cannot be built.
    """
    reject_fact_override(cfg, key, attr, what)
    md = getattr(ctx, "metadata", None) if ctx is not None else None
    value = getattr(md, attr, None) if md is not None else None
    if value is None:
        raise ValueError(
            f"{what} is not declared: set it on ModelMetadata.{attr}. "
            "A silent default is intentionally not used (risk R3)."
        )
    return value

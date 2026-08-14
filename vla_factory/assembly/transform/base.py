"""Transform base class: sample-level preprocessing/postprocessing step.

A step is described in two forms, and each rule that relates them lives in
exactly one place — on the step class itself:

``TransformStepCall``  (name + resolved args, serializable)
        │  ``from_call``
        ▼
``TransformStep``      (built, executable)

* ``compile_call`` — resolver-selected policy + facts → the call's args, or ``None``
  when this step is a no-op for these facts. Called by the composition resolver
  when it plans a pipeline, and only there: deciding *what* to run is resolution,
  and it happens once.
* ``from_call``    — args (+ runtime context, for the live objects args cannot
  carry, e.g. statistics) → an executable step. Called by the training and
  inference sides, which execute a resolved plan and decide nothing.
* ``inverse_call`` — args → the ``(name, args)`` of the reverse step, or
  ``None``. The single home of every forward/inverse pairing: a pipeline is
  never inverted by reversing its list (architecture §4.2.4).

There is deliberately no "build a step straight from a declaration" entry point:
that would be a second place where a fact gets derived, and the two would drift.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PlanContext:
    """The facts a step needs to resolve its own call args.

    Deliberately flat and free of live objects, so the same context type serves
    the composition resolver (which has no recipe, dataset or GPU) and the
    build path (which derives one from its ``TransformContext``).
    """

    # The model's declaration — source of every image/vector contract fact.
    metadata: Any = None
    # Source and target interface shapes. Steps consume these decisions; they
    # do not report shapes back to the resolver.
    target_state_dim: int = 0
    target_action_dim: int = 0
    source_state_dim: int = 0
    source_action_dim: int = 0
    source_camera_shapes: dict[str, tuple[int, int]] | None = None
    target_camera_shapes: dict[str, tuple[int, int]] | None = None
    # Whether dataset statistics are available at all, and for actions
    # specifically (an inverse needs the action half).
    has_norm_stats: bool = False
    has_action_stats: bool = False
    # Language fallbacks resolved by the caller (recipe / controlled override).
    default_task: str | None = None
    tokenizer_repo: str | None = None


class TransformStep(ABC):
    """A single transform step.

    A step takes and returns a ``dict`` with the same flat-key structure used
    by ``VLADataset.__getitem__`` — typically ``"state"``, ``"actions"`` and
    ``"images.<cam>"`` keys.
    """

    @abstractmethod
    def __call__(self, sample: dict) -> dict:
        """Apply this step.  Take a sample dict, return the transformed sample."""
        ...

    # ── planning ───────────────────────────────────────────────────

    @classmethod
    def compile_call(cls, cfg: dict, ctx: PlanContext) -> dict[str, Any] | None:
        """Resolve planner-selected policy + facts into this call's args.

        ``None`` means the step is a no-op for these facts and is dropped from
        the pipeline — the skip rules (nothing to pad, no statistics, no resize
        target) live here so a plan lists only calls that really run.

        The default forwards every supplied key except ``type``; steps that
        read a model fact or have a skip rule override it.
        """
        return {k: v for k, v in cfg.items() if k != "type"}

    @classmethod
    def from_call(cls, args: dict, ctx: Any | None = None) -> "TransformStep":
        """Build the step from resolved args.

        The default passes them straight to the constructor. Steps needing a
        live object args cannot carry (statistics) override this and take it
        from the runtime context.
        """
        return cls(**args)

    @classmethod
    def inverse_call(cls, args: dict, ctx: PlanContext) -> tuple[str, dict] | None:
        """``(registered_name, args)`` of the reverse step, or ``None``.

        Input-only steps (image ops, tokenization) have no inverse and keep the
        default. A lossy step must return ``None`` rather than a lookalike.
        """
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


def model_fact(cfg: dict, key: str, ctx: PlanContext, attr: str, what: str) -> Any:
    """Read a model-side transform fact from ``ctx.metadata``.

    Rejects a per-run override (see :func:`reject_fact_override`). Absence is
    also an error rather than a quiet default: if the declaration does not carry
    the fact, the pipeline cannot be built.
    """
    reject_fact_override(cfg, key, attr, what)
    md = getattr(ctx, "metadata", None)
    value = getattr(md, attr, None) if md is not None else None
    if value is None:
        raise ValueError(
            f"{what} is not declared: set it on ModelMetadata.{attr}. "
            "A silent default is intentionally not used (risk R3)."
        )
    return value

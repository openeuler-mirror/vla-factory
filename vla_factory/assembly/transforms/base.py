"""Transform base class: sample-level preprocessing/postprocessing step.

A step is described in two forms, and each rule that relates them lives in
exactly one place — on the step class itself:

``TransformStepCall``  (name + resolved args, serializable)
        │  ``from_call``
        ▼
``TransformStep``      (built, executable)

* ``compile_call`` — declared config + facts → the call's args, or ``None``
  when this step is a no-op for these facts. The composition resolver calls it
  to plan a pipeline; ``from_config`` calls it too, so the planner and the
  build path can never drift apart.
* ``from_call``    — args (+ runtime context, for the live objects args cannot
  carry, e.g. statistics) → an executable step.
* ``inverse_call`` — args → the ``(name, args)`` of the reverse step, or
  ``None``. The single home of every forward/inverse pairing: a pipeline is
  never inverted by reversing its list (architecture §4.2.4).
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
    # Vector widths: the pad target and what the dataset actually provides.
    model_action_dim: int = 0
    dataset_action_dim: int = 0
    # Whether dataset statistics are available at all, and for actions
    # specifically (an inverse needs the action half).
    has_norm_stats: bool = False
    has_action_stats: bool = False
    # Language fallbacks resolved by the caller (recipe / controlled override).
    default_task: str | None = None
    tokenizer_repo: str | None = None

    @classmethod
    def of(cls, ctx: Any | None) -> "PlanContext":
        """Accept a ``PlanContext``, a ``TransformContext``, or ``None``."""
        if isinstance(ctx, cls):
            return ctx
        if ctx is None:
            return cls()
        plan = getattr(ctx, "plan", None)
        return plan() if callable(plan) else cls()


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
        """Resolve declared config + facts into this call's args.

        ``None`` means the step is a no-op for these facts and is dropped from
        the pipeline — the skip rules (nothing to pad, no statistics, no resize
        target) live here so a plan lists only calls that really run.

        The default forwards every declared key except ``type``; steps that
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

    @classmethod
    def output_widths(
        cls, args: dict, input_widths: dict[str, int],
    ) -> dict[str, int]:
        """Vector widths after this call, given the widths going in.

        A fold rather than an absolute answer: padding produces
        ``max(input, target)``, and a future crop or projection needs the input
        width just as much. Returning absolute widths here would put the planner
        back in the business of knowing whether a number is a floor, a cap or a
        delta — the same leak that step-name knowledge was.

        Only steps that change a vector's width override this.
        """
        return input_widths

    # ── construction from user/declaration config ──────────────────

    @classmethod
    def from_config(cls, cfg: dict, ctx: Any | None = None) -> "TransformStep | None":
        """Construct from a YAML/config dictionary.

        Composition of the two halves above, so the build path applies exactly
        the rules the planner applies. ``None`` when ``compile_call`` skips.
        """
        args = cls.compile_call(cfg, PlanContext.of(ctx))
        if args is None:
            return None
        return stamp_call_args(cls.from_call(args, ctx), args)

    def call_args(self) -> dict[str, Any]:
        """This instance's args, as ``compile_call`` produced them.

        Filled in by :func:`stamp_call_args` on every construction path, so a
        step that declares an ``inverse_call`` cannot lose its pairing at
        runtime by forgetting to implement this. Only a step that must also
        support being hand-constructed *and* inverted overrides it.
        """
        return dict(getattr(self, "_compiled_call_args", {}))

    def inverse_for_output(self, ctx: Any | None = None) -> "TransformStep | None":
        """The built postprocessor step matching this one, if any.

        Generic: asks :meth:`inverse_call` for the pairing — the same answer the
        resolver plans — then builds it.
        """
        from .registry import TransformRegistry

        inverse = type(self).inverse_call(self.call_args(), PlanContext.of(ctx))
        if inverse is None:
            return None
        name, args = inverse
        step_cls = TransformRegistry.get(name)
        return stamp_call_args(step_cls.from_call(args, ctx), args)


def stamp_call_args(step: "TransformStep | None", args: dict) -> "TransformStep | None":
    """Record the args a step was built from, on the step.

    Applied by the *outer* construction paths rather than inside ``from_call``:
    subclasses override ``from_call`` (the denormalization steps take their
    statistics from the context there), so stamping in the base implementation
    would be silently skipped exactly where an inverse matters most.
    """
    if step is not None:
        step._compiled_call_args = dict(args)
    return step


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

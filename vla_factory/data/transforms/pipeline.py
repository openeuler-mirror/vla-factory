"""TransformPipeline + the preprocessor builder.

``TransformPipeline`` is an ordered list of forward-only :class:`TransformStep`
instances.  ``build_preprocessor`` turns a model's declared transform list
(e.g. ACT's ``("normalize", "resize_images", "pad_dimensions")``) into a
concrete pipeline, injecting runtime parameters (image_size, action dims,
norm_stats) per step and skipping a step when its preconditions aren't met.

The pre/postprocessor split (auto-deriving a reverse pipeline from
``paired_reverse`` metadata) and full ``transforms.json`` round-tripping land in
a later phase; this module establishes the forward-only pipeline + the
registry-driven builder they will build on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .base import TransformStep
from .registry import TransformRegistry, _accepted_kwargs


class TransformPipeline:
    """An ordered list of forward-only transform steps."""

    def __init__(self, steps: Iterable[TransformStep] | None = None) -> None:
        self._steps: list[TransformStep] = list(steps) if steps else []

    def __call__(self, sample: dict) -> dict:
        for step in self._steps:
            sample = step(sample)
        return sample

    @property
    def steps(self) -> list[TransformStep]:
        return self._steps

    def state_dict(self) -> list[dict]:
        return [step.state_dict() for step in self._steps]

    @classmethod
    def from_state_dict(cls, d: list[dict], **shared) -> "TransformPipeline":
        """Reconstruct a pipeline from a list of step ``state_dict``s.

        ``shared`` (e.g. ``stats=norm_stats``) is forwarded to every step that
        needs it.
        """
        return cls([TransformRegistry.create(item, **shared) for item in d])

    def __len__(self) -> int:
        return len(self._steps)

    def __iter__(self):
        return iter(self._steps)

    def __getitem__(self, key):
        """Index like a list of steps (matches ``torch.nn.Sequential``).

        An int returns the i-th step; a slice returns a new
        :class:`TransformPipeline` over the selected steps.
        """
        if isinstance(key, slice):
            return TransformPipeline(self._steps[key])
        return self._steps[key]

    def __repr__(self) -> str:
        names = [type(s).__name__ for s in self._steps]
        return f"TransformPipeline({names})"


@dataclass
class BuildContext:
    """Runtime context used to instantiate a model's default transform list.

    Fields are sourced from model metadata + dataset schema.  Each transform
    type draws only what it needs: Normalize → ``norm_stats``, ResizeImages →
    ``image_size``, PadDimensions → ``model_action_dim`` / ``dataset_action_dim``.
    """

    norm_stats: Any | None = None
    image_size: tuple[int, int] | None = None
    model_action_dim: int = 0
    dataset_action_dim: int = 0


# Default ordered transform list when a model declares no ``default_transforms``.
# Keeps behaviour identical to the legacy hardcoded ``_build_transforms``: every
# step is present, and each one self-skips when its precondition isn't met.
_DEFAULT_TRANSFORM_TYPES: tuple[str, ...] = (
    "normalize",
    "resize_images",
    "pad_dimensions",
)


def build_preprocessor(
    transform_types: Iterable[str] | None,
    ctx: BuildContext,
) -> TransformPipeline:
    """Build a forward preprocessor pipeline from a model's declared transforms.

    Parameters
    ----------
    transform_types
        Ordered iterable of registered transform type names.  ``None`` or empty
        falls back to :data:`_DEFAULT_TRANSFORM_TYPES` (legacy behaviour).
    ctx
        Runtime parameters used to instantiate each step.

    Each name is resolved through the registry.  Steps whose preconditions
    aren't satisfied (no stats, zero image_size, no padding needed) are skipped
    — exactly matching the legacy ``_build_transforms`` behaviour.
    """
    if not transform_types:
        transform_types = _DEFAULT_TRANSFORM_TYPES

    steps: list[TransformStep] = []
    for name in transform_types:
        step = _instantiate(name, ctx)
        if step is not None:
            steps.append(step)
    return TransformPipeline(steps)


def _instantiate(name: str, ctx: BuildContext) -> TransformStep | None:
    """Construct one step from runtime context, or ``None`` to skip it.

    resize_images / pad_dimensions branch here because their parameters are
    runtime-resolved (image_size, action dims) rather than carried in a static
    config.  Self-contained step types (Phase 3's ``delta_actions`` with its
    mask, etc.) are built generically via the registry.
    """
    if name == "normalize":
        if ctx.norm_stats is None:
            return None
        return TransformRegistry.create({"type": "normalize"}, stats=ctx.norm_stats)

    if name == "resize_images":
        if not ctx.image_size or ctx.image_size[0] <= 0 or ctx.image_size[1] <= 0:
            return None
        h, w = ctx.image_size
        return TransformRegistry.create(
            {"type": "resize_images", "height": int(h), "width": int(w)}
        )

    if name == "pad_dimensions":
        if ctx.model_action_dim <= 0 or ctx.model_action_dim <= ctx.dataset_action_dim:
            return None
        return TransformRegistry.create(
            {"type": "pad_dimensions", "target_dim": ctx.model_action_dim}
        )

    # Self-contained registered type — build generically, passing shared stats.
    try:
        return TransformRegistry.create({"type": name}, stats=ctx.norm_stats)
    except (KeyError, TypeError):
        return None


def build_transforms(
    transform_types: Iterable[str] | None,
    ctx: BuildContext,
) -> tuple[TransformPipeline, TransformPipeline]:
    """Build ``(preprocessor, postprocessor)`` from a model's declared transforms.

    The postprocessor is **derived from the built preprocessor**: walk its steps
    in reverse, and for each step that has a registered ``paired_reverse``, build
    that reverse step. The reverse step shares ``ctx.norm_stats`` and inherits
    the forward step's config (filtered to the kwargs the reverse ``__init__``
    accepts — e.g. a future delta mask is forwarded, while normalize's
    ``use_imagenet_stats`` is dropped for ``UnnormalizeAction``).

    Forward-only steps (ResizeImages, PadDimensions) have no ``paired_reverse``
    and contribute nothing to the postprocessor — they have no inverse at
    inference (you never un-resize or un-pad an action).

    Examples:
      ACT       pre=[Normalize, ResizeImages?, PadDimensions?]  post=[UnnormalizeAction]
      Future δ  pre=[DeltaActions, Normalize]                   post=[UnnormalizeAction, AbsoluteActions]
    """
    pre = build_preprocessor(transform_types, ctx)
    post_steps: list[TransformStep] = []
    for step in reversed(pre.steps):
        name = TransformRegistry.name_of(step)
        if name is None:
            continue
        rev_name = TransformRegistry.get_reverse(name)
        if rev_name is None:
            continue
        rev_cls = TransformRegistry.get(rev_name)
        fwd_cfg = {k: v for k, v in step.state_dict().items() if k != "type"}
        accepted = _accepted_kwargs(rev_cls)
        if accepted is not None:
            fwd_cfg = {k: v for k, v in fwd_cfg.items() if k in accepted}
        post_steps.append(
            TransformRegistry.create({"type": rev_name, **fwd_cfg}, stats=ctx.norm_stats)
        )
    return pre, TransformPipeline(post_steps)

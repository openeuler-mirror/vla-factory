"""TransformPipeline and YAML-driven builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .base import PlanContext, TransformStep
from .registry import TransformRegistry


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
class TransformContext:
    """Runtime context used to instantiate a model's default transform list.

    Fields are sourced from model metadata + dataset schema.  Each transform
    type draws only what it needs: Normalize -> ``norm_stats``,
    ResizeImages -> its own ``height`` / ``width`` config, PadDimensions ->
    ``model_action_dim`` / ``dataset_action_dim``.
    """

    norm_stats: Any | None = None
    model_action_dim: int = 0
    dataset_action_dim: int = 0
    recipe: Any | None = None
    schema: Any | None = None
    model_config: dict[str, Any] | None = None
    split: str = "train"
    # The model's ModelMetadata — the single source for model-side transform
    # facts (image range/normalize, vector normalization) per decision D3.
    metadata: Any | None = None

    def plan(self) -> PlanContext:
        """The fact-only view a step's ``compile_call`` works from.

        Resolving the recipe-shaped fallbacks here (tokenizer repo, default
        task) is what lets the steps themselves stay free of recipe knowledge —
        and lets the composition resolver, which has no recipe at all, build the
        same context out of its own inputs.
        """
        from vla_factory.recipe.recipe import get_default_task

        recipe = self.recipe
        stats = self.norm_stats
        return PlanContext(
            metadata=self.metadata,
            model_action_dim=self.model_action_dim,
            dataset_action_dim=self.dataset_action_dim,
            has_norm_stats=stats is not None,
            has_action_stats=getattr(stats, "action", None) is not None,
            default_task=get_default_task(recipe) if recipe is not None else None,
            tokenizer_repo=getattr(recipe, "model_path", None) if recipe else None,
        )

def build_preprocessor(
    transform_types: Iterable[dict] | None,
    ctx: TransformContext,
) -> TransformPipeline:
    """Build a forward preprocessor pipeline from YAML transform configs."""
    steps: list[TransformStep] = []
    for item in transform_types:
        step = TransformRegistry.create_from_config(item, ctx)
        if step is not None:
            steps.append(step)
    return TransformPipeline(steps)


def build_transforms(
    transform_types: Iterable[dict] | None,
    ctx: TransformContext,
) -> tuple[TransformPipeline, TransformPipeline]:
    """Build ``(preprocessor, postprocessor)`` from YAML transform configs."""
    pre = build_preprocessor(transform_types, ctx)
    post_steps: list[TransformStep] = []
    for step in reversed(pre.steps):
        inverse = step.inverse_for_output(ctx)
        if inverse is not None:
            post_steps.append(inverse)
    return pre, TransformPipeline(post_steps)

"""TransformPipeline: an ordered list of built steps, and how a plan becomes one."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .base import TransformStep
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
    """Runtime context for instantiating a resolved plan.

    A ``TransformStepCall``'s args carry every *value* a step needs; this
    carries the one thing they cannot — the dataset statistics, which are live
    numpy arrays. Everything else a step used to pull from here (the model
    declaration, the schema, the recipe) was used to *derive* facts, and that
    now happens once, in the composition resolver.
    """

    norm_stats: Any | None = None


def build_pipeline(plan, ctx: TransformContext) -> TransformPipeline:
    """Instantiate a resolved ``TransformPipelinePlan`` into a runnable pipeline.

    Every argument was already resolved by the composition resolver, so this is
    pure construction: no fact is derived, no step is skipped, nothing is
    re-decided. The context supplies only what a serialized call cannot carry —
    the dataset statistics (``NormalizeVector`` and its inverses take them from
    there).

    An unresolved plan is refused rather than built as an empty pipeline: a
    pipeline that does nothing is indistinguishable from one that worked, and on
    the reverse path it means sending normalized actions to a robot.
    """
    if not plan.resolved:
        raise ValueError(
            "Cannot build a pipeline from an unresolved TransformPipelinePlan: "
            "the resolver did not plan this path. An empty pipeline would run "
            "silently instead of failing."
        )
    return TransformPipeline([
        TransformRegistry.get(call.type).from_call(dict(call.args), ctx)
        for call in plan.calls
    ])

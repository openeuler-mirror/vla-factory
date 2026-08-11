"""Plan Pipeline: a declared step list → resolved calls.

The planner knows *no* step names. Each step owns its own planning rule
(``TransformStep.compile_call`` / ``inverse_call``), registered under its name
in the ``TransformRegistry``, so adding a transform never means editing this
module — and the build path runs the very same rules, which is what keeps a
plan and the pipeline it describes from drifting apart.

``robot_to_model`` is not planned here: the T1 steps it needs (joint reorder,
gripper flip) have no implementation to reference yet.
"""

from __future__ import annotations

from typing import Any

from vla_factory.assembly.transforms import TransformRegistry
from vla_factory.assembly.transforms.base import PlanContext
from vla_factory.data.manifest import DataSchema, NormStats
from vla_factory.model.interfaces.model import ModelMetadata

from .errors import PIPELINE_WIDTH_MISMATCH, make_error
from .types import TransformPipelinePlan, TransformStepCall


def transform_declaration(
    metadata: ModelMetadata, model_config: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """The step list to compile: the recipe's resolved ``model.config`` when
    given (``resolve_recipe`` has already merged the model's declared defaults
    under it), else the declaration alone."""
    src = model_config if model_config is not None else (metadata.params or {})
    return list(((src or {}).get("transforms") or {}).get("inputs") or [])


def pad_target(schema: DataSchema, metadata: ModelMetadata) -> int:
    """The width vector-padding steps pad to.

    Mirrors what training feeds them today (``train.py``: ``metadata.action_dim
    or <dataset action dim>``). For a model that declares no internal width
    (ACT) this equals the dataset width, i.e. "no padding".
    """
    return int(metadata.action_dim or 0) or int(schema.action_dim)


def plan_context(
    schema: DataSchema, norm_stats: NormStats, metadata: ModelMetadata,
    overrides: dict[str, Any],
) -> PlanContext:
    """The resolver's half of the context every ``compile_call`` reads.

    ``TransformContext.plan()`` builds the same type from the runtime side; the
    two agreeing is what makes the equivalence test meaningful. ``tokenizer_repo``
    stays ``None`` here: its fallback is the recipe's ``model.path``, which the
    resolver does not receive — every shipped model that tokenizes declares the
    repo outright, so no combination depends on the fallback today.
    """
    return PlanContext(
        metadata=metadata,
        model_action_dim=pad_target(schema, metadata),
        dataset_action_dim=int(schema.action_dim),
        has_norm_stats=True,        # a required resolver input
        has_action_stats=norm_stats.action is not None,
        default_task=overrides.get("default_task"),
        tokenizer_repo=None,
    )


def plan_data_to_model(
    declaration: list[dict[str, Any]], ctx: PlanContext,
) -> TransformPipelinePlan:
    """Compile the declared step list into resolved calls, in declared order.

    The *order* comes from the model declaration and is never re-derived: it is
    upstream semantics (pi05 must tokenize before padding so the state is
    digitized at its native width; pi0's letterbox must follow the layout flip),
    and the framework holds no model architecture knowledge to reinvent it.
    """
    if not declaration:
        return TransformPipelinePlan()
    calls: list[TransformStepCall] = []
    for declared in declaration:
        step_type = declared.get("type")
        if not step_type:
            continue
        # An unregistered type is a typo in the declaration, not an extension
        # point: TransformRegistry.get raises with the available names.
        args = TransformRegistry.get(step_type).compile_call(declared, ctx)
        if args is None:
            continue                # this step is a no-op for these facts
        calls.append(TransformStepCall(type=step_type, args=args))
    return TransformPipelinePlan(calls=tuple(calls), resolved=True)


def plan_model_to_robot(
    data_to_model: TransformPipelinePlan, ctx: PlanContext,
) -> TransformPipelinePlan:
    """Invert the forward plan, asking each step for its own pairing.

    Not "reverse the list" (architecture §4.2.4): a call whose step declares no
    inverse disappears instead of being mirrored, and a lossy step is required
    to declare ``None`` rather than a lookalike.

    With no robot declared the target space is the dataset's action space —
    which is exactly what ``InferenceEngine``'s postprocessor produces today.
    """
    if not data_to_model.resolved:
        return TransformPipelinePlan()
    calls: list[TransformStepCall] = []
    for call in reversed(data_to_model.calls):
        inverse = TransformRegistry.get(call.type).inverse_call(call.args, ctx)
        if inverse is None:
            continue
        name, args = inverse
        calls.append(TransformStepCall(type=name, args=args))
    return TransformPipelinePlan(calls=tuple(calls), resolved=True)


def vector_widths(
    plan: TransformPipelinePlan, schema: DataSchema, metadata: ModelMetadata,
) -> tuple[int, int]:
    """``(state_width, action_width)`` of the model-side vectors.

    Folded through the planned calls, each step reporting its own effect on the
    widths going in, so the model IO spec reports exactly what the pipeline
    emits — the mappings and the plan cannot disagree about a width because they
    read the same number.

    The model declaration does not *supply* a width here, it *constrains* one:
    a model that declares an action width the pipeline never reaches is a
    self-inconsistent declaration and fails rather than being silently papered
    over with one of the two numbers.

    With no plan at all (no step list declared) there is no pipeline to read a
    width off, and nothing for the declaration to contradict — the declared
    width is then simply reported.
    """
    if not plan.resolved:
        return int(schema.state_dim), pad_target(schema, metadata)

    widths = {"state": int(schema.state_dim), "actions": int(schema.action_dim)}
    for call in plan.calls:
        widths = TransformRegistry.get(call.type).output_widths(call.args, widths)

    action_width = widths["actions"]
    if metadata.action_dim and action_width != metadata.action_dim:
        raise make_error(
            PIPELINE_WIDTH_MISMATCH, "model.action_dim",
            field="actions", model_dim=int(metadata.action_dim),
            model_dim_source="metadata", pipeline_dim=action_width,
        )
    return widths["state"], action_width

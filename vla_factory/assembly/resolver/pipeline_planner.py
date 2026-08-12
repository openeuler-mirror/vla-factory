"""Plan Pipeline: a target ModelIOSpec + declared policies → resolved calls.

Each step owns its call argument and inverse rules (``compile_call`` /
``inverse_call``). The planner additionally owns the two standard interface
reconciliations: it ensures required zero-padding is present, and requires an
explicit resize policy when model and data image sizes differ. This is the
deliberate boundary: ModelIOSpec defines *what shape* must be reached; step
policy defines *how* to reach it.

``robot_to_model`` is not planned here: the T1 steps it needs (joint reorder,
gripper flip) have no implementation to reference yet.
"""

from __future__ import annotations

from typing import Any

from vla_factory.assembly.transforms import TransformRegistry
from vla_factory.assembly.transforms.base import PlanContext
from vla_factory.data.manifest import DataSchema, NormStats
from vla_factory.model.interfaces.model import ModelMetadata

from .types import ModelIOSpec, TransformPipelinePlan, TransformStepCall


def transform_declaration(
    metadata: ModelMetadata, model_config: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """The step list to compile: the recipe's resolved ``model.config`` when
    given (``resolve_recipe`` has already merged the model's declared defaults
    under it), else the declaration alone."""
    src = model_config if model_config is not None else (metadata.params or {})
    return list(((src or {}).get("transforms") or {}).get("inputs") or [])


def plan_context(
    schema: DataSchema, norm_stats: NormStats, metadata: ModelMetadata,
    io_spec: ModelIOSpec, overrides: dict[str, Any],
    model_path: str | None = None,
) -> PlanContext:
    """The resolver's half of the context every ``compile_call`` reads.

    ``model_path`` is the recipe's checkpoint selection, passed in purely as the
    ``tokenizer_repo`` fallback (``task_tokenize`` accepts a base checkpoint that
    ships its own tokenizer instead of a declared repo). The resolver never opens
    it — reading a checkpoint is the caller's job. Without it a plan for such a
    model would serialize a call with no tokenizer address at all, and the
    execution side has no fallback left to fill in.
    """
    dataset_camera_shapes = {
        camera.key: (int(camera.resolution[0]), int(camera.resolution[1]))
        for camera in schema.cameras_entries
        if camera.resolution and len(camera.resolution) == 2
    }
    return PlanContext(
        metadata=metadata,
        target_state_dim=int(io_spec.state_dim),
        target_action_dim=int(io_spec.action_dim),
        source_state_dim=int(schema.state_dim),
        source_action_dim=int(schema.action_dim),
        source_camera_shapes=dataset_camera_shapes,
        target_camera_shapes=dict(io_spec.camera_shapes),
        has_norm_stats=True,        # a required resolver input
        has_action_stats=norm_stats.action is not None,
        default_task=overrides.get("default_task"),
        tokenizer_repo=model_path,
    )


def plan_data_to_model(
    declaration: list[dict[str, Any]], ctx: PlanContext,
) -> TransformPipelinePlan:
    """Compile the declared step list into resolved calls, in declared order.

    Declared steps preserve their declared order: it is upstream semantics
    (pi05 must tokenize before padding so state is digitized at native width;
    pi0's letterbox follows the layout flip). Standard reconciliation may add
    missing zero-padding at the end, but never reorders declared calls.
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

    # A target image size alone cannot choose stretch vs letterbox, so unlike
    # zero-padding this reconciliation cannot be invented safely. Require the
    # model's step template to provide the resize policy whenever source and
    # target sizes differ. Its height/width still come only from ModelIOSpec.
    resize_required = TransformRegistry.get("resize_images").compile_call({}, ctx)
    if resize_required is not None and not any(
        call.type == "resize_images" for call in calls
    ):
        raise ValueError(
            "The resolved ModelIOSpec requires image resizing, but the model's "
            "transform declaration contains no resize_images policy. Add a "
            "resize_images step with mode/interpolation only; dimensions come "
            "from the model interface."
        )

    # Width reconciliation is required by the already-resolved ModelIOSpec; it
    # is not optional merely because an old transform template omitted a pad
    # placeholder. Keep an explicitly declared call in its declared position
    # (pi05 tokenizes native-width state before padding), and append only the
    # uncovered fields.
    required_pad_fields = {
        field for field, source, target in (
            ("state", ctx.source_state_dim, ctx.target_state_dim),
            ("actions", ctx.source_action_dim, ctx.target_action_dim),
        )
        if int(target or 0) > int(source or 0)
    }
    covered_pad_fields = {
        field
        for call in calls
        if call.type == "pad_dimensions"
        for field in (call.args.get("fields") or ())
    }
    missing_pad_fields = [
        field for field in ("state", "actions")
        if field in required_pad_fields - covered_pad_fields
    ]
    if missing_pad_fields:
        args = TransformRegistry.get("pad_dimensions").compile_call(
            {"fields": missing_pad_fields}, ctx,
        )
        if args is not None:  # pragma: no branch - required fields cannot no-op
            calls.append(TransformStepCall(type="pad_dimensions", args=args))
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

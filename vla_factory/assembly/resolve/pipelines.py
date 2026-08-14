"""Derive executable transform calls from the resolved model interface.

Recipes do not declare transform steps. ``ModelMetadata`` states the immutable
model requirements; this planner selects and orders the registered operations
needed to reconcile raw DataSchema samples with those requirements. Each step
still owns argument compilation and inverse pairing (``compile_call`` /
``inverse_call``). ``robot_to_model`` reuses ``data_to_model`` because platform
adapters already emit the checkpoint DataSchema interface.
"""

from __future__ import annotations

from vla_factory.assembly.transform import TransformRegistry
from vla_factory.assembly.transform.base import PlanContext
from vla_factory.data.data_schema import DataSchema, NormStats
from vla_factory.model.model_interface import ModelMetadata

from ..resolve_assembly import ModelIOSpec
from ..transform.plan import TransformPipelinePlan, TransformStepCall


def plan_context(
    schema: DataSchema, norm_stats: NormStats, metadata: ModelMetadata,
    io_spec: ModelIOSpec, default_task: str | None,
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
        default_task=default_task,
        tokenizer_repo=model_path,
    )


def _append_call(
    calls: list[TransformStepCall], step_type: str, cfg: dict, ctx: PlanContext,
) -> None:
    """Compile one resolver-selected operation and append it unless it is a no-op."""
    args = TransformRegistry.get(step_type).compile_call(cfg, ctx)
    if args is not None:
        calls.append(TransformStepCall(type=step_type, args=args))


def _tokenizer_config(ctx: PlanContext) -> dict | None:
    metadata = ctx.metadata
    max_length = metadata.tokenizer_max_length
    if not metadata.requires_prompt:
        if max_length is not None or metadata.prompt_includes_state:
            raise ValueError(
                f"Model {metadata.name!r} declares tokenizer facts but "
                "requires_prompt=False."
            )
        return None
    if metadata.prompt_includes_state:
        if int(ctx.source_state_dim or 0) <= 0:
            raise ValueError(
                f"Model {metadata.name!r} embeds state in its prompt, but the "
                "dataset declares no state vector."
            )
        if metadata.vector_normalization is None:
            raise ValueError(
                f"Model {metadata.name!r} embeds state in its prompt but declares "
                "no vector_normalization; state must be normalized before "
                "tokenization."
            )
    if max_length is None or int(max_length) <= 0:
        raise ValueError(
            f"Model {metadata.name!r} requires a prompt but declares no positive "
            "ModelMetadata.tokenizer_max_length."
        )
    cfg = {
        "max_length": int(max_length),
        "discrete_state": bool(metadata.prompt_includes_state),
    }
    if metadata.tokenizer_repo is not None:
        cfg["tokenizer_repo"] = metadata.tokenizer_repo
    if metadata.tokenizer_repo is None and ctx.tokenizer_repo is None:
        raise ValueError(
            f"Model {metadata.name!r} requires tokenization but declares no "
            "tokenizer_repo and no model checkpoint was provided as a fallback."
        )
    return cfg


def _append_padding_calls(
    calls: list[TransformStepCall], ctx: PlanContext,
) -> None:
    """Group width reconciliation by target so unequal targets stay independent."""
    grouped: dict[int, list[str]] = {}
    for field, source, target in (
        ("state", ctx.source_state_dim, ctx.target_state_dim),
        ("actions", ctx.source_action_dim, ctx.target_action_dim),
    ):
        if int(target or 0) > int(source or 0):
            grouped.setdefault(int(target), []).append(field)
    for fields in grouped.values():
        _append_call(calls, "pad_dimensions", {"fields": fields}, ctx)


def plan_data_to_model(ctx: PlanContext) -> TransformPipelinePlan:
    """Derive the complete input plan from DataSchema and ModelMetadata facts.

    Ordering follows dependencies, not model-specific lists: images are brought
    to their model contract first; vectors are normalized before anything reads
    them; a prompt that embeds state is tokenized before padding; padding is the
    final shape reconciliation. A normal prompt is independent and follows it.
    """
    metadata = ctx.metadata
    calls: list[TransformStepCall] = []

    if metadata.image_input_range is not None:
        _append_call(calls, "image_to_float", {}, ctx)
    if metadata.image_layout is not None:
        _append_call(calls, "image_layout", {"to": metadata.image_layout}, ctx)

    resize_args = TransformRegistry.get("resize_images").compile_call(
        {"mode": metadata.image_resize_mode or "stretch"}, ctx,
    )
    if resize_args is not None:
        if metadata.image_resize_mode is None:
            raise ValueError(
                f"Model {metadata.name!r} needs image resizing but declares no "
                "ModelMetadata.image_resize_mode."
            )
        calls.append(TransformStepCall(type="resize_images", args=resize_args))

    if metadata.image_normalize_mode is not None:
        if metadata.image_input_range is None:
            raise ValueError(
                f"Model {metadata.name!r} declares image_normalize_mode without "
                "an image_input_range."
            )
        _append_call(calls, "image_normalize", {}, ctx)

    vector_fields = [
        field for field, width in (
            ("state", ctx.source_state_dim),
            ("actions", ctx.source_action_dim),
        )
        if int(width or 0) > 0
    ]
    if metadata.vector_normalization is not None and vector_fields:
        _append_call(
            calls, "normalize_vector", {"fields": vector_fields}, ctx,
        )

    tokenizer_cfg = _tokenizer_config(ctx)
    if tokenizer_cfg is not None and metadata.prompt_includes_state:
        _append_call(calls, "task_tokenize", tokenizer_cfg, ctx)

    _append_padding_calls(calls, ctx)

    if tokenizer_cfg is not None and not metadata.prompt_includes_state:
        _append_call(calls, "task_tokenize", tokenizer_cfg, ctx)

    return TransformPipelinePlan(calls=tuple(calls))


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
    calls: list[TransformStepCall] = []
    for call in reversed(data_to_model.calls):
        inverse = TransformRegistry.get(call.type).inverse_call(call.args, ctx)
        if inverse is None:
            continue
        name, args = inverse
        calls.append(TransformStepCall(type=name, args=args))
    return TransformPipelinePlan(calls=tuple(calls))

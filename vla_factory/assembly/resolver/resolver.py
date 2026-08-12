"""``resolve_assembly`` — the deterministic composition-resolution entry point.

This module is the stage orchestration only; each stage lives in its own module:

==================  ==========================
Stage               Module
==================  ==========================
Load                here
Validate            here
Check Pairs         :mod:`.compatibility`
Resolve Mappings    :mod:`.mappings`
Build IO Spec       :mod:`.io_spec`
Plan Pipeline       :mod:`.pipeline_planner`
Emit                here
==================  ==========================

Mappings contain only real semantic correspondences, never synthetic padding
slots, so they can all be resolved first. Model/data facts then define the
runtime interface, and transform steps compile the reconciliation needed to
reach it; the plan is never a second source of tensor shapes.

The function is pure: it creates no model, no DataLoader, no output directory
and uses no GPU, and its result is fully serializable.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from vla_factory.data.manifest import DataSchema, NormStats, resolve_vector_keys
from vla_factory.model.interfaces.model import ModelMetadata
from vla_factory.robot.profile import RobotProfile

from . import io_spec as io_spec_resolver, mappings, pipeline_planner
from .compatibility import check_pairs
from .errors import (
    INVALID_DESCRIPTION,
    MISSING_INPUT,
    UNSUPPORTED_OVERRIDE,
    make_error,
)
from .types import ResolvedAssembly


# ── helpers ───────────────────────────────────────────────────────


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclass/tuple structures into JSON-friendly
    plain ``dict`` / ``list`` / scalar values."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


# Controlled overrides the resolver actually reads. A recipe may set any
# ``AssemblyConfig`` field, but one nothing consumes is rejected rather than
# quietly dropped: a user who writes ``gripper_flip: true`` would otherwise
# believe a gripper decision had been made, when no stage looks at it. Same
# guard the model-config surface applies to a declared-but-unread key
# (``utils/tracked_config.py``). Add a key here in the commit that starts
# consuming it — ``test_every_assembly_override_is_accounted_for`` fails
# otherwise.
CONSUMED_OVERRIDES = frozenset({"camera_mapping", "default_task"})


# ── stages implemented here ───────────────────────────────────────


def _load(
    schema: DataSchema | None,
    norm_stats: NormStats | None,
    metadata: ModelMetadata | None,
) -> tuple[DataSchema, NormStats, ModelMetadata]:
    """Require the three core descriptions."""
    if schema is None:
        raise make_error(MISSING_INPUT, "schema", field_name="schema", detail="DataSchema is required")
    if not isinstance(schema, DataSchema):
        raise make_error(
            INVALID_DESCRIPTION, "schema",
            field_name="schema", value=type(schema).__name__,
            detail="schema must be a DataSchema instance",
        )
    if norm_stats is None:
        raise make_error(
            MISSING_INPUT, "norm_stats", field_name="norm_stats", detail="NormStats is required"
        )
    if not isinstance(norm_stats, NormStats):
        raise make_error(
            INVALID_DESCRIPTION, "norm_stats",
            field_name="norm_stats", value=type(norm_stats).__name__,
            detail="norm_stats must be a NormStats instance",
        )
    if metadata is None:
        raise make_error(
            MISSING_INPUT, "model", field_name="metadata", detail="ModelMetadata is required"
        )
    if not isinstance(metadata, ModelMetadata):
        raise make_error(
            INVALID_DESCRIPTION, "model",
            field_name="metadata", value=type(metadata).__name__,
            detail="metadata must be a ModelMetadata instance",
        )
    return schema, norm_stats, metadata


def _validate_overrides(overrides: dict[str, Any]) -> None:
    """Reject a controlled override no stage consumes."""
    unsupported = sorted(set(overrides) - CONSUMED_OVERRIDES)
    if unsupported:
        raise make_error(
            UNSUPPORTED_OVERRIDE, "assembly",
            keys=unsupported, supported=sorted(CONSUMED_OVERRIDES),
        )


def _validate(schema: DataSchema, robot_profile: RobotProfile | None) -> None:
    """Validate each description's internal structure."""
    # Every non-empty vector must carry exactly one name per dimension: the
    # dimension→motor-key correspondence is a description fact that the deploy
    # side reads back, and it is never invented by sorting. Validating it here
    # rather than in ``train()`` means an incomplete dataset description is
    # reported before anything downstream (an output directory, a model) exists.
    try:
        resolve_vector_keys(schema)
    except ValueError as e:
        raise make_error(
            INVALID_DESCRIPTION, "schema",
            field_name="state_dims/action_dims", value=None, detail=str(e),
        ) from e

    if robot_profile is not None:
        if not isinstance(robot_profile, RobotProfile):
            raise make_error(
                INVALID_DESCRIPTION, "robot",
                field_name="robot_profile", value=type(robot_profile).__name__,
                detail="robot_profile must be a RobotProfile instance",
            )
        try:
            robot_profile.validate()
        except ValueError as e:
            raise make_error(
                INVALID_DESCRIPTION, f"robot({robot_profile.name})",
                field_name="robot_profile", value=None, detail=str(e),
            ) from e


# ── public entry point ────────────────────────────────────────────


def resolve_assembly(
    schema: DataSchema | None,
    norm_stats: NormStats | None,
    metadata: ModelMetadata,
    *,
    robot_profile: RobotProfile | None = None,
    overrides: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
    model_path: str | None = None,
) -> ResolvedAssembly:
    """Resolve a ``data × model × robot`` combination into a ``ResolvedAssembly``.

    Deterministic pure logic (architecture §4.2.2): no model construction, no
    DataLoader, no training, no deploy platform, no GPU, no output directory.

    Parameters
    ----------
    schema, norm_stats, metadata
        Core descriptions (required). ``schema`` / ``norm_stats`` come from the
        data reader; ``metadata`` comes from the model registry.
    robot_profile
        Optional robot-body description.
    overrides
        Controlled overrides from the recipe's ``assembly`` block. Only the
        keys in :data:`CONSUMED_OVERRIDES` are supported — any other key is
        rejected rather than ignored. Stored on the result (``overrides_ref``)
        so every adjusted field records its source (architecture §3.4).
    model_config
        The recipe's resolved ``model.config`` — this model's tunables after
        ``resolve_recipe()`` merged the declared defaults under the per-run
        overrides. The IO stage reads explicit interface tunables such as ACT's
        ``action_horizon`` / ``input_image_size``; the planner reads the
        ``transforms.inputs`` policy list. Omit it and the model's own
        declaration is used.
    model_path
        The recipe's checkpoint selection, used only as the ``task_tokenize``
        tokenizer fallback when a model declares no ``tokenizer_repo`` (see
        :func:`pipeline_planner.plan_context`). No file is read from it here.

    Returns
    -------
    ResolvedAssembly
        The serialized combination.

    Raises
    ------
    ResolutionError
        Every way this *combination* can fail — a missing or invalid
        description, an incompatible pair, an
        override that names something absent. Structured (``code`` / ``path`` /
        ``params``) and safe for tools to key on.
    ValueError, KeyError
        A registry entry declaring something impossible: a transform step whose
        model fact it never declared, an invalid model image size or missing
        resize policy, a ``vector_normalization`` no NormalizeVector method
        implements, or a step type that is not registered. These are programming
        errors in a model or transform entry, not properties of this
        data × model × robot combination, so they stay plain exceptions with no
        stable code — the same failures the pipeline build raises.
    """
    overrides_ref = dict(overrides or {})

    # 1. Load
    schema, norm_stats, metadata = _load(schema, norm_stats, metadata)

    # 2. Validate (only the optional descriptions need extra checks here; the
    #    core descriptions were type-checked in Load).
    _validate_overrides(overrides_ref)
    _validate(schema, robot_profile)

    # 3. Check Pairs
    check_pairs(schema, norm_stats, metadata, robot_profile, overrides_ref)

    # 4. Resolve Mappings. They describe only real correspondences, so none
    #    depends on a model target width or a planned padding call.
    camera_mapping = mappings.resolve_camera_mapping(schema, metadata, overrides_ref)
    state_mapping = mappings.resolve_state_mapping(schema)
    action_mapping = mappings.resolve_action_mapping(schema)
    language_mapping = mappings.resolve_language_mapping(schema, metadata, overrides_ref)
    joint_mapping = mappings.resolve_joint_mapping(schema, robot_profile)

    # 5. Build IO Spec directly from model/data facts.
    io_spec = io_spec_resolver.resolve_model_io_spec(
        schema, metadata, model_config, camera_mapping,
    )

    # 6. Plan Pipeline against that target interface.
    declaration = pipeline_planner.transform_declaration(metadata, model_config)
    plan_ctx = pipeline_planner.plan_context(
        schema, norm_stats, metadata, io_spec, overrides_ref, model_path,
    )
    data_to_model = pipeline_planner.plan_data_to_model(declaration, plan_ctx)
    model_to_robot = pipeline_planner.plan_model_to_robot(data_to_model, plan_ctx)

    # 7. Emit
    return ResolvedAssembly(
        schema_ref=schema.to_dict(),
        norm_stats_ref=_to_jsonable(norm_stats),
        metadata_ref=_to_jsonable(metadata),
        robot_ref=robot_profile.to_dict() if robot_profile is not None else None,
        overrides_ref=overrides_ref,
        model_io_spec=io_spec,
        camera_mapping=camera_mapping,
        state_mapping=state_mapping,
        action_mapping=action_mapping,
        language_mapping=language_mapping,
        joint_mapping=joint_mapping,
        data_to_model=data_to_model,
        model_to_robot=model_to_robot,
    )

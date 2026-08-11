"""``resolve_assembly`` — the deterministic composition-resolution entry point.

This module is the stage orchestration only; each stage lives in its own module:

==================  ==========================
Stage               Module
==================  ==========================
Load                here
Validate            here
Check Pairs         :mod:`.compatibility`
Plan Pipeline       :mod:`.pipeline_planner`
Build IO Spec       :mod:`.pipeline_planner`
Resolve Mapping     :mod:`.mappings`
Emit                here
==================  ==========================

Build IO Spec runs *after* Plan Pipeline: the widths in the model IO spec are
folded through the planned calls, so the spec can only ever report what the
pipeline really produces.

The function is pure: it creates no model, no DataLoader, no output directory
and uses no GPU, and its result is fully serializable.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from vla_factory.data.manifest import DataSchema, NormStats
from vla_factory.model.interfaces.model import ModelMetadata
from vla_factory.robot.profile import RobotProfile

from . import mappings, pipeline_planner
from .compatibility import check_pairs
from .errors import (
    INVALID_DESCRIPTION,
    MISSING_INPUT,
    UNSUPPORTED_OVERRIDE,
    make_error,
)
from .types import ModelIOSpec, ResolvedAssembly


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


def _validate(robot_profile: RobotProfile | None) -> None:
    """Validate each description's internal structure."""
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
        overrides. Only the ``transforms.inputs`` step list is read, to plan
        the pipelines. Omit it and the model's own declaration is used.

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
        model fact it never declared, a ``resize_images`` without usable
        dimensions, a ``vector_normalization`` no NormalizeVector method
        implements, a step type that is not registered. These are programming
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
    _validate(robot_profile)

    # 3. Check Pairs
    check_pairs(schema, norm_stats, metadata, robot_profile, overrides_ref)

    # 4. Plan Pipeline
    declaration = pipeline_planner.transform_declaration(metadata, model_config)
    plan_ctx = pipeline_planner.plan_context(schema, norm_stats, metadata, overrides_ref)
    data_to_model = pipeline_planner.plan_data_to_model(declaration, plan_ctx)
    model_to_robot = pipeline_planner.plan_model_to_robot(data_to_model, plan_ctx)

    # 5. Build IO Spec (after Plan Pipeline — see the module docstring)
    state_dim, action_dim = pipeline_planner.vector_widths(data_to_model, schema, metadata)
    io_spec = ModelIOSpec(
        action_dim=action_dim,
        action_horizon=metadata.action_horizon,
        state_dim=state_dim,
        cameras=tuple(schema.cameras),
        requires_language=bool(metadata.requires_prompt),
    )

    # 6. Resolve Mapping
    camera_mapping = mappings.resolve_camera_mapping(schema, metadata, overrides_ref)
    state_mapping = mappings.resolve_state_mapping(schema, state_dim)
    action_mapping = mappings.resolve_action_mapping(schema, action_dim)
    language_mapping = mappings.resolve_language_mapping(schema, metadata, overrides_ref)
    joint_mapping = mappings.resolve_joint_mapping(schema, robot_profile)

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

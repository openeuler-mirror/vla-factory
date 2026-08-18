"""``resolve_assembly`` — the deterministic composition-resolution entry point.

This module is the stage orchestration only; each stage lives in its own module:

==================  ==========================
Responsibility      Module
==================  ==========================
Validate inputs     here
Check Pairs         :mod:`.checks`
Resolve Mappings    :mod:`.mappings`
Build IO Spec       :mod:`.model_io`
Plan Pipeline       :mod:`.pipelines`
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

from vla_factory.data.data_schema import DataSchema, NormStats, resolve_vector_keys
from vla_factory.model.model_interface import ModelMetadata
from vla_factory.user_interface import AssemblyOverrides
from vla_factory.robot import RobotProfile

from . import mappings, model_io, pipelines
from .checks import check_pairs
from .errors import (
    INVALID_DESCRIPTION,
    MISSING_INPUT,
    make_error,
)
from ..resolve_assembly import ResolvedAssembly


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


# ── stages implemented here ───────────────────────────────────────


def _require_inputs(
    schema: DataSchema | None,
    norm_stats: NormStats | None,
    metadata: ModelMetadata | None,
) -> tuple[DataSchema, NormStats, ModelMetadata]:
    """Require the three core descriptions."""
    if schema is None:
        raise make_error(MISSING_INPUT, "schema", field="schema", detail="DataSchema is required")
    if not isinstance(schema, DataSchema):
        raise make_error(
            INVALID_DESCRIPTION, "schema",
            field="schema", value=type(schema).__name__,
            detail="schema must be a DataSchema instance",
        )
    if norm_stats is None:
        raise make_error(
            MISSING_INPUT, "norm_stats", field="norm_stats", detail="NormStats is required"
        )
    if not isinstance(norm_stats, NormStats):
        raise make_error(
            INVALID_DESCRIPTION, "norm_stats",
            field="norm_stats", value=type(norm_stats).__name__,
            detail="norm_stats must be a NormStats instance",
        )
    if metadata is None:
        raise make_error(
            MISSING_INPUT, "model", field="metadata", detail="ModelMetadata is required"
        )
    if not isinstance(metadata, ModelMetadata):
        raise make_error(
            INVALID_DESCRIPTION, "model",
            field="metadata", value=type(metadata).__name__,
            detail="metadata must be a ModelMetadata instance",
        )
    return schema, norm_stats, metadata


def _validate_descriptions(
    schema: DataSchema,
    metadata: ModelMetadata,
    robot_profile: RobotProfile | None,
) -> None:
    """Validate each description independently before comparing them."""
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
            field="state_dims/action_dims", value=None, detail=str(e),
        ) from e

    if metadata.dim_policy == "flexible":
        if metadata.action_dim:
            raise ValueError(
                f"Model {metadata.name!r} declares dim_policy='flexible' and "
                f"a fixed action_dim={metadata.action_dim}. A flexible model "
                "must leave action_dim=0 so DataSchema supplies the width."
            )
    elif int(metadata.dim_policy_max or 0) <= 0:
        raise ValueError(
            f"Model {metadata.name!r} declares dim_policy="
            f"{metadata.dim_policy!r} but no positive dim_policy_max. A capping policy "
            "without a cap leaves the widths undefined. Set dim_policy_max, or "
            "declare dim_policy='flexible'."
        )

    if "transforms" in metadata.params:
        raise ValueError(
            f"Model {metadata.name!r} declares params['transforms'], but transform "
            "operations are derived by the assembly resolver from named "
            "ModelMetadata facts and are not tunable."
        )
    if metadata.image_layout not in (None, "CHW", "HWC"):
        raise ValueError(
            f"Model {metadata.name!r} declares unsupported image_layout="
            f"{metadata.image_layout!r}; expected 'CHW', 'HWC', or None."
        )
    if metadata.image_resize_mode not in (None, "stretch", "pad"):
        raise ValueError(
            f"Model {metadata.name!r} declares unsupported image_resize_mode="
            f"{metadata.image_resize_mode!r}; expected 'stretch', 'pad', or None."
        )

    if robot_profile is not None:
        if not isinstance(robot_profile, RobotProfile):
            raise make_error(
                INVALID_DESCRIPTION, "robot",
                field="robot_profile", value=type(robot_profile).__name__,
                detail="robot_profile must be a RobotProfile instance",
            )
        try:
            robot_profile.validate()
        except ValueError as e:
            raise make_error(
                INVALID_DESCRIPTION, f"robot({robot_profile.name})",
                field="robot_profile", value=None, detail=str(e),
            ) from e


# ── public entry point ────────────────────────────────────────────


def resolve_from_facts(
    schema: DataSchema | None,
    norm_stats: NormStats | None,
    metadata: ModelMetadata,
    *,
    robot_profile: RobotProfile | None = None,
    overrides: AssemblyOverrides | None = None,
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
        Typed controlled overrides from the recipe's ``overrides`` block.
        Stored on the result (``overrides_ref``) so every adjusted field records
        its source (architecture §3.4).
    model_config
        The recipe's resolved ``model.config`` — this model's tunables after
        ``merge_model_config()`` merged the declared defaults under the per-run
        overrides. The IO stage reads explicit interface tunables such as ACT's
        ``action_horizon`` / ``input_image_size``. Transform operations never
        come from this mapping; the planner derives them from model facts.
    model_path
        The recipe's checkpoint selection, used only as the ``task_tokenize``
        tokenizer fallback when a model declares no ``tokenizer_repo`` (see
        :func:`pipelines.plan_context`). No file is read from it here.

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
        A registry entry declaring something impossible: an invalid model image
        size or missing resize/tokenizer policy, or a ``vector_normalization``
        no NormalizeVector method implements. These are programming errors in a
        model entry, not properties of this
        data × model × robot combination, so they stay plain exceptions with no
        stable code — the same failures the pipeline build raises.
    """
    schema, norm_stats, metadata = _require_inputs(schema, norm_stats, metadata)
    if model_config is not None and "transforms" in model_config:
        raise ValueError(
            "model.config.transforms is not supported: transform operations are "
            "derived by the assembly resolver and cannot be overridden per run."
        )
    if overrides is None:
        overrides = AssemblyOverrides()
    elif not isinstance(overrides, AssemblyOverrides):
        raise TypeError("overrides must be an AssemblyOverrides instance")
    overrides_ref = {
        name: value for name, value in asdict(overrides).items()
        if value is not None
    }

    _validate_descriptions(schema, metadata, robot_profile)

    # Check facts that share an explicit vocabulary.
    check_pairs(schema, norm_stats, metadata, robot_profile)

    # Resolve mappings. They describe only real correspondences, so none
    # depends on a model target width or a planned padding call.
    camera_mapping = mappings.resolve_camera_mapping(
        schema, metadata, overrides.camera_mapping,
    )
    state_mapping = mappings.resolve_state_mapping(schema)
    action_mapping = mappings.resolve_action_mapping(schema)
    language_mapping = mappings.resolve_language_mapping(
        schema, metadata, overrides.default_task,
    )

    # Build the IO spec directly from model/data facts.
    io_spec = model_io.resolve_model_io_spec(
        schema, metadata, model_config, camera_mapping,
    )

    # Plan pipelines against that target interface.
    plan_ctx = pipelines.plan_context(
        schema, norm_stats, metadata, io_spec, overrides.default_task, model_path,
    )
    data_to_model = pipelines.plan_data_to_model(plan_ctx)
    model_to_robot = pipelines.plan_model_to_robot(data_to_model, plan_ctx)
    # Platform adapters translate native payloads into the same DataSchema
    # interface the dataset reader produced. The semantic robot input path
    # therefore executes the exact same calls; no second planning pass exists.
    robot_to_model = data_to_model

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
        data_to_model=data_to_model,
        robot_to_model=robot_to_model,
        model_to_robot=model_to_robot,
    )

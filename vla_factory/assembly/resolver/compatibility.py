"""Check Pairs: the pairwise/triple compatibility checks.

Six matrix rows (architecture §4.2.2): state dim, action dim, camera slots,
control mode, norm stats, joint order ×2. The other five (language, gripper,
rotation, frequency, safety) are deliberately out of scope —
``docs/plans/phase2-resolution-diagnostics.cn.md`` records why.

Each check raises on the first problem it finds rather than collecting them:
the user re-runs ``resolve`` after fixing one, exactly as for the earlier
stages, and a second error-reporting convention would buy nothing here.
"""

from __future__ import annotations

from typing import Any

from vla_factory.data.manifest import ActionDim, DataSchema, NormStats, StateDim
from vla_factory.model.interfaces.model import ModelMetadata, VisionSlot
from vla_factory.robot.profile import RobotProfile
from vla_factory.utils.vocabulary import CONTROL_MODES

from .errors import (
    ACTION_DIM_INCOMPATIBLE,
    CAMERA_SLOT_AMBIGUOUS,
    CAMERA_SLOT_UNRESOLVED,
    CONTROL_MODE_INCOMPATIBLE,
    JOINT_ORDER_AMBIGUOUS,
    JOINT_ORDER_MISMATCH,
    NORM_STATS_INSUFFICIENT,
    STATE_DIM_INCOMPATIBLE,
    make_error,
)
from .matching import (
    camera_candidates,
    data_camera_candidates,
    embed_joints,
    robot_camera_candidates,
)


def _model_dim_limit(dim_policy: str, dim_policy_max: int | None) -> tuple[int, str] | None:
    """Return ``(limit, source)`` if ``dim_policy`` caps the dimension.

    ``fixed`` and ``padded_to_max`` both cap; ``flexible`` (or a missing
    ``dim_policy_max``) means "no declared limit" — ``None``.
    """
    if dim_policy_max is None or dim_policy == "flexible":
        return None
    return dim_policy_max, "metadata.dim_policy_max"


def _check_state_dim(schema: DataSchema, metadata: ModelMetadata) -> None:
    """Matrix row 1 — data vs model."""
    data_dim = schema.state_dim
    if data_dim == 0:
        return
    limit_info = _model_dim_limit(metadata.dim_policy, metadata.dim_policy_max)
    if limit_info is None:
        return
    limit, source = limit_info
    exact = metadata.dim_policy == "fixed"
    if (exact and data_dim != limit) or (not exact and data_dim > limit):
        raise make_error(
            STATE_DIM_INCOMPATIBLE, "schema.state_dim",
            data_dim=data_dim, limit=limit, limit_source=source,
        )


def _check_action_dim(
    schema: DataSchema, metadata: ModelMetadata,
    robot_profile: RobotProfile | None,
) -> None:
    """Matrix row 2 — data vs model vs robot."""
    data_dim = schema.action_dim
    if data_dim == 0:
        return
    if metadata.dim_policy != "flexible":
        limit = int(metadata.action_dim or metadata.dim_policy_max or 0)
        if limit > 0:
            exact = metadata.dim_policy == "fixed"
            if (exact and data_dim != limit) or (not exact and data_dim > limit):
                raise make_error(
                    ACTION_DIM_INCOMPATIBLE, "schema.action_dim",
                    data_dim=data_dim, limit=limit,
                    limit_source="metadata",
                )
    # Robot side is a coarse necessary condition only: the dataset's action
    # width cannot exceed how many joints the robot physically has. The
    # precise per-name relationship is the joint-order check's job — a robot
    # legitimately having MORE joints than the dataset records (e.g. LeKiwi's
    # mobile base, absent from arm-only training data) is not an error here.
    if robot_profile is not None and robot_profile.native_action_type in CONTROL_MODES:
        robot_limit = len(robot_profile.joints.names)
        if data_dim > robot_limit:
            raise make_error(
                ACTION_DIM_INCOMPATIBLE, "schema.action_dim",
                data_dim=data_dim, limit=robot_limit, limit_source="robot.joints",
            )


def _check_camera_slots_against(
    path_prefix: str,
    vision_slots: tuple[VisionSlot, ...],
    candidates: list[tuple[str, str]],
    missing_slot_policy: str,
) -> None:
    """One side (data or robot) of matrix row 3, checked independently."""
    for slot in vision_slots:
        hits = camera_candidates(slot, candidates)
        if len(hits) > 1:
            raise make_error(
                CAMERA_SLOT_AMBIGUOUS, f"{path_prefix}.{slot.name}",
                slot_name=slot.name, candidates=hits,
            )
        if not hits and slot.required and missing_slot_policy == "error":
            raise make_error(
                CAMERA_SLOT_UNRESOLVED, f"{path_prefix}.{slot.name}",
                slot_name=slot.name, missing_slot_policy=missing_slot_policy,
            )


def _check_camera_slots(
    schema: DataSchema, metadata: ModelMetadata, robot_profile: RobotProfile | None,
    overrides: dict[str, Any] | None = None,
) -> None:
    """Matrix row 3 — data/robot cameras vs model slots.

    Data and robot are checked independently (they feed different pipelines —
    ``data_to_model`` vs ``robot_to_model``), not pooled into one candidate
    set. A model with no declared ``vision_slots`` (e.g. ACT — vision follows
    the dataset) has nothing to check; the loop below is then trivially empty.

    A recipe that supplies ``assembly.camera_mapping`` skips the data side
    entirely: a controlled override *is* the final Mapping (architecture
    §4.2.3), so inferring anything for it — including whether it is ambiguous —
    is not this stage's call. The robot side still runs: the override names
    dataset cameras (that is what it is validated against), so it says nothing
    about the robot's.
    """
    if not (overrides or {}).get("camera_mapping"):
        _check_camera_slots_against(
            "model.vision_slots.data", metadata.vision_slots,
            data_camera_candidates(schema), metadata.missing_slot_policy,
        )
    if robot_profile is not None:
        _check_camera_slots_against(
            "model.vision_slots.robot", metadata.vision_slots,
            robot_camera_candidates(robot_profile), metadata.missing_slot_policy,
        )


def _check_control_mode(
    schema: DataSchema, metadata: ModelMetadata, robot_profile: RobotProfile | None,
) -> None:
    """Matrix row 5 — data/model/robot.

    A per-dim ``mode`` of ``None`` (undeclared) is not itself a problem here —
    data-module §8.3: the ``data_to_model`` path allows it; only a downstream
    ``model_to_robot`` plan would need every dim resolved, and that planning
    is the pipeline planner's job, not this check's.
    """
    data_modes = {d.mode for d in schema.action_dims if d.mode is not None}
    if not data_modes:
        return
    model_modes = set(metadata.control_mode_pref)
    robot_modes = set(robot_profile.control_modes) if robot_profile is not None else set()
    unsupported = set()
    if model_modes:
        unsupported |= data_modes - model_modes
    if robot_modes:
        unsupported |= data_modes - robot_modes
    if unsupported:
        raise make_error(
            CONTROL_MODE_INCOMPATIBLE, "schema.action.mode",
            data_modes=sorted(data_modes), model_modes=sorted(model_modes),
            robot_modes=sorted(robot_modes) if robot_profile is not None else None,
        )


def _norm_stats_missing_fields(stats: Any | None, method: str) -> list[str]:
    if stats is None:
        return ["mean", "std"] if method == "mean_std" else \
               ["q01", "q99"] if method == "quantile" else \
               ["min", "max"] if method == "min_max" else []
    missing: list[str] = []
    if method == "mean_std":
        if not stats.mean: missing.append("mean")
        if not stats.std: missing.append("std")
    elif method == "quantile":
        if not stats.q01: missing.append("q01")
        if not stats.q99: missing.append("q99")
    elif method == "min_max":
        if not stats.min: missing.append("min")
        if not stats.max: missing.append("max")
    return missing


def _check_norm_stats(
    schema: DataSchema, norm_stats: NormStats, metadata: ModelMetadata,
) -> None:
    """Matrix row 8 — data stats vs model method."""
    method = metadata.vector_normalization
    if method is None:
        return
    if schema.state_dim > 0:
        missing = _norm_stats_missing_fields(norm_stats.state, method)
        if missing:
            raise make_error(
                NORM_STATS_INSUFFICIENT, "norm_stats.state",
                field="state", method=method, missing=missing,
            )
    if schema.action_dim > 0:
        missing = _norm_stats_missing_fields(norm_stats.action, method)
        if missing:
            raise make_error(
                NORM_STATS_INSUFFICIENT, "norm_stats.action",
                field="action", method=method, missing=missing,
            )


def _check_joint_order(
    field: str, dims: tuple[StateDim, ...] | tuple[ActionDim, ...],
    robot_profile: RobotProfile | None,
) -> None:
    """Matrix row 10 — data keys vs robot joints (decision D4)."""
    if robot_profile is None:
        return
    names = [d.name for d in dims if d.name is not None]
    if not names:
        return
    _, unmatched, duplicates = embed_joints(names, robot_profile.joints.names)
    if unmatched:
        raise make_error(
            JOINT_ORDER_MISMATCH, f"schema.{field}.joint_order",
            field=field, unmatched_names=unmatched,
            robot_joint_names=list(robot_profile.joints.names),
        )
    if duplicates:
        raise make_error(
            JOINT_ORDER_AMBIGUOUS, f"schema.{field}.joint_order",
            field=field, duplicate_names=duplicates,
        )


def check_pairs(
    schema: DataSchema,
    norm_stats: NormStats,
    metadata: ModelMetadata,
    robot_profile: RobotProfile | None,
    overrides: dict[str, Any],
) -> None:
    """Run the six compatibility checks in order."""
    _check_state_dim(schema, metadata)
    _check_action_dim(schema, metadata, robot_profile)
    _check_camera_slots(schema, metadata, robot_profile, overrides)
    _check_control_mode(schema, metadata, robot_profile)
    _check_norm_stats(schema, norm_stats, metadata)
    _check_joint_order("state", schema.state_dims, robot_profile)
    _check_joint_order("action", schema.action_dims, robot_profile)

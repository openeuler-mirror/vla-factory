"""Check Pairs: compatibility checks backed by explicit interface facts.

The active checks cover state/action dimensions, control mode, and
normalization statistics. Camera validation belongs to camera mapping, because
validation and construction must use the same candidate set. Robot camera and joint names are
not compared with DataSchema names: those are different namespaces unless a
future explicit binding relates them. The other checks (language, gripper,
rotation, frequency, safety) remain out of scope until both sides expose
comparable facts and a real runtime consumer exists.

Each check raises on the first problem it finds rather than collecting them:
the user re-runs ``resolve`` after fixing one, exactly as for the earlier
stages, and a second error-reporting convention would buy nothing here.
"""

from __future__ import annotations

from typing import Any

from vla_factory.data.data_schema import DataSchema, NormStats
from vla_factory.model.model_interface import ModelMetadata
from vla_factory.robot import RobotProfile
from .errors import (
    ACTION_DIM_INCOMPATIBLE,
    CONTROL_MODE_INCOMPATIBLE,
    NORM_STATS_INSUFFICIENT,
    STATE_DIM_INCOMPATIBLE,
    make_error,
)


def _model_dim_limit(dim_policy: str, dim_policy_max: int | None) -> tuple[int, str] | None:
    """Return ``(limit, source)`` if ``dim_policy`` caps the dimension.

    ``fixed`` and ``padded_to_max`` both cap; ``flexible`` means "no declared
    limit". Validation rejects a bounded policy without a positive
    ``dim_policy_max`` before this helper can observe it.
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
            field="state", data_dim=data_dim, limit=limit, limit_source=source,
        )


def _check_action_dim(
    schema: DataSchema, metadata: ModelMetadata,
) -> None:
    """Matrix row 2 — data vs model."""
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
                    field="action", data_dim=data_dim, limit=limit,
                    limit_source="metadata",
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


def check_pairs(
    schema: DataSchema,
    norm_stats: NormStats,
    metadata: ModelMetadata,
    robot_profile: RobotProfile | None,
) -> None:
    """Run the compatibility checks in deterministic order."""
    _check_state_dim(schema, metadata)
    _check_action_dim(schema, metadata)
    _check_control_mode(schema, metadata, robot_profile)
    _check_norm_stats(schema, norm_stats, metadata)

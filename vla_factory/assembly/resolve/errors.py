"""Structured assembly-resolution failures with stable parameter shapes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MISSING_INPUT = "missing_input"
INVALID_DESCRIPTION = "invalid_description"
UNKNOWN_MODEL = "unknown_model"
UNKNOWN_ROBOT = "unknown_robot"
STATE_DIM_INCOMPATIBLE = "state_dim_incompatible"
ACTION_DIM_INCOMPATIBLE = "action_dim_incompatible"
CAMERA_SLOT_AMBIGUOUS = "camera_slot_ambiguous"
CAMERA_SLOT_UNRESOLVED = "camera_slot_unresolved"
CONTROL_MODE_INCOMPATIBLE = "control_mode_incompatible"
NORM_STATS_INSUFFICIENT = "norm_stats_insufficient"
CAMERA_MAPPING_INVALID = "camera_mapping_invalid"


@dataclass
class ResolutionError(Exception):
    code: str
    path: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(f"[{self.code}] {self.path}: {self.params}")

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "path": self.path, "params": dict(self.params)}


ERROR_PARAMS: dict[str, frozenset[str]] = {
    MISSING_INPUT: frozenset({"field", "detail"}),
    INVALID_DESCRIPTION: frozenset({"field", "value", "detail"}),
    UNKNOWN_MODEL: frozenset({"model_name", "known"}),
    UNKNOWN_ROBOT: frozenset({"robot_name", "known"}),
    STATE_DIM_INCOMPATIBLE: frozenset({"field", "data_dim", "limit", "limit_source"}),
    ACTION_DIM_INCOMPATIBLE: frozenset({"field", "data_dim", "limit", "limit_source"}),
    CAMERA_SLOT_AMBIGUOUS: frozenset({"slot_name", "candidates"}),
    CAMERA_SLOT_UNRESOLVED: frozenset({"slot_name", "missing_slot_policy"}),
    CONTROL_MODE_INCOMPATIBLE: frozenset({"data_modes", "model_modes", "robot_modes"}),
    NORM_STATS_INSUFFICIENT: frozenset({"field", "method", "missing"}),
    CAMERA_MAPPING_INVALID: frozenset({"field", "requested", "known"}),
}


def make_error(code: str, path: str, **params: Any) -> ResolutionError:
    """Create an error while enforcing the stable params schema for its code."""
    expected = ERROR_PARAMS.get(code)
    if expected is None:  # pragma: no cover - defensive
        raise KeyError(f"Unknown resolution error code: {code!r}")
    provided = set(params)
    if provided != expected:
        raise TypeError(
            f"{code} params mismatch: missing={sorted(expected - provided)}, "
            f"extra={sorted(provided - expected)}"
        )
    return ResolutionError(code=code, path=path, params=dict(params))

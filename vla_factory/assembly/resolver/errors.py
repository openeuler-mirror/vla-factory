"""Structured resolution errors (architecture §4.2.5).

The error contract keeps exactly three stable concepts:

* ``code``   — a stable machine error code (tests / CLI / tools key on this);
* ``path``   — the resolution target the error refers to (not necessarily a
               user recipe field path);
* ``params`` — the JSON-serializable facts needed to render a message.

User-readable text is **not** part of the stable contract. Each ``code`` is
produced only through a dedicated constructor in :data:`FACTORIES`, which fixes
the allowed ``params`` keys for that code — callers never assemble free-form
params.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


# ── Stable error codes ────────────────────────────────────────────

# A required input description was missing or unreadable.
MISSING_INPUT = "missing_input"
# A description is internally invalid (unknown field, bad enum value, ...).
INVALID_DESCRIPTION = "invalid_description"
# The named model is not in the registry.
UNKNOWN_MODEL = "unknown_model"
# The named robot profile is not registered.
UNKNOWN_ROBOT = "unknown_robot"

# ── Check Pairs codes (architecture §4.2.2) ───────────────────────
# Six of the eleven matrix rows: dimension, camera, stats, control mode, field
# order. The other five (language, gripper, rotation, frequency, safety) are
# deliberately out of scope; docs/plans/phase2-resolution-diagnostics.cn.md
# records why.

# Dataset state/action width cannot be reconciled with the model's declared
# dim_policy, or (action only) exceeds what the robot's joint vector can hold.
STATE_DIM_INCOMPATIBLE = "state_dim_incompatible"
ACTION_DIM_INCOMPATIBLE = "action_dim_incompatible"
# A model vision slot has more than one equally-valid camera candidate.
CAMERA_SLOT_AMBIGUOUS = "camera_slot_ambiguous"
# A model vision slot has zero candidates and its missing_slot_policy is
# "error" (the common "zero_pad"/"drop" cases are not errors — see WP2).
CAMERA_SLOT_UNRESOLVED = "camera_slot_unresolved"
# The dataset's declared action control modes share nothing with what the
# model (and, if given, the robot) accept.
CONTROL_MODE_INCOMPATIBLE = "control_mode_incompatible"
# NormStats is missing the fields the model's declared normalization method
# needs (mean/std for mean_std, q01/q99 for quantile).
NORM_STATS_INSUFFICIENT = "norm_stats_insufficient"
# Dataset dim names (after suffix stripping) collide onto the same robot joint.
JOINT_ORDER_AMBIGUOUS = "joint_order_ambiguous"
# One or more dataset dim names (after suffix stripping) match no robot joint.
JOINT_ORDER_MISMATCH = "joint_order_mismatch"

# ── Resolve Mapping / Plan Pipeline codes (architecture §4.2.3) ────

# A recipe set a controlled override the resolver has no consumer for. Silently
# dropping it would let a user believe they had adjusted a relationship the
# resolver never looked at — the same failure mode the model-config surface
# guards against by rejecting a declared-but-unread key.
UNSUPPORTED_OVERRIDE = "unsupported_override"

# A controlled ``assembly.camera_mapping`` override names a model slot the model
# does not declare, or a camera the dataset does not have. Without this the
# override would silently degrade to slot padding — the exact silent failure
# conservative resolution exists to prevent (§1.7).
CAMERA_MAPPING_INVALID = "camera_mapping_invalid"


@dataclass
class ResolutionError(Exception):
    """Structured composition-resolution failure.

    Attributes
    ----------
    code : str
        One of the ``*_code`` constants above.
    path : str
        Dotted resolution target this error refers to (e.g. ``"schema"``,
        ``"model.action_dim"``).
    params : dict
        JSON-serializable facts. The allowed keys are fixed per ``code`` by the
        dedicated constructor in :data:`FACTORIES`.
    """

    code: str
    path: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Allow ``raise ResolutionError(code=..., path=..., params=...)`` to
        # behave both as a dataclass and as an exception.
        super().__init__(f"[{self.code}] {self.path}: {self.params}")

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "path": self.path, "params": dict(self.params)}


# ── Dedicated constructors (fix the allowed params per code) ──────
#
# Each entry maps a code to a function ``(path, **facts) -> ResolutionError``.
# The function's signature documents and enforces the allowed params keys for
# that code; tests and the CLI should build errors through these helpers so the
# params schema cannot drift.


def _missing_input(path: str, *, field_name: str, detail: str = "") -> ResolutionError:
    return ResolutionError(
        code=MISSING_INPUT,
        path=path,
        params={"field": field_name, "detail": detail},
    )


def _invalid_description(
    path: str, *, field_name: str, value: Any = None, detail: str = ""
) -> ResolutionError:
    return ResolutionError(
        code=INVALID_DESCRIPTION,
        path=path,
        params={"field": field_name, "value": value, "detail": detail},
    )


def _unknown_model(path: str, *, model_name: str, known: list[str]) -> ResolutionError:
    return ResolutionError(
        code=UNKNOWN_MODEL,
        path=path,
        params={"model_name": model_name, "known": known},
    )


def _unknown_robot(path: str, *, robot_name: str, known: list[str]) -> ResolutionError:
    return ResolutionError(
        code=UNKNOWN_ROBOT,
        path=path,
        params={"robot_name": robot_name, "known": known},
    )


def _dim_incompatible(
    code: str, path: str, *, field: str, data_dim: int, limit: int, limit_source: str
) -> ResolutionError:
    return ResolutionError(
        code=code,
        path=path,
        params={
            "field": field, "data_dim": data_dim,
            "limit": limit, "limit_source": limit_source,
        },
    )


def _state_dim_incompatible(
    path: str, *, data_dim: int, limit: int, limit_source: str
) -> ResolutionError:
    return _dim_incompatible(
        STATE_DIM_INCOMPATIBLE, path,
        field="state", data_dim=data_dim, limit=limit, limit_source=limit_source,
    )


def _action_dim_incompatible(
    path: str, *, data_dim: int, limit: int, limit_source: str
) -> ResolutionError:
    return _dim_incompatible(
        ACTION_DIM_INCOMPATIBLE, path,
        field="action", data_dim=data_dim, limit=limit, limit_source=limit_source,
    )


def _camera_slot_ambiguous(
    path: str, *, slot_name: str, candidates: list[str]
) -> ResolutionError:
    return ResolutionError(
        code=CAMERA_SLOT_AMBIGUOUS,
        path=path,
        params={"slot_name": slot_name, "candidates": candidates},
    )


def _camera_slot_unresolved(
    path: str, *, slot_name: str, missing_slot_policy: str
) -> ResolutionError:
    return ResolutionError(
        code=CAMERA_SLOT_UNRESOLVED,
        path=path,
        params={"slot_name": slot_name, "missing_slot_policy": missing_slot_policy},
    )


def _control_mode_incompatible(
    path: str, *, data_modes: list[str], model_modes: list[str],
    robot_modes: list[str] | None,
) -> ResolutionError:
    return ResolutionError(
        code=CONTROL_MODE_INCOMPATIBLE,
        path=path,
        params={
            "data_modes": data_modes, "model_modes": model_modes,
            "robot_modes": robot_modes,
        },
    )


def _norm_stats_insufficient(
    path: str, *, field: str, method: str, missing: list[str]
) -> ResolutionError:
    return ResolutionError(
        code=NORM_STATS_INSUFFICIENT,
        path=path,
        params={"field": field, "method": method, "missing": missing},
    )


def _joint_order_ambiguous(
    path: str, *, field: str, duplicate_names: list[str]
) -> ResolutionError:
    return ResolutionError(
        code=JOINT_ORDER_AMBIGUOUS,
        path=path,
        params={"field": field, "duplicate_names": duplicate_names},
    )


def _joint_order_mismatch(
    path: str, *, field: str, unmatched_names: list[str], robot_joint_names: list[str]
) -> ResolutionError:
    return ResolutionError(
        code=JOINT_ORDER_MISMATCH,
        path=path,
        params={
            "field": field, "unmatched_names": unmatched_names,
            "robot_joint_names": robot_joint_names,
        },
    )


def _unsupported_override(
    path: str, *, keys: list[str], supported: list[str]
) -> ResolutionError:
    return ResolutionError(
        code=UNSUPPORTED_OVERRIDE,
        path=path,
        params={"keys": keys, "supported": supported},
    )


def _camera_mapping_invalid(
    path: str, *, field: str, requested: str, known: list[str]
) -> ResolutionError:
    """``field`` is ``"slot"`` or ``"camera"`` — which half of the override
    entry could not be found; ``known`` is the sorted candidate list for it."""
    return ResolutionError(
        code=CAMERA_MAPPING_INVALID,
        path=path,
        params={"field": field, "requested": requested, "known": known},
    )


# code → constructor. Callers should go through this mapping so the allowed
# params shape stays coupled to the code.
FACTORIES: dict[str, Callable[..., ResolutionError]] = {
    MISSING_INPUT: _missing_input,
    INVALID_DESCRIPTION: _invalid_description,
    UNKNOWN_MODEL: _unknown_model,
    UNKNOWN_ROBOT: _unknown_robot,
    STATE_DIM_INCOMPATIBLE: _state_dim_incompatible,
    ACTION_DIM_INCOMPATIBLE: _action_dim_incompatible,
    CAMERA_SLOT_AMBIGUOUS: _camera_slot_ambiguous,
    CAMERA_SLOT_UNRESOLVED: _camera_slot_unresolved,
    CONTROL_MODE_INCOMPATIBLE: _control_mode_incompatible,
    NORM_STATS_INSUFFICIENT: _norm_stats_insufficient,
    JOINT_ORDER_AMBIGUOUS: _joint_order_ambiguous,
    JOINT_ORDER_MISMATCH: _joint_order_mismatch,
    CAMERA_MAPPING_INVALID: _camera_mapping_invalid,
    UNSUPPORTED_OVERRIDE: _unsupported_override,
}


def make_error(code: str, path: str, **params: Any) -> ResolutionError:
    """Build a ``ResolutionError`` through the dedicated constructor for *code*.

    Centralising construction here guarantees each code only ever carries its
    allowed params keys.
    """
    factory = FACTORIES.get(code)
    if factory is None:  # pragma: no cover - defensive
        raise ValueError(f"Unknown resolution error code {code!r}")
    return factory(path, **params)

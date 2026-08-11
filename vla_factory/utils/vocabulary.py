"""Cross-dimension controlled vocabularies and source annotations.

Architecture §4.5 / model-module §4.5 require three vocabularies to be defined
in exactly one place and referenced by the data, model and robot dimensions —
so a value cannot silently drift between a dataset's ``dims[].mode``, a model's
``control_mode_pref`` and a ``RobotProfile.control_modes``. The leaf layers
(``data/``, ``model/``, ``robot/``) may import this module; nothing here
depends on ``assembly/`` (architecture §2.2 dependency direction).

Phase 1 (WP0) lands the vocabularies + source-annotation types. The first
control-mode vocabulary is joint-space only — ``joint_pos`` / ``joint_delta`` /
``joint_vel``. EEF modes (``eef_pos`` / ``eef_delta`` / ``se3``) and rotation
representations are deferred as a group, entering together with EEF model
adaptation (data-module §8.3); the resolver reports an unknown control mode via
a structured ``ResolutionError`` rather than accepting it silently.
"""

from __future__ import annotations

from typing import Final, Literal


# ── Camera semantics (data cameras[].semantic = model slots[].semantic_accepts)
# Specific view roles plus the generalization ``third_person`` (a model slot may
# accept any third-person view via the generalization).
CAMERA_SEMANTICS: Final[frozenset[str]] = frozenset({
    "third_person_front",
    "third_person_top",
    "third_person_side",
    "wrist_left",
    "wrist_right",
    "wrist",
    "third_person",  # generalization: any third-person view
})

CameraSemantic = Literal[
    "third_person_front", "third_person_top", "third_person_side",
    "wrist_left", "wrist_right", "wrist", "third_person",
]


# ── Control modes (data action.dims[].mode = model control_mode_pref =
#    RobotProfile control_modes). Joint-space only in the first version.
CONTROL_MODES: Final[tuple[str, ...]] = ("joint_pos", "joint_delta", "joint_vel")
_CONTROL_MODE_SET: Final[frozenset[str]] = frozenset(CONTROL_MODES)

ControlMode = Literal["joint_pos", "joint_delta", "joint_vel"]


def is_control_mode(value: str) -> bool:
    """True if *value* is in the control-mode vocabulary."""
    return value in _CONTROL_MODE_SET


# ── Action heads (model action_head_type; tokenized AR output folds into
#    ``autoregressive``). Stable across model metadata and the resolver.
ACTION_HEADS: Final[tuple[str, ...]] = (
    "flow_matching", "diffusion", "autoregressive", "regression",
)
_ACTION_HEAD_SET: Final[frozenset[str]] = frozenset(ACTION_HEADS)

ActionHead = Literal["flow_matching", "diffusion", "autoregressive", "regression"]


def is_action_head(value: str) -> bool:
    """True if *value* is in the action-head vocabulary."""
    return value in _ACTION_HEAD_SET


# ── Source annotations — every declared fact records where it came from.
#
# Data-side facts (DataSchema): measured (directly probed) / inferred (unique
# deterministic match under a controlled vocabulary) / undeclared (null — not an
# error, it is the resolver's trigger for a controlled override).
DataSource = Literal["measured", "inferred", "undeclared"]

DATA_SOURCES: Final[frozenset[str]] = frozenset({"measured", "inferred", "undeclared"})


__all__ = [
    "CAMERA_SEMANTICS",
    "CameraSemantic",
    "CONTROL_MODES",
    "ControlMode",
    "is_control_mode",
    "ACTION_HEADS",
    "ActionHead",
    "is_action_head",
    "DataSource",
    "DATA_SOURCES",
]

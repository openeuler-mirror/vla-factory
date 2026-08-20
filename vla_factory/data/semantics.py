"""Deterministic inference rules for data-side semantics (data-module §8.5).

Two data facts are *inferred* (not directly probed): a camera's ``semantic``
role and an action dim's ``mode``. Both use explicit, versioned rules:

- **Camera: unique best match only** — more specific rules outrank general
  ones, while two semantics tied at the highest matching priority yield
  ``None``. Rule-table order never breaks a tie.
- **Action: exact suffix only** — only a controlled suffix maps to a mode.
- Zero or ambiguous evidence stays undeclared so the resolver can require a
  controlled override. No similarity guessing (§1.7).
- **Container formats carry no default** — a generic container (lerobot can
  hold any action source) gets no per-format default; only a format whose spec
  binds the production pipeline (RoboTwin) yields ``measured`` evidence, and
  that is decided in the reader, not here.

These rules are framework code (versioned, unit-tested), not user config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from vla_factory.utils.vocabulary import CAMERA_SEMANTICS, CONTROL_MODES


@dataclass(frozen=True)
class _CameraSemanticRule:
    semantic: str
    priority: int
    all_of: tuple[str, ...] = ()
    any_of: tuple[str, ...] = ()

    def matches(self, key: str) -> bool:
        return all(token in key for token in self.all_of) and (
            not self.any_of or any(token in key for token in self.any_of)
        )


_CAMERA_SEMANTIC_RULES: tuple[_CameraSemanticRule, ...] = (
    # Directional wrist evidence is more specific than a generic wrist view.
    _CameraSemanticRule("wrist_left", priority=30, all_of=("wrist", "left")),
    _CameraSemanticRule("wrist_right", priority=30, all_of=("wrist", "right")),
    _CameraSemanticRule("wrist", priority=20, all_of=("wrist",)),
    # Top and side are equally specific: a key carrying both remains ambiguous.
    _CameraSemanticRule(
        "third_person_top", priority=10,
        any_of=("top", "high", "overhead"),
    ),
    _CameraSemanticRule("third_person_side", priority=10, any_of=("side",)),
    # Explicit top/side wording outranks the weaker front/head convention.
    _CameraSemanticRule(
        "third_person_front", priority=0, any_of=("front", "head"),
    ),
)


def infer_camera_semantic(key: str) -> str | None:
    """Return the unique highest-priority semantic matching a camera key."""
    matches = [
        rule for rule in _CAMERA_SEMANTIC_RULES
        if rule.semantic in CAMERA_SEMANTICS and rule.matches(key.lower())
    ]
    if not matches:
        return None
    highest_priority = max(rule.priority for rule in matches)
    best = {
        rule.semantic for rule in matches
        if rule.priority == highest_priority
    }
    return next(iter(best)) if len(best) == 1 else None


# lerobot-style names carry the source as a suffix (``.pos`` / ``.vel`` /
# ``.delta``). Keep suffix recognition in one table so mode inference and any
# explicit name-normalization consumer cannot drift apart.
_SUFFIX_TO_MODE = {"pos": "joint_pos", "vel": "joint_vel", "delta": "joint_delta"}


def infer_action_mode(name: str) -> str | None:
    """Map an action dim name suffix to a ``CONTROL_MODES`` value.

    Returns the unique match or ``None`` (undeclared) — including for an
    empty/``None`` name.
    """
    if not name:
        return None
    suffix = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    mode = _SUFFIX_TO_MODE.get(suffix)
    return mode if mode in CONTROL_MODES else None


def strip_known_suffix(name: str) -> str:
    """Strip a trailing ``.pos`` / ``.vel`` / ``.delta`` suffix, if present.

    Used to compare a dataset dim name (``shoulder_pan.pos``, suffix kept per
    data-module §8.3) against a robot's joint name (``shoulder_pan``, no
    suffix) on equal footing. Names without a known suffix pass through
    unchanged — the resolver's joint-order match then falls through to "no
    match", not a false strip.
    """
    if "." not in name:
        return name
    base, suffix = name.rsplit(".", 1)
    return base if suffix.lower() in _SUFFIX_TO_MODE else name


# Source-label convenience (kept as plain strings for JSON serialization).
DATA_MEASURED: Literal["measured"] = "measured"
DATA_INFERRED: Literal["inferred"] = "inferred"
DATA_UNDECLARED: Literal["undeclared"] = "undeclared"

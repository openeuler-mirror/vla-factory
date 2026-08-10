"""Deterministic inference rules for data-side semantics (data-module §8.5).

Two data facts are *inferred* (not directly probed): a camera's ``semantic``
role and an action dim's ``mode``. Both follow the same discipline:

- **Unique match only** — exactly one vocabulary candidate may hit; zero or
  several candidates yield ``None`` (undeclared). The resolver then asks for a
  controlled override. No dictionary-order / similarity guessing (§1.7).
- **Container formats carry no default** — a generic container (lerobot can
  hold any action source) gets no per-format default; only a format whose spec
  binds the production pipeline (RoboTwin) yields ``measured`` evidence, and
  that is decided in the reader, not here.

These rules are framework code (versioned, unit-tested), not user config.
"""

from __future__ import annotations

from typing import Literal

from vla_factory.utils.vocabulary import CAMERA_SEMANTICS, CONTROL_MODES


def infer_camera_semantic(key: str) -> str | None:
    """Map a dataset camera key to a ``CAMERA_SEMANTICS`` role.

    Returns the unique matching role, or ``None`` when zero / more than one
    candidate matches. Matching is case-insensitive substring on the key.
    """
    k = key.lower()
    # Predicates are made mutually exclusive so that e.g. ``cam_left_wrist``
    # matches ``wrist_left`` only (not also the bare ``wrist``), preserving the
    # "unique match" guarantee.
    candidates: list[str] = []
    if "wrist" in k and "left" in k:
        candidates.append("wrist_left")
    if "wrist" in k and "right" in k:
        candidates.append("wrist_right")
    if "wrist" in k and "left" not in k and "right" not in k:
        candidates.append("wrist")
    if ("top" in k or "high" in k) and "wrist" not in k:
        candidates.append("third_person_top")
    if ("front" in k or "head" in k) and "wrist" not in k \
            and "top" not in k and "high" not in k and "side" not in k:
        candidates.append("third_person_front")
    if "side" in k and "wrist" not in k:
        candidates.append("third_person_side")
    if len(candidates) != 1:
        return None
    role = candidates[0]
    return role if role in CAMERA_SEMANTICS else None


def infer_action_mode(name: str) -> str | None:
    """Map an action dim name suffix to a ``CONTROL_MODES`` value.

    lerobot-style names carry the source as a suffix (``.pos`` / ``.vel`` /
    ``.delta``). Returns the unique match or ``None`` (undeclared) — including
    for an empty/``None`` name.
    """
    if not name:
        return None
    suffix = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    mapping = {"pos": "joint_pos", "vel": "joint_vel", "delta": "joint_delta"}
    mode = mapping.get(suffix)
    return mode if mode in CONTROL_MODES else None


# Source-label convenience (kept as plain strings for JSON serialization).
DATA_MEASURED: Literal["measured"] = "measured"
DATA_INFERRED: Literal["inferred"] = "inferred"
DATA_UNDECLARED: Literal["undeclared"] = "undeclared"

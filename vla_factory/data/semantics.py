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


# lerobot-style names carry the source as a suffix (``.pos`` / ``.vel`` /
# ``.delta``). Single source of truth for both the mode inference below and
# the resolver's joint-order matching (architecture §7.4 phase-2 decision D4)
# — a dataset name and a robot joint name refer to the same joint iff they are
# equal after stripping one of these.
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

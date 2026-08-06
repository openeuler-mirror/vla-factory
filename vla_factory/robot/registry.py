"""Robot-profile registry.

Profiles ship as YAML under ``vla_factory/robot/profiles/*.yaml`` and are
looked up by name on demand. Unknown names raise ``KeyError``-style errors so
callers (CLI ``resolve``, composition resolver) can surface a clear message.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import yaml

from .profile import RobotProfile, profile_from_dict


def _profiles_dir() -> Path:
    """Absolute path to the bundled ``profiles/`` directory."""
    # ``files()`` resolves to the source location (and installed package-data).
    return Path(str(resources.files("vla_factory.robot").joinpath("profiles")))


def list_robot_profiles() -> list[str]:
    """Return the sorted names of all bundled robot profiles (without ``.yaml``)."""
    d = _profiles_dir()
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.yaml"))


def get_robot_profile(name: str) -> RobotProfile:
    """Load and validate the bundled robot profile ``name``.

    Raises ``FileNotFoundError`` if no such profile exists. Use
    ``list_robot_profiles()`` to discover available names.
    """
    if not name:
        raise ValueError("robot profile name must be a non-empty string")
    candidate = _profiles_dir() / f"{name}.yaml"
    if not candidate.is_file():
        available = list_robot_profiles()
        raise FileNotFoundError(
            f"Unknown robot profile {name!r}. "
            f"Known profiles: {available}."
        )
    raw = yaml.safe_load(candidate.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"robot profile {name!r} must be a YAML mapping")
    return profile_from_dict(raw)

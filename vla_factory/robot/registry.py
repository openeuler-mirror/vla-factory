"""Robot-profile registry.

Profiles ship as YAML under ``vla_factory/robot/profiles/*.yaml`` and are
looked up by name on demand. Unknown names raise ``KeyError``-style errors so
callers (CLI ``resolve``, composition resolver) can surface a clear message.
"""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path

import yaml

from .profile import RobotProfile


# Name lookup must stay inside the bundled ``profiles/`` directory: a bare
# stem only, so ``..``-segments or absolute paths cannot escape it.
_PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


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


def load_robot_profile(path: str | Path) -> RobotProfile:
    """Load and validate a robot profile from a YAML file."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"robot profile {path!r} must be a YAML mapping")
    return RobotProfile.from_dict(raw)


def get_robot_profile(name: str) -> RobotProfile:
    """Load and validate the bundled robot profile ``name``.

    Raises ``FileNotFoundError`` if no such profile exists. Use
    ``list_robot_profiles()`` to discover available names.
    """
    if not name:
        raise ValueError("robot profile name must be a non-empty string")
    if _PROFILE_NAME_RE.match(name) is None:
        raise ValueError(
            f"Invalid robot profile name {name!r}: expected a bare profile "
            "stem (letters, digits, '_', '-'), not a path. Use "
            "load_robot_profile(path) to load a profile from a specific file."
        )
    candidate = _profiles_dir() / f"{name}.yaml"
    if not candidate.is_file():
        available = list_robot_profiles()
        raise FileNotFoundError(
            f"Unknown robot profile {name!r}. "
            f"Known profiles: {available}."
        )
    return load_robot_profile(candidate)

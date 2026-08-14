"""Dataset reader registration and selection."""

from __future__ import annotations

from pathlib import Path

from .base import FormatReader
from .registry import ReaderRegistry

# Import built-ins once so their decorators populate the registry. Optional
# format dependencies remain lazy inside reader methods.
from .lerobot_v3 import LeRobotV3Reader
from .robotwin import RoboTwinReader


def get_reader(format_name: str, path: Path | None = None) -> FormatReader:
    """Construct a named reader or auto-detect one from ``path``."""
    if format_name.lower() == "auto":
        if path is None:
            raise ValueError("'auto' reader selection requires a dataset path")
        return ReaderRegistry.detect(path)
    return ReaderRegistry.create(format_name)


__all__ = [
    "FormatReader",
    "LeRobotV3Reader",
    "ReaderRegistry",
    "RoboTwinReader",
    "get_reader",
]

"""Format reader registry.

Public API:
    get_reader(format, path)  -> FormatReader
"""

from __future__ import annotations

from pathlib import Path

from .base import Episode, FormatReader, Frame, VideoRef
from .lerobot_v3 import LeRobotV3Reader

__all__ = [
    "Episode",
    "FormatReader",
    "Frame",
    "LeRobotV3Reader",
    "VideoRef",
    "get_reader",
]


# ── Format reader registry ───────────────────────────────────────

_READERS: dict[str, FormatReader] = {
    "lerobot-v3": LeRobotV3Reader(),
    "lerobot_v3": LeRobotV3Reader(),
}


def get_reader(format_name: str, path: Path | None = None) -> FormatReader:
    """Look up a ``FormatReader`` by format name.

    If ``format_name`` is ``"auto"``, probes all registered readers.
    """
    if format_name == "auto" and path is not None:
        for reader in _READERS.values():
            if reader.can_read(path):
                return reader
        raise ValueError(f"No reader found for path: {path}")

    key = format_name.lower()
    if key in _READERS:
        return _READERS[key]

    raise ValueError(
        f"Unknown format '{format_name}'. Available: {list(_READERS.keys())}"
    )

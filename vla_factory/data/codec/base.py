"""Video codec protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from numpy.typing import NDArray

from ..formats.base import VideoRef


@runtime_checkable
class VideoCodec(Protocol):
    """Pluggable video decoding strategy."""

    @property
    def name(self) -> str: ...

    def decode_frame(self, ref: VideoRef) -> NDArray:
        """Decode a single frame -> numpy HWC uint8."""
        ...

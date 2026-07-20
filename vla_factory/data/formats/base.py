"""Canonical IR and format reader protocol.

Framework-agnostic intermediate representation:
  - ``VideoRef``  — lazy video frame reference (no decoding logic)
  - ``Frame``     — single timestep (state, action, image refs)
  - ``Episode``   — collection of frames with lazy loading

Protocols:
  - ``FormatReader``  — dataset format reader (parquet, hdf5, etc.)

Video decoding is handled by ``codec/`` (see ``VideoCodec`` protocol).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ..manifest import DataSchema, NormStats

if TYPE_CHECKING:
    from ..codec.base import VideoCodec


# ── Canonical IR ──────────────────────────────────────────────────


@dataclass(frozen=True)
class VideoRef:
    """Lazy video frame reference — records location only, no loader closure."""

    video_path: Path
    frame_index: int
    height: int
    width: int
    channels: int = 3


@dataclass
class Frame:
    """Single timestep: state, action, and per-camera video references."""

    index: int
    images: dict[str, VideoRef]  # cam_name → VideoRef
    state: NDArray | None
    action: NDArray | None
    timestamp: float | None = None
    is_first: bool = False
    is_last: bool = False
    language: str | None = None  # task instruction (language-conditioned models: pi0, pi05)


@dataclass
class Episode:
    """An episode with lazy frame loading.

    Frames are materialised on demand via ``load_frames()`` or iterated
    via ``frames()``.  The internal ``_frame_loader`` is a callable that
    produces an iterator of ``Frame`` objects.
    """

    episode_id: str
    episode_index: int
    num_frames: int
    _frame_loader: object | None = None  # Callable[[], Iterator[Frame]]
    _frames_cache: list[Frame] | None = field(default=None, repr=False)

    def frames(self) -> Iterator[Frame]:
        """Iterate frames (uses cache if already loaded)."""
        if self._frames_cache is not None:
            yield from self._frames_cache
            return
        if self._frame_loader is not None:
            yield from self._frame_loader()

    def load_frames(self) -> list[Frame]:
        """Force materialise all frames and cache them."""
        if self._frames_cache is not None:
            return self._frames_cache
        if self._frame_loader is None:
            return []
        self._frames_cache = list(self._frame_loader())
        return self._frames_cache


# ── Protocols ─────────────────────────────────────────────────────


@runtime_checkable
class FormatReader(Protocol):
    """Dataset format reader (parquet, hdf5, etc.)."""

    def can_read(self, path: Path) -> bool:
        """Return True if this reader can handle the given path."""
        ...

    def get_schema(self, path: Path) -> DataSchema:
        """Read dataset schema from the given path."""
        ...

    def get_norm_stats(self, path: Path) -> NormStats:
        """Read normalisation statistics from the given path."""
        ...

    def get_episode_lengths(self, path: Path) -> dict[int, int]:
        """Return ``{episode_index: num_frames}``."""
        ...

    def get_episode_ranges(self, path: Path) -> dict[int, tuple[int, int]]:
        """Return ``{episode_index: (global_start, global_end)}`` (inclusive)."""
        ...

    def read_episode(
        self, path: Path, episode_index: int, codec: VideoCodec
    ) -> Episode:
        """Read a single episode with the given video codec."""
        ...

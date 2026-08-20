"""Base interface implemented by dataset format readers."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable, TYPE_CHECKING

from ..data_schema import DataSchema, Episode, NormStats

if TYPE_CHECKING:
    from ..codec.base import VideoCodec


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

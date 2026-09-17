"""Video codec protocol and the shared open-handle registry."""

from __future__ import annotations

from collections import OrderedDict
from typing import Protocol, runtime_checkable

from numpy.typing import NDArray

from ..data_schema import VideoRef


@runtime_checkable
class VideoCodec(Protocol):
    """Pluggable video decoding strategy."""

    @property
    def name(self) -> str: ...

    def decode_frame(self, ref: VideoRef) -> NDArray:
        """Decode a single frame -> numpy HWC uint8."""
        ...


class OpenHandleLRU:
    """Bounded LRU registry of open per-file handles.

    Works like a textbook LRU cache with one extra rule: values are
    per-file decoding handles (an fd plus decoder state) and **evicting an
    entry closes it** — so ``workers x max_open`` stays under the fd limit
    no matter how many distinct files a dataset has.

    Not thread-safe by design: each instance lives in a single thread (or
    one fork-isolated DataLoader worker; the codec is inherited, so the
    bound is per process).
    """

    def __init__(self, max_open: int) -> None:
        self.max_open = max_open
        self.entries: OrderedDict = OrderedDict()

    def get(self, key):
        """Return the handle for ``key`` (marking it most-recently-used), or None."""
        if key not in self.entries:
            return None
        self.entries.move_to_end(key)
        return self.entries[key]

    def put(self, key, value):
        """Insert ``value``; evict-and-close LRU entries beyond capacity."""
        if key in self.entries:
            self.entries.move_to_end(key)
        self.entries[key] = value
        while len(self.entries) > self.max_open:
            _, evicted = self.entries.popitem(last=False)
            evicted.close()

    def close(self):
        """Close every handle and empty the registry."""
        for handle in self.entries.values():
            handle.close()
        self.entries.clear()

    def __len__(self):
        return len(self.entries)

    def __contains__(self, key):
        return key in self.entries

    def __getitem__(self, key):
        """Peek at the entry for ``key`` without touching LRU recency."""
        return self.entries[key]

    def values(self):
        """Snapshot of the retained handles (inspection only)."""
        return list(self.entries.values())

"""Video codec protocol and shared decoder cache."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Hashable, Protocol, runtime_checkable

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


class CachedVideoCodec(ABC):
    """Shared per-file decoder and decoded-frame caches for video codecs."""

    def __init__(self, max_cached_per_video: int = 32, max_open_videos: int = 32) -> None:
        self._max_cached = max_cached_per_video
        self._decoders = VideoDecoderCache(max_open_videos, self._close_decoder)

    def decode_frame(self, ref: VideoRef) -> NDArray:
        """Return a cached frame or decode and cache it."""
        decoder = self._decoder_for(ref.video_path)
        key = self._cache_key(ref)
        cached = decoder["_cache"].get(key)
        if cached is not None:
            decoder["_cache"].move_to_end(key)
            return cached.copy()

        image = self._decode_frame(decoder, ref)
        decoder["_cache"][key] = image.copy()
        if len(decoder["_cache"]) > self._max_cached:
            decoder["_cache"].popitem(last=False)
        return image

    def _decoder_for(self, video_path: Path) -> dict[str, Any]:
        decoder = self._decoders.get(video_path)
        if decoder is None:
            decoder = self._open_decoder(video_path)
            decoder["_cache"] = OrderedDict()
            decoder["video_path"] = video_path
            self._decoders.put(video_path, decoder)
        return decoder

    def _cache_key(self, ref: VideoRef) -> Hashable:
        return ref.frame_index

    @abstractmethod
    def _open_decoder(self, video_path: Path) -> dict[str, Any]:
        """Create one backend-specific decoder state dictionary."""

    @abstractmethod
    def _decode_frame(self, decoder: dict[str, Any], ref: VideoRef) -> NDArray:
        """Decode one uncached frame using the backend-specific state."""

    def close(self) -> None:
        """Close all backend decoders and clear their frame caches."""
        self._decoders.close()

    def __del__(self) -> None:
        self.close()

    def _close_decoder(self, decoder: dict[str, Any]) -> None:
        decoder["_cache"].clear()


class VideoDecoderCache:
    """Bounded LRU cache of per-file video decoders.

    Works like a textbook LRU cache with one extra rule: values are
    per-file decoders (an fd plus decoder state) and **evicting an
    entry closes it** — so ``workers x max_size`` stays under the fd limit
    no matter how many distinct files a dataset has.

    Not thread-safe by design: each instance lives in a single thread (or
    one fork-isolated DataLoader worker; the codec is inherited, so the
    bound is per process).
    """

    def __init__(
        self, max_size: int, close: Callable[[Any], None] | None = None
    ) -> None:
        self.max_size = max_size
        self._cache: OrderedDict = OrderedDict()
        self._close = close

    def get(self, key):
        """Return the decoder for ``key`` (marking it most-recently-used), or None."""
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key, value):
        """Insert a decoder; evict-and-close LRU entries beyond capacity."""
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        while len(self._cache) > self.max_size:
            _, evicted = self._cache.popitem(last=False)
            self._close_entry(evicted)

    def close(self):
        """Close every decoder and empty the cache."""
        for handle in self._cache.values():
            self._close_entry(handle)
        self._cache.clear()

    def _close_entry(self, entry) -> None:
        if self._close is None:
            entry.close()
        else:
            self._close(entry)

    def __len__(self):
        return len(self._cache)

    def __contains__(self, key):
        return key in self._cache

    def __getitem__(self, key):
        """Peek at the entry for ``key`` without touching LRU recency."""
        return self._cache[key]

    def values(self):
        """Snapshot of the retained handles (inspection only)."""
        return list(self._cache.values())

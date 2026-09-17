"""TorchCodec-based video decoder — optional VideoCodec implementation.

Uses PyTorch's ``torchcodec`` library (https://github.com/pytorch/torchcodec)
for frame-accurate decoding. ``torchcodec`` is an optional dependency: it is
imported lazily on first decode so the codec registry stays importable without
it (rationale: framework-wide "optional deps defer to call time").

Caching mirrors :class:`PyAVCodec` so both codecs behave identically on the
pipeline: one open decoder handle per video file, an in-memory LRU of recently
decoded frames (``max_cached_per_video``), and an optional ``.npy`` disk cache
under ``<video>.frame_cache/`` shared with PyAV. torchcodec's
``get_frame_at(index=...)`` is frame-accurate random access, so no manual seek
bookkeeping is needed (unlike PyAV's ``_seek_to``).

Frames are requested with ``dimension_order="NHWC"`` so they come out as numpy
HWC uint8 with no transposition, matching the codec contract used by the rest
of the pipeline. Decoding runs on CPU: the contract is numpy arrays, so there
is no point holding frames on a CUDA device.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from ..data_schema import VideoRef
from .registry import CodecRegistry

logger = logging.getLogger(__name__)


def _load_torchcodec() -> Any:
    """Import the torchcodec ``VideoDecoder`` lazily with an actionable error.

    A torchcodec wheel built against a different torch version fails to load
    its C++ core at import time; torchcodec surfaces that as ``RuntimeError``
    ("Could not load libtorchcodec"), while a missing install raises
    ``ImportError``. We catch all three so the user gets a hint pointing at
    the fix, not a raw linker traceback.
    """
    try:
        from torchcodec.decoders import VideoDecoder
    except (ImportError, OSError, RuntimeError) as exc:  # pragma: no cover - exercised only without the extra / with an ABI-mismatched wheel
        raise ImportError(
            "Reading videos with the 'torchcodec' codec requires the "
            "'torchcodec' package built for the installed torch (wheels are "
            "torch-version-locked; a mismatched wheel fails to load). Install "
            "the extra: pip install -e \".[torchcodec]\"."
        ) from exc
    return VideoDecoder


class _TorchFrameCache:
    """Per-video-file cache that keeps a torchcodec ``VideoDecoder`` open.

    Mirrors ``_VideoFrameCache`` (PyAV): one decoder handle per video file plus
    an LRU of recently decoded frames. torchcodec decodes by frame index
    directly (``get_frame_at``), so unlike PyAV there is no seek/position
    tracking to get wrong on out-of-order access.
    """

    def __init__(self, video_path: Path, max_cached: int = 32) -> None:
        self.video_path = video_path
        self.max_cached = max_cached
        self._decoder = None
        self._cache: OrderedDict[int, NDArray] = OrderedDict()

    def _ensure_open(self) -> None:
        """Open the torchcodec decoder lazily on first access."""
        if self._decoder is not None:
            return
        VideoDecoder = _load_torchcodec()
        # NHWC matches the codec contract (numpy HWC uint8). device="cpu": the
        # pipeline never holds video frames on GPU (see module docstring).
        self._decoder = VideoDecoder(
            source=str(self.video_path),
            dimension_order="NHWC",
            device="cpu",
        )

    def get_frame(self, frame_idx: int, dims: tuple[int, ...]) -> NDArray:
        """Get a decoded frame as numpy HWC uint8, LRU-cached."""
        # Check cache first
        if frame_idx in self._cache:
            self._cache.move_to_end(frame_idx)
            # Hand out a copy: the cached array is shared across all future
            # reads, so callers must never be able to mutate it in place.
            return self._cache[frame_idx].copy()

        self._ensure_open()
        frame = self._decoder.get_frame_at(index=frame_idx).data
        img = frame.cpu().numpy()
        h, w, c = dims
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h))

        # Cache (FIFO eviction on overflow, mirroring PyAV's in-memory LRU).
        # Store a private copy and return the freshly decoded array: callers
        # own their result and cannot corrupt the LRU by mutating it.
        self._cache[frame_idx] = img.copy()
        if len(self._cache) > self.max_cached:
            self._cache.popitem(last=False)
        return img

    def close(self) -> None:
        """Drop the decoder handle and clear the frame cache."""
        self._decoder = None
        self._cache.clear()

    def __del__(self) -> None:
        self.close()


@CodecRegistry.register("torchcodec")
class TorchCodec:
    """Optional video codec — uses torchcodec to decode frames to numpy.

    Caching mirrors :class:`PyAVCodec`: a per-video-file cache of decoder
    handles plus a decoded-frame LRU, and a shared ``.npy`` disk cache under
    ``<video>.frame_cache/`` (the same files ``preprocess_video`` fills).
    Native decoder resources are released when the cache is closed/dropped.
    """

    def __init__(self, max_cached_per_video: int = 32, disk_cache: bool = True) -> None:
        self._caches: dict[Path, _TorchFrameCache] = {}
        self._max_cached = max_cached_per_video
        self._disk_cache = disk_cache

    @property
    def name(self) -> str:
        return "torchcodec"

    def _disk_cache_path(self, ref: VideoRef) -> Path:
        """Return the ``.npy`` path for a given frame reference (shared with PyAV)."""
        cache_dir = ref.video_path.parent / (ref.video_path.name + ".frame_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{ref.frame_index:06d}.npy"

    def _get_cache(self, video_path: Path) -> _TorchFrameCache:
        if video_path not in self._caches:
            self._caches[video_path] = _TorchFrameCache(
                video_path, max_cached=self._max_cached
            )
        return self._caches[video_path]

    def decode_frame(self, ref: VideoRef) -> NDArray:
        """Decode a single frame -> numpy HWC uint8.

        Checks the disk cache first; falls back to torchcodec decoding and
        saves the result to disk for future runs (same layout as PyAV).
        """
        if self._disk_cache:
            npy_path = self._disk_cache_path(ref)
            if npy_path.exists():
                return np.load(npy_path)

        # Decode from video
        cache = self._get_cache(ref.video_path)
        img = cache.get_frame(
            ref.frame_index, (ref.height, ref.width, ref.channels)
        )

        # Save to disk cache
        if self._disk_cache:
            np.save(self._disk_cache_path(ref), img)

        return img

    def close(self) -> None:
        """Close all decoder handles and clear the frame caches."""
        for cache in self._caches.values():
            cache.close()
        self._caches.clear()

    def __del__(self) -> None:
        self.close()

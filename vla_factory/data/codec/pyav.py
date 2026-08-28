"""PyAV-based video decoder — default VideoCodec implementation.

Uses PyAV for sequential decoding with an LRU-style frame cache.
Each video file gets its own ``_VideoFrameCache`` instance that keeps
the ``av.container.InputContainer`` open for fast sequential reads.

Disk cache: decoded frames are saved as ``.npy`` files next to the video.
On subsequent runs, frames are loaded from disk instead of re-decoding.
"""

from __future__ import annotations

import logging
import itertools
from collections import OrderedDict
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from numpy.typing import NDArray

from ..data_schema import VideoRef
from .registry import CodecRegistry

logger = logging.getLogger(__name__)


class _VideoFrameCache:
    """Per-video-file cache that keeps an ``av`` container open.

    Decodes frames sequentially from the video and caches recently
    decoded frames for fast re-reads.  Seek operations are minimised
    by tracking the current decode position.
    """

    def __init__(self, video_path: Path, max_cached: int = 32) -> None:
        self.video_path = video_path
        self.max_cached = max_cached
        self._container = None
        self._stream = None
        self._decoder = None
        self._current_pos = 0
        self._first_pts = 0
        self._cache: OrderedDict[int, NDArray] = OrderedDict()

    def _ensure_open(self) -> None:
        """Open the AV container lazily on first access."""
        if self._container is not None:
            return

        try:
            self._container = av.open(str(self.video_path))
        except Exception as e:
            self._container = None
            raise RuntimeError(f'Failed to open video {self.video_path}: {e}') from e
        self._stream = self._container.streams.video[0]
        self._stream.codec_context.skip_frame = "NONKEY"
        # Re-enable all frames after the initial seek setup
        self._stream.codec_context.skip_frame = "DEFAULT"
        self._decoder = self._container.decode(self._stream)
        self._current_pos = 0
        # Record the real first-frame PTS as the zero-point for frame-number
        # arithmetic (some streams have a non-zero start PTS). Consume the first
        # frame, then chain it back so _current_pos=0 still means "next() yields frame 0".
        try:
            first_frame = next(self._decoder)
            self._first_pts = first_frame.pts if first_frame.pts is not None else 0
            self._decoder = itertools.chain([first_frame], self._decoder)
        except StopIteration:
            pass

    def _frame_to_pts(self, frame_idx: int) -> int:
        """Convert a frame ordinal to the PTS (stream time_base units) of that frame.

        ``av`` seeks by timestamp in the stream's ``time_base`` (e.g. 1/15360),
        not by frame ordinal. Passing the bare frame index as the timestamp is
        orders of magnitude too small — it lands at (nearly) the start of the
        video, forcing the forward-skip loop in ``_seek_to`` to re-decode the
        whole prefix on every backward access (O(frame_idx) per seek). With
        time_base=1/N at F frames per second, each frame spans N/F time units.
        """
        rate = self._stream.average_rate
        if rate is not None:
            span = Fraction(self._stream.time_base.denominator) / rate
            # Floor division: never overshoots the target frame's true PTS, so
            # av's keyframe-aligned backward seek always lands at or before it.
            pts = (frame_idx * span.numerator) // span.denominator
            return pts + self._first_pts
        # Streams without an average rate: infer the per-frame duration from
        # the stream duration (in time_base units) and the frame count.
        duration = self._stream.duration
        n_frames = self._stream.frames
        if duration and n_frames:
            return int(frame_idx * duration / n_frames) + self._first_pts
        # Last resort: no timing info available — fall back to a frame-ordinal
        # seek (acceptable for timestamp-less streams where a PTS-based seek is
        # impossible anyway).
        return frame_idx + self._first_pts

    def _pts_per_frame(self) -> float:
        """Average number of time_base units spanned by one frame."""
        rate = self._stream.average_rate
        if rate is not None:
            return self._stream.time_base.denominator / float(rate)
        duration = self._stream.duration
        n_frames = self._stream.frames
        if duration and n_frames:
            return duration / n_frames
        return 1.0

    def _seek_to(self, frame_idx: int) -> None:
        """Seek to a target frame index.

        On return the decoder is positioned so the caller's next ``next()``
        yields frame ``frame_idx``.
        """
        self._ensure_open()
        # Flush existing decoder state
        if self._current_pos > frame_idx or self._current_pos == 0:
            # Seek backward or initial seek. av's seek is keyframe-aligned, so
            # it lands at some frame L <= target. Read the landing frame to
            # learn L, then skip only the remaining frames before the target —
            # skipping frame_idx frames from the landing position would overshoot
            # the target (and the video end).
            target_ts = self._frame_to_pts(frame_idx)
            self._container.seek(target_ts, stream=self._stream)
            self._decoder = self._container.decode(self._stream)
            self._current_pos = 0
            span = self._pts_per_frame()
            try:
                landing = next(self._decoder)
            except StopIteration:
                return
            if landing.pts is not None and span:
                # Round absorbs small PTS offsets from the nominal index*span.
                land = int(round((landing.pts - self._first_pts) / span))
            else:
                land = 0
            if land == frame_idx:
                # Keyframe-aligned seek landed exactly on the target. The
                # landing frame has already been consumed to learn its index,
                # so chain it back in front; otherwise the caller's next()
                # would yield frame_idx + 1 instead of frame_idx.
                self._decoder = itertools.chain([landing], self._decoder)
                self._current_pos = frame_idx
            else:
                self._current_pos = land + 1
                while self._current_pos < frame_idx:
                    try:
                        next(self._decoder)
                        self._current_pos += 1
                    except StopIteration:
                        break

    def get_frame(self, frame_idx: int, dims: tuple[int, ...]) -> NDArray:
        """Get a decoded frame as numpy HWC uint8.

        Uses cache if available; otherwise seeks + decodes.
        """
        # Check cache first
        if frame_idx in self._cache:
            self._cache.move_to_end(frame_idx)
            return self._cache[frame_idx]

        self._ensure_open()

        # Decide whether to seek or continue sequential
        if frame_idx < self._current_pos:
            self._seek_to(frame_idx)
        elif frame_idx > self._current_pos:
            # Decode forward to the target
            while self._current_pos < frame_idx:
                try:
                    next(self._decoder)
                    self._current_pos += 1
                except StopIteration:
                    # End of video, re-seek
                    self._seek_to(frame_idx)
                    break

        # Decode the target frame
        try:
            av_frame = next(self._decoder)
            self._current_pos += 1
        except StopIteration:
            # If decoder exhausted, seek back
            self._seek_to(frame_idx)
            try:
                av_frame = next(self._decoder)
                self._current_pos += 1
            except StopIteration:
                # Return black frame as fallback
                return np.zeros(dims, dtype=np.uint8)

        # Convert to numpy HWC uint8
        img = av_frame.to_ndarray(format="rgb24")
        h, w, c = dims
        if img.shape[0] != h or img.shape[1] != w:
            import cv2

            img = cv2.resize(img, (w, h))

        # Cache
        self._cache[frame_idx] = img
        if len(self._cache) > self.max_cached:
            self._cache.popitem(last=False)  # FIFO eviction

        return img

    def close(self) -> None:
        """Close the AV container."""
        if self._container is not None:
            self._container.close()
            self._container = None
            self._stream = None
            self._decoder = None
        self._cache.clear()

    def __del__(self) -> None:
        self.close()


@CodecRegistry.register("pyav")
class PyAVCodec:
    """Default video codec — uses PyAV to decode video frames to numpy.

    Maintains a per-video-file cache of ``_VideoFrameCache`` instances
    so that the same video container can be reused across frames.

    Disk cache: decoded frames are saved as ``.npy`` files under
    ``<video_path>.frame_cache/``.  On subsequent calls, frames are
    loaded directly from disk instead of re-decoding the video.
    """

    def __init__(self, max_cached_per_video: int = 32, disk_cache: bool = True) -> None:
        self._caches: dict[Path, _VideoFrameCache] = {}
        self._max_cached = max_cached_per_video
        self._disk_cache = disk_cache

    @property
    def name(self) -> str:
        return "pyav"

    def _disk_cache_path(self, ref: VideoRef) -> Path:
        """Return the ``.npy`` path for a given frame reference."""
        cache_dir = ref.video_path.parent / (ref.video_path.name + ".frame_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{ref.frame_index:06d}.npy"

    def _get_cache(self, video_path: Path) -> _VideoFrameCache:
        if video_path not in self._caches:
            self._caches[video_path] = _VideoFrameCache(
                video_path, max_cached=self._max_cached
            )
        return self._caches[video_path]

    def decode_frame(self, ref: VideoRef) -> NDArray:
        """Decode a single frame -> numpy HWC uint8.

        Checks disk cache first; falls back to video decoding and
        saves the result to disk for future runs.
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

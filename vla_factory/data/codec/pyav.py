"""PyAV-based video decoder — default VideoCodec implementation.

Uses PyAV for random-access decoding with an LRU-style frame cache.
Each video file gets its own ``_VideoSession`` instance that keeps
the ``av.container.InputContainer`` open for fast random access.
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
from .base import OpenHandleLRU
from .registry import CodecRegistry

logger = logging.getLogger(__name__)

# Decoding forward is only cheaper than a keyframe-aligned seek when the gap
# is smaller than the keyframe interval (a seek re-decodes from the nearest
# keyframe before the target). Gaps beyond this many frames re-seek instead:
# uniform random access used to turn far-forward jumps into O(gap) sequential
# decodes (~1.4 s/frame on 640x480 H.264 robot videos, 118x slower than
# torchcodec).
_MAX_DECODE_FORWARD_GAP = 12


class _VideoSession:
    """Per-video-file decoding session holding an ``av`` container open.

    Serves exact frames by frame ordinal: keyframe-aligned seeks plus the
    minimum forward decode, with recently decoded frames cached for fast
    re-reads. Seek operations are minimised by tracking the current
    decode position.
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
        yields frame ``frame_idx``. Always issues a fresh container seek:
        av's keyframe-aligned seek lands at some frame L <= target whether
        we approach from before or after it, so the same path serves
        backward access and far-forward jumps.
        """
        self._ensure_open()
        # Read the landing frame to learn its index, then skip only the
        # remaining frames before the target — skipping frame_idx frames from
        # the landing position would overshoot the target (and the video end).
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
            # Hand out a copy: the cached array is shared across all future
            # reads, so callers must never be able to mutate it in place.
            return self._cache[frame_idx].copy()

        self._ensure_open()

        # Backward access, or a forward jump longer than the decode-forward
        # window, takes the keyframe-aligned seek; only near-forward access
        # decodes through (a seek re-decodes from the nearest keyframe, so
        # decode-forward is cheaper only inside one keyframe interval).
        if (
            frame_idx < self._current_pos
            or frame_idx > self._current_pos + _MAX_DECODE_FORWARD_GAP
        ):
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
        # Store a private copy and return the freshly decoded array: callers
        # own their result and cannot corrupt the LRU by mutating it.
        self._cache[frame_idx] = img.copy()
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

    Maintains a per-video-file cache of ``_VideoSession`` instances
    so that the same video container can be reused across frames. The
    open set is bounded by a file-level LRU (``max_open_videos``):
    every entry holds an fd plus an H.264 decoder context, so an
    unbounded registry eventually exhausts fds.
    """

    def __init__(
        self,
        max_cached_per_video: int = 32,
        max_open_videos: int = 32,
    ) -> None:
        self._open_handles: OpenHandleLRU[Path, _VideoSession] = OpenHandleLRU(
            max_open_videos
        )
        self._max_cached = max_cached_per_video

    @property
    def name(self) -> str:
        return "pyav"

    def _session_for(self, video_path: Path) -> _VideoSession:
        session = self._open_handles.get(video_path)
        if session is None:
            session = _VideoSession(video_path, max_cached=self._max_cached)
            self._open_handles.put(video_path, session)
        return session

    def decode_frame(self, ref: VideoRef) -> NDArray:
        """Decode a single frame -> numpy HWC uint8."""
        session = self._session_for(ref.video_path)
        return session.get_frame(
            ref.frame_index, (ref.height, ref.width, ref.channels)
        )

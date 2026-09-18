"""PyAV-based video decoder — default VideoCodec implementation.

Uses PyAV for random-access decoding with an LRU-style frame cache.
Each video file keeps its container and decoder state in the shared cache.
This keeps the ``av.container.InputContainer`` open for fast random access.
"""

from __future__ import annotations

import logging
import itertools
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from numpy.typing import NDArray

from ..data_schema import VideoRef
from .base import CachedVideoCodec
from .registry import CodecRegistry

logger = logging.getLogger(__name__)

# Decoding forward is only cheaper than a keyframe-aligned seek when the gap
# is smaller than the keyframe interval (a seek re-decodes from the nearest
# keyframe before the target). Gaps beyond this many frames re-seek instead:
# uniform random access used to turn far-forward jumps into O(gap) sequential
# decodes (~1.4 s/frame on 640x480 H.264 robot videos, 118x slower than
# torchcodec).
_MAX_DECODE_FORWARD_GAP = 12


@CodecRegistry.register("pyav")
class PyAVCodec(CachedVideoCodec):
    """Default video codec — uses PyAV to decode video frames to numpy.

    Maintains a per-video-file cache of open video containers. The
    open set is bounded by a file-level LRU (``max_open_videos``):
    every entry holds an fd plus an H.264 decoder context, so an
    unbounded registry eventually exhausts fds.
    """

    def __init__(
        self,
        max_cached_per_video: int = 32,
        max_open_videos: int = 32,
    ) -> None:
        super().__init__(max_cached_per_video, max_open_videos)

    @property
    def name(self) -> str:
        return "pyav"

    def _open_decoder(self, video_path: Path) -> dict:
        return {"container": None, "stream": None, "decoder": None, "position": 0, "first_pts": 0}

    def _ensure_open(self, decoder: dict) -> None:
        if decoder["container"] is not None:
            return
        try:
            decoder["container"] = av.open(str(decoder["video_path"]))
        except Exception as exc:
            raise RuntimeError(f'Failed to open video {decoder["video_path"]}: {exc}') from exc
        decoder["stream"] = decoder["container"].streams.video[0]
        decoder["decoder"] = decoder["container"].decode(decoder["stream"])
        try:
            first = next(decoder["decoder"])
            decoder["first_pts"] = first.pts or 0
            decoder["decoder"] = itertools.chain([first], decoder["decoder"])
        except StopIteration:
            pass

    def _frame_to_pts(self, decoder: dict, frame_idx: int) -> int:
        stream = decoder["stream"]
        if stream.average_rate is not None:
            span = Fraction(stream.time_base.denominator) / stream.average_rate
            return (frame_idx * span.numerator) // span.denominator + decoder["first_pts"]
        if stream.duration and stream.frames:
            return int(frame_idx * stream.duration / stream.frames) + decoder["first_pts"]
        return frame_idx + decoder["first_pts"]

    def _pts_per_frame(self, decoder: dict) -> float:
        stream = decoder["stream"]
        if stream.average_rate is not None:
            return stream.time_base.denominator / float(stream.average_rate)
        if stream.duration and stream.frames:
            return stream.duration / stream.frames
        return 1.0

    def _seek_to(self, decoder: dict, frame_idx: int) -> None:
        self._ensure_open(decoder)
        decoder["container"].seek(self._frame_to_pts(decoder, frame_idx), stream=decoder["stream"])
        decoder["decoder"] = decoder["container"].decode(decoder["stream"])
        decoder["position"] = 0
        try:
            landing = next(decoder["decoder"])
        except StopIteration:
            return
        span = self._pts_per_frame(decoder)
        land = int(round((landing.pts - decoder["first_pts"]) / span)) if landing.pts is not None and span else 0
        if land == frame_idx:
            decoder["decoder"] = itertools.chain([landing], decoder["decoder"])
            decoder["position"] = frame_idx
        else:
            decoder["position"] = land + 1
            while decoder["position"] < frame_idx:
                try:
                    next(decoder["decoder"])
                    decoder["position"] += 1
                except StopIteration:
                    break

    def _decode_frame(self, decoder: dict, ref: VideoRef) -> NDArray:
        frame_idx = ref.frame_index
        self._ensure_open(decoder)
        if frame_idx < decoder["position"] or frame_idx > decoder["position"] + _MAX_DECODE_FORWARD_GAP:
            self._seek_to(decoder, frame_idx)
        while decoder["position"] < frame_idx:
            try:
                next(decoder["decoder"])
                decoder["position"] += 1
            except StopIteration:
                self._seek_to(decoder, frame_idx)
                break
        try:
            frame = next(decoder["decoder"])
            decoder["position"] += 1
        except StopIteration:
            self._seek_to(decoder, frame_idx)
            try:
                frame = next(decoder["decoder"])
                decoder["position"] += 1
            except StopIteration:
                return np.zeros((ref.height, ref.width, ref.channels), dtype=np.uint8)
        image = frame.to_ndarray(format="rgb24")
        if image.shape[:2] != (ref.height, ref.width):
            import cv2
            image = cv2.resize(image, (ref.width, ref.height))
        return image

    def _close_decoder(self, decoder: dict) -> None:
        if decoder["container"] is not None:
            decoder["container"].close()
            decoder["container"] = decoder["stream"] = decoder["decoder"] = None
        super()._close_decoder(decoder)

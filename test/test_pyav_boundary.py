"""Boundary tests for PyAVCodec seek / fallback / caching paths.

These tests cover branches that the main test_data_pipeline suite does not
reach:

* ``_frame_to_pts`` fallbacks when ``average_rate`` / duration / frame count
  are unavailable
* ``_seek_to`` must position the decoder so the next frame is exactly the
  target, even when the seek lands on the target keyframe (which the real
  LeRobot test video can mask because adjacent frames are duplicated)
* out-of-range frame indices return a black frame instead of raising
* resizing on decode when the caller requests different dimensions
* disk-cache round-trip
* opening a missing video raises RuntimeError
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import av
import numpy as np
import pytest

from vla_factory.data.codec.pyav import PyAVCodec, _VideoFrameCache
from vla_factory.data.data_schema import VideoRef

DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "test"
    / "data"
    / "lerobot_train_data_3_episodes"
)
VIDEO_PATH = (
    DATASET_PATH
    / "videos"
    / "observation.images.front"
    / "chunk-000"
    / "file-000.mp4"
)
IMAGE_H = 480
IMAGE_W = 640

pytestmark = pytest.mark.skipif(
    not VIDEO_PATH.exists(), reason="LeRobot test video not found"
)


def _ref(video_path: Path, frame_index: int, height: int = IMAGE_H, width: int = IMAGE_W) -> VideoRef:
    return VideoRef(
        video_path=video_path,
        frame_index=frame_index,
        height=height,
        width=width,
        channels=3,
    )


def _write_distinct_mp4(
    path: Path, n_frames: int = 5, h: int = 32, w: int = 32, *, pts_offset: int = 0
) -> None:
    """Write an mp4 whose adjacent frames are all different solid colors.

    When ``pts_offset`` is non-zero, each frame's PTS is set to
    ``pts_offset + i`` in stream time units, producing a stream whose
    first-frame PTS is non-zero. This tests the ``_first_pts`` zero-point
    arithmetic in ``_seek_to``.
    """
    container = av.open(str(path), "w")
    try:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        for i in range(n_frames):
            g = 20 * (i + 1)  # distinct per frame, far enough from black
            frame = av.VideoFrame.from_ndarray(
                np.full((h, w, 3), g, dtype=np.uint8), format="rgb24"
            )
            if pts_offset:
                frame.pts = pts_offset + i
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def _landing_index(stream, frame) -> int:
    span = stream.time_base.denominator / float(stream.average_rate)
    return int(round(frame.pts / span)) if frame.pts is not None else -1


# ── _frame_to_pts / _pts_per_frame fallbacks ─────────────────────────


class _DummyTimeBase:
    def __init__(self, denominator: int):
        self.denominator = denominator


class _DummyStream:
    def __init__(self, average_rate, denominator: int, duration, frames):
        self.average_rate = average_rate
        self.time_base = _DummyTimeBase(denominator)
        self.duration = duration
        self.frames = frames


def test_frame_to_pts_uses_stream_timebase():
    codec = PyAVCodec(disk_cache=False)
    cache = codec._get_cache(VIDEO_PATH)
    try:
        cache._ensure_open()
        # Real LeRobot test video: 1/15360 time_base at 30 fps => 512 units/frame.
        assert cache._stream.time_base.denominator == 15360
        assert float(cache._stream.average_rate) == 30.0
        assert cache._frame_to_pts(1100) == 1100 * 512
        assert cache._pts_per_frame() == 512.0
    finally:
        cache.close()


def test_frame_to_pts_fallback_duration_and_frames():
    cache = _VideoFrameCache(Path("dummy.mp4"))
    cache._stream = _DummyStream(
        average_rate=None, denominator=1000, duration=1000, frames=100
    )
    assert cache._frame_to_pts(50) == 500
    assert cache._pts_per_frame() == 10.0


def test_frame_to_pts_last_resort_frame_ordinal():
    cache = _VideoFrameCache(Path("dummy.mp4"))
    cache._stream = _DummyStream(
        average_rate=None, denominator=1000, duration=0, frames=0
    )
    assert cache._frame_to_pts(50) == 50
    assert cache._pts_per_frame() == 1.0


# ── _seek_to exact positioning ───────────────────────────────────────


def test_seek_to_positions_decoder_at_exact_target():
    """_seek_to must leave the decoder so the next frame is the target.

    This catches the off-by-one where a keyframe-aligned seek lands exactly on
    the target keyframe, the landing frame is consumed to learn its index, and
    the caller then gets target+1 instead of target.
    """
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "distinct.mp4"
        _write_distinct_mp4(path)

        codec = PyAVCodec(disk_cache=False)
        cache = codec._get_cache(path)
        try:
            cache._ensure_open()
            cache._seek_to(2)
            frame = next(cache._decoder)
            assert _landing_index(cache._stream, frame) == 2
            assert cache._current_pos == 2  # _seek_to left the next frame at target
        finally:
            cache.close()


def test_backward_access_exact_frame_with_distinct_frames():
    """Backward access after LRU eviction must return the exact target frame."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "distinct.mp4"
        _write_distinct_mp4(path)

        codec = PyAVCodec(disk_cache=False, max_cached_per_video=2)
        target = 2
        expected = codec.decode_frame(_ref(path, target))
        codec.decode_frame(_ref(path, 3))  # evicts target from the 2-entry LRU
        codec.decode_frame(_ref(path, 4))

        img = codec.decode_frame(_ref(path, target))  # backward -> _seek_to
        assert np.array_equal(img, expected)
        # Guard against the old off-by-one: returned frame 3 is not acceptable.
        wrong = codec.decode_frame(_ref(path, target + 1))
        assert not np.array_equal(img, wrong)


# ── decode_frame fallbacks ───────────────────────────────────────────


def test_backward_access_exact_frame_with_pts_offset():
    """Backward access must return the exact target frame even when the
    stream's first-frame PTS is non-zero (the _first_pts zero-point fix)."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "offset.mp4"
        _write_distinct_mp4(path, pts_offset=1)

        codec = PyAVCodec(disk_cache=False, max_cached_per_video=2)
        target = 2
        expected = codec.decode_frame(_ref(path, target))
        codec.decode_frame(_ref(path, 3))  # evicts target from the 2-entry LRU
        codec.decode_frame(_ref(path, 4))

        img = codec.decode_frame(_ref(path, target))  # backward -> _seek_to
        assert np.array_equal(img, expected)
        wrong = codec.decode_frame(_ref(path, target + 1))
        assert not np.array_equal(img, wrong)





def test_decode_beyond_end_returns_black_frame():
    codec = PyAVCodec(disk_cache=False)
    container = av.open(str(VIDEO_PATH))
    total = container.streams.video[0].frames
    container.close()

    img = codec.decode_frame(_ref(VIDEO_PATH, total))
    assert img.shape == (IMAGE_H, IMAGE_W, 3)
    assert img.dtype == np.uint8
    assert np.all(img == 0)


def test_decode_resizes_when_dims_differ():
    codec = PyAVCodec(disk_cache=False)
    img = codec.decode_frame(_ref(VIDEO_PATH, 0, height=24, width=32))
    assert img.shape == (24, 32, 3)


def test_decode_missing_video_raises_runtime_error():
    codec = PyAVCodec(disk_cache=False)
    with pytest.raises(RuntimeError, match="Failed to open video"):
        codec.decode_frame(_ref(Path("/nonexistent/video.mp4"), 0))


def test_disk_cache_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "distinct.mp4"
        _write_distinct_mp4(path)

        ref = _ref(path, 1, height=32, width=32)
        codec = PyAVCodec(disk_cache=True)
        first = codec.decode_frame(ref)
        npy = path.parent / (path.name + ".frame_cache") / "000001.npy"
        assert npy.exists()

        # A fresh codec must serve the frame from the .npy disk cache.
        codec2 = PyAVCodec(disk_cache=True)
        second = codec2.decode_frame(ref)
        assert np.array_equal(first, second)

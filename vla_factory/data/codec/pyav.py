"""PyAV-based video decoder — default VideoCodec implementation.

Uses PyAV for sequential decoding with an LRU-style frame cache.
Each video file gets its own ``_VideoFrameCache`` instance that keeps
the ``av.container.InputContainer`` open for fast sequential reads.

Disk cache: decoded frames are saved as ``.npy`` files next to the video.
On subsequent runs, frames are loaded from disk instead of re-decoding.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import av
import numpy as np
from numpy.typing import NDArray

from ..formats.base import VideoRef

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
        self._cache: OrderedDict[int, NDArray] = OrderedDict()

    def _ensure_open(self) -> None:
        """Open the AV container lazily on first access."""
        if self._container is not None:
            return

        self._container = av.open(str(self.video_path))
        self._stream = self._container.streams.video[0]
        self._stream.codec_context.skip_frame = "NONKEY"
        # Re-enable all frames after the initial seek setup
        self._stream.codec_context.skip_frame = "DEFAULT"
        self._decoder = self._container.decode(self._stream)
        self._current_pos = 0

    def _seek_to(self, frame_idx: int) -> None:
        """Seek to a target frame index."""
        self._ensure_open()
        # Flush existing decoder state
        if self._current_pos > frame_idx or self._current_pos == 0:
            # Seek backward or initial seek
            target_ts = frame_idx
            self._container.seek(target_ts, stream=self._stream)
            self._decoder = self._container.decode(self._stream)
            self._current_pos = 0
            # Skip frames up to target
            while self._current_pos < frame_idx:
                try:
                    frame = next(self._decoder)
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


# ---------------------------------------------------------------------------
# Video pre-processing: MP4 → .npy disk cache
# ---------------------------------------------------------------------------

def preprocess_video(video_path: Path, *, pbar=None) -> int:
    """Decode all frames of a video file to ``.npy`` disk cache.

    Output layout: ``<video_path>.frame_cache/000000.npy, 000001.npy, ...``

    If the cache already exists and is complete (frame count matches), it is
    skipped.  Individual frames that already exist on disk are not re-decoded.

    Parameters
    ----------
    pbar : tqdm.tqdm, optional
        Outer progress bar (per-video).  Updated with the video filename
        and advanced by 1 after this video finishes.

    Returns
    -------
    int
        Number of frames extracted (total, including previously cached).
    """
    if pbar is not None:
        pbar.set_postfix_str(video_path.name, refresh=False)

    cache_dir = video_path.parent / (video_path.name + ".frame_cache")

    # Fast path: if cache already complete, skip entirely
    if cache_dir.exists():
        existing = len(list(cache_dir.glob("*.npy")))
        if existing > 0:
            container = av.open(str(video_path))
            total = container.streams.video[0].frames
            container.close()
            if total > 0 and existing >= total:
                logger.info("Cache complete for %s (%d frames), skipping",
                            video_path.name, total)
                if pbar is not None:
                    pbar.update(1)
                return existing

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    cache_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for frame_idx, av_frame in enumerate(container.decode(stream)):
        npy_path = cache_dir / f"{frame_idx:06d}.npy"
        if not npy_path.exists():
            img = av_frame.to_ndarray(format="rgb24")
            np.save(npy_path, img)
        count += 1

    container.close()
    logger.info("Preprocessed %s: %d frames -> %s",
                video_path.name, count, cache_dir)
    if pbar is not None:
        pbar.update(1)
    return count


def preprocess_dataset(dataset_path: Path) -> None:
    """Scan a dataset directory for MP4 files and pre-decode all frames.

    This fills the ``.frame_cache/`` directories that
    :meth:`PyAVCodec.decode_frame` checks before falling back to PyAV
    decoding, so subsequent training runs never need to touch the video
    decoder.
    """
    from tqdm import tqdm

    mp4_files = sorted(dataset_path.rglob("*.mp4"))
    if not mp4_files:
        logger.warning("No MP4 files found in %s", dataset_path)
        return

    logger.info("Found %d video files in %s", len(mp4_files), dataset_path)
    total_frames = 0
    with tqdm(mp4_files, desc="Preprocessing", unit="video",
              dynamic_ncols=True) as pbar:
        for mp4 in mp4_files:
            n = preprocess_video(mp4, pbar=pbar)
            total_frames += n
    logger.info("Preprocessing complete: %d videos, %d total frames",
                len(mp4_files), total_frames)

"""hdf5-embedded JPEG codec — decodes RoboTwin camera frames.

RoboTwin stores every camera's frames as JPEG byte streams *inside* the
episode hdf5 file (``/observation/{camera}/rgb`` — one encoded frame per
timestep), rather than as separate MP4 files like LeRobot. This codec adapts
that layout to the :class:`VideoCodec` contract: a :class:`VideoRef` whose
``video_path`` points at the ``.hdf5`` file and whose ``stream`` names the
camera; ``frame_index`` selects the timestep.

Caching mirrors :class:`PyAVCodec`'s in-memory LRU: one open ``h5py.File``
handle per hdf5 path plus a bounded LRU of recently decoded frames
(``max_cached_per_video``). Because one hdf5 file holds every camera's JPEG
streams, the LRU key is ``(stream, frame_index)`` — not just ``frame_index``
as for MP4, where each file carries a single camera.

``h5py`` is an optional (``[robotwin]``) dependency: it is imported lazily on
first decode so that ``resolve_codec`` and the codec registry stay importable
without it (rationale: framework-wide "optional deps defer to call time").
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


def _load_h5py() -> Any:
    """Import ``h5py`` lazily with an actionable error if it is missing."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Reading RoboTwin hdf5 datasets requires 'h5py'. Install the "
            "RoboTwin extra: pip install -e \".[robotwin]\" (or pip install h5py)."
        ) from exc
    return h5py


class _Hdf5FrameCache:
    """Per-hdf5-file cache: one open handle plus an LRU of decoded frames.

    Mirrors PyAV's ``_VideoFrameCache`` and torchcodec's ``_TorchFrameCache``:
    keeps a single ``h5py.File`` handle open (files are read many times — once
    per frame per camera — so re-opening each call would be wasteful) and an
    ``OrderedDict`` LRU of decoded frames. The LRU key is ``(stream,
    frame_index)``: the same index in different cameras of one hdf5 file
    decodes to different pixels, so the camera must be part of the key.
    """

    def __init__(self, path: Path, rgb_key_template: str, max_cached: int = 32) -> None:
        self.path = path
        self._rgb_key_template = rgb_key_template
        self.max_cached = max_cached
        self._handle: Any = None
        self._cache: OrderedDict[tuple[str, int], NDArray] = OrderedDict()

    def _ensure_open(self) -> Any:
        """Open the h5py handle lazily on first access."""
        if self._handle is None:
            h5py = _load_h5py()
            self._handle = h5py.File(str(self.path), "r")
        return self._handle

    def get_frame(self, ref: VideoRef) -> NDArray:
        """Decode one frame -> numpy HWC uint8 RGB, LRU-cached."""
        key = (ref.stream, ref.frame_index)
        if key in self._cache:
            self._cache.move_to_end(key)
            # Hand out a copy: the cached array is shared across all future
            # reads, so callers must never be able to mutate it in place.
            return self._cache[key].copy()

        f = self._ensure_open()
        ds_key = self._rgb_key_template.format(stream=ref.stream)
        try:
            raw = f[ds_key][ref.frame_index]
        except KeyError as exc:
            raise KeyError(
                f"Camera stream '{ds_key}' not found in {ref.video_path}. "
                f"Available: {list(f.get('observation', {}).keys())}"
            ) from exc

        buf = np.frombuffer(bytes(raw), dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(
                f"Failed to JPEG-decode frame {ref.frame_index} of "
                f"'{ref.stream}' in {ref.video_path}."
            )
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if img.shape[0] != ref.height or img.shape[1] != ref.width:
            img = cv2.resize(img, (ref.width, ref.height))

        # Cache (FIFO eviction on overflow, mirroring PyAV's in-memory LRU).
        # Store a private copy and return the freshly decoded array: callers
        # own their result and cannot corrupt the LRU by mutating it.
        self._cache[key] = img.copy()
        if len(self._cache) > self.max_cached:
            self._cache.popitem(last=False)
        return img

    def close(self) -> None:
        """Close the h5py handle and clear the frame LRU."""
        if self._handle is not None:
            try:
                self._handle.close()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
            self._handle = None
        self._cache.clear()

    def __del__(self) -> None:
        self.close()


@CodecRegistry.register("hdf5_jpeg")
class Hdf5JpegCodec:
    """Decode JPEG frames stored inside RoboTwin episode hdf5 files.

    Caching mirrors :class:`PyAVCodec`: a per-hdf5-file cache of open handles
    plus a decoded-frame LRU (``max_cached_per_video``, default 32).
    """

    def __init__(
        self,
        rgb_key_template: str = "/observation/{stream}/rgb",
        max_cached_per_video: int = 32,
    ) -> None:
        self._rgb_key_template = rgb_key_template
        self._max_cached = max_cached_per_video
        self._caches: dict[Path, _Hdf5FrameCache] = {}

    @property
    def name(self) -> str:
        return "hdf5_jpeg"

    def _get_cache(self, path: Path) -> _Hdf5FrameCache:
        if path not in self._caches:
            self._caches[path] = _Hdf5FrameCache(
                path, self._rgb_key_template, max_cached=self._max_cached
            )
        return self._caches[path]

    def decode_frame(self, ref: VideoRef) -> NDArray:
        """Decode one frame -> numpy HWC uint8 RGB.

        Reads the JPEG byte stream at ``/observation/{ref.stream}/rgb`` for
        ``ref.frame_index`` and decodes it. Resizes to ``(ref.height,
        ref.width)`` when the stored frame differs, matching the codec
        contract used by the rest of the pipeline.
        """
        if ref.stream is None:
            raise ValueError(
                "Hdf5JpegCodec requires VideoRef.stream (the camera name); got "
                f"None for {ref.video_path}. The RoboTwin reader must set it."
            )
        return self._get_cache(ref.video_path).get_frame(ref)

    def close(self) -> None:
        """Close all open hdf5 handles and clear the frame LRUs."""
        for cache in self._caches.values():
            cache.close()
        self._caches.clear()

    def __del__(self) -> None:
        self.close()

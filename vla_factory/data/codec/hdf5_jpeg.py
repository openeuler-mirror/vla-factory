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
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from ..data_schema import VideoRef
from .base import CachedVideoCodec
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


@CodecRegistry.register("hdf5_jpeg")
class Hdf5JpegCodec(CachedVideoCodec):
    """Decode JPEG frames stored inside RoboTwin episode hdf5 files.

    Caching mirrors :class:`PyAVCodec`: a per-hdf5-file cache of open handles
    plus a decoded-frame LRU (``max_cached_per_video``, default 32). The
    open set is bounded by a file-level LRU (``max_open_videos``): every
    entry holds an open h5py handle (an fd), so an unbounded registry
    eventually exhausts fds.
    """

    def __init__(
        self,
        rgb_key_template: str = "/observation/{stream}/rgb",
        max_cached_per_video: int = 32,
        max_open_videos: int = 32,
    ) -> None:
        super().__init__(max_cached_per_video, max_open_videos)
        self._rgb_key_template = rgb_key_template

    @property
    def name(self) -> str:
        return "hdf5_jpeg"

    def _open_decoder(self, video_path: Path) -> dict[str, Any]:
        return {"handle": None}

    def _cache_key(self, ref: VideoRef) -> tuple[str | None, int]:
        return ref.stream, ref.frame_index

    def _decode_frame(self, decoder: dict[str, Any], ref: VideoRef) -> NDArray:
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
        if decoder["handle"] is None:
            decoder["handle"] = _load_h5py().File(str(ref.video_path), "r")
        f = decoder["handle"]
        ds_key = self._rgb_key_template.format(stream=ref.stream)
        try:
            raw = f[ds_key][ref.frame_index]
        except KeyError as exc:
            raise KeyError(
                f"Camera stream '{ds_key}' not found in {ref.video_path}. "
                f"Available: {list(f.get('observation', {}).keys())}"
            ) from exc

        bgr = cv2.imdecode(np.frombuffer(bytes(raw), dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(
                f"Failed to JPEG-decode frame {ref.frame_index} of "
                f"'{ref.stream}' in {ref.video_path}."
            )
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if image.shape[:2] != (ref.height, ref.width):
            image = cv2.resize(image, (ref.width, ref.height))
        return image

    def _close_decoder(self, decoder: dict[str, Any]) -> None:
        if decoder["handle"] is not None:
            try:
                decoder["handle"].close()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
            decoder["handle"] = None
        super()._close_decoder(decoder)

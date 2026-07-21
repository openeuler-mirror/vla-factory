"""hdf5-embedded JPEG codec — decodes RoboTwin camera frames.

RoboTwin stores every camera's frames as JPEG byte streams *inside* the
episode hdf5 file (``/observation/{camera}/rgb`` — one encoded frame per
timestep), rather than as separate MP4 files like LeRobot. This codec adapts
that layout to the :class:`VideoCodec` contract: a :class:`VideoRef` whose
``video_path`` points at the ``.hdf5`` file and whose ``stream`` names the
camera; ``frame_index`` selects the timestep.

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

from ..formats.base import VideoRef

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


class Hdf5JpegCodec:
    """Decode JPEG frames stored inside RoboTwin episode hdf5 files.

    Keeps one open ``h5py.File`` handle per hdf5 path (files are read many
    times — once per frame per camera — so re-opening each call would be
    wasteful), mirroring :class:`PyAVCodec`'s per-video container cache.
    """

    def __init__(self, rgb_key_template: str = "/observation/{stream}/rgb") -> None:
        self._rgb_key_template = rgb_key_template
        self._files: dict[Path, Any] = {}

    @property
    def name(self) -> str:
        return "hdf5_jpeg"

    def _get_file(self, path: Path) -> Any:
        handle = self._files.get(path)
        if handle is None:
            h5py = _load_h5py()
            handle = h5py.File(str(path), "r")
            self._files[path] = handle
        return handle

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
        f = self._get_file(ref.video_path)
        key = self._rgb_key_template.format(stream=ref.stream)
        try:
            raw = f[key][ref.frame_index]
        except KeyError as exc:
            raise KeyError(
                f"Camera stream '{key}' not found in {ref.video_path}. "
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
        return img

    def close(self) -> None:
        """Close all open hdf5 handles."""
        for handle in self._files.values():
            try:
                handle.close()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
        self._files.clear()

    def __del__(self) -> None:
        self.close()

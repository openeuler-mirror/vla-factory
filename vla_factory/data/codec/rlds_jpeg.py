"""RLDS embedded-image codec — decodes frames carried inside TFRecord examples.

RLDS stores every camera's frames as JPEG/PNG-encoded bytes *inside* the
``tf.train.Example`` proto of each episode record (one encoded image per
timestep under ``steps/observation/<camera>``), so there is no video file to
point at. The :class:`RldsReader` parses an episode record once and hands each
frame to this codec as an :class:`ImageRef` whose ``payload`` holds the encoded
bytes; ``frame_index`` is only informational (step within the episode).

Caching follows :class:`Hdf5JpegCodec`'s pattern, but the key is the payload
content itself: one TFRecord shard holds many episodes, so ``(path,
frame_index)`` would collide across episodes, while identical encoded bytes
always decode to the same image and can safely share one cache entry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from ..data_schema import ImageRef, ImageStorage
from .base import CachedImageDecoder
from .registry import CodecRegistry

@CodecRegistry.register("rlds_jpeg")
class RldsJpegCodec(CachedImageDecoder):
    """Decode JPEG/PNG frames embedded in RLDS TFRecord examples.

    The decoder is stateless (no file handles — the bytes arrive on the
    ref), so ``_open_decoder`` only allocates the shared frame LRU. The LRU
    key is the encoded payload: content-addressed, immune to the
    cross-episode ``(path, frame_index)`` collisions a shard layout would
    cause.
    """

    supported_storages = frozenset({ImageStorage.ENCODED_BYTES})

    @property
    def name(self) -> str:
        return "rlds_jpeg"

    def _open_decoder(self, video_path: Path) -> dict[str, Any]:
        return {}

    def _cache_key(self, ref: ImageRef) -> Any:
        if ref.payload is None:
            # Fall back to positional keying for refs without payload; the
            # collision risk is theoretical (this codec always receives
            # payload refs from the RLDS reader).
            return (ref.stream, ref.frame_index)
        return ref.payload

    def _decode_frame(self, decoder: dict[str, Any], ref: ImageRef) -> NDArray:
        """Decode one frame -> numpy HWC uint8 RGB."""
        if ref.payload is None:
            raise ValueError(
                "RldsJpegCodec requires ImageRef.payload (the encoded frame "
                f"bytes); got None for {ref.path}. The RLDS reader must "
                "set it."
            )
        buf = np.frombuffer(ref.payload, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(
                f"Failed to decode frame {ref.frame_index} of "
                f"'{ref.stream}' in {ref.path}. The payload is not "
                "valid JPEG/PNG data."
            )
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if image.shape[:2] != (ref.height, ref.width):
            image = cv2.resize(image, (ref.width, ref.height))
        return image

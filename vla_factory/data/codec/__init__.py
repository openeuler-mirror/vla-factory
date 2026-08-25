"""Video codec registration and selection."""

from __future__ import annotations

from .base import VideoCodec
from .registry import CodecRegistry

# Built-ins register on import. HDF5 keeps h5py lazy until its first decode.
from .hdf5_jpeg import Hdf5JpegCodec
from .pyav import PyAVCodec

# torchcodec is optional - import the module to register it, but tolerate
# missing torch/torchcodec (the codec raises only on first decode).
try:
    from . import torchcodec  # noqa: F401 - registers "torchcodec" in the registry
except Exception:
    pass

# Probe cache for torchcodec availability. ``None`` = not probed yet,
# ``True``/``False`` = import probe result (including ABI compatibility).
_TORCHCODEC_AVAILABLE: bool | None = None


def resolve_codec(name: str = "auto", format: str | None = None) -> VideoCodec:
    """Construct a registered codec.

    ``auto`` is format-aware:
      - lerobot-v3  -> torchcodec when importable (ABI-safe), else PyAV
      - robotwin    -> hdf5_jpeg
      - other/None  -> PyAV stable default

    Explicit ``name`` always overrides format-based defaults.
    """
    global _TORCHCODEC_AVAILABLE
    if name.lower() != "auto":
        return CodecRegistry.create(name)

    fmt = (format or "").lower()
    if fmt in ("lerobot-v3", "lerobot_v3", "lerobot"):
        if _TORCHCODEC_AVAILABLE is None:
            try:
                from .torchcodec import _load_torchcodec

                _load_torchcodec()
                _TORCHCODEC_AVAILABLE = True
            except Exception:
                _TORCHCODEC_AVAILABLE = False
        if _TORCHCODEC_AVAILABLE:
            return CodecRegistry.create("torchcodec")
        return CodecRegistry.create("pyav")
    if fmt in ("robotwin",):
        return CodecRegistry.create("hdf5_jpeg")
    return CodecRegistry.create("pyav")


__all__ = [
    "CodecRegistry",
    "Hdf5JpegCodec",
    "PyAVCodec",
    "VideoCodec",
    "resolve_codec",
]

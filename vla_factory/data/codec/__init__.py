"""Video codec registration and selection."""

from __future__ import annotations

from .base import VideoCodec
from .registry import CodecRegistry

# Built-ins register on import. HDF5 keeps h5py lazy until its first decode.
from .hdf5_jpeg import Hdf5JpegCodec
from .pyav import PyAVCodec


def resolve_codec(name: str = "auto") -> VideoCodec:
    """Construct a registered codec; ``auto`` selects the stable PyAV default."""
    return CodecRegistry.create("pyav" if name.lower() == "auto" else name)


__all__ = [
    "CodecRegistry",
    "Hdf5JpegCodec",
    "PyAVCodec",
    "VideoCodec",
    "resolve_codec",
]

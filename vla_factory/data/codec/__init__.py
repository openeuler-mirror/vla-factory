"""Video codec registry and resolver.

Public API:
    resolve_codec(name)   -> VideoCodec
"""

from __future__ import annotations

import logging

from .base import VideoCodec
from .pyav import PyAVCodec

logger = logging.getLogger(__name__)

__all__ = ["PyAVCodec", "VideoCodec", "resolve_codec"]


def resolve_codec(name: str = "auto") -> VideoCodec:
    """Instantiate a ``VideoCodec`` by name.

    Parameters
    ----------
    name : str
        One of ``"auto"``, ``"pyav"``, or ``"torchcodec"``.
        ``"auto"`` defaults to PyAV.
    """
    if name in ("auto", "pyav"):
        return PyAVCodec()
    if name == "hdf5_jpeg":
        # Deferred import: keeps the registry importable without h5py (the
        # RoboTwin extra). Constructing the codec does not need h5py; only
        # decoding does (h5py is imported there).
        from .hdf5_jpeg import Hdf5JpegCodec
        return Hdf5JpegCodec()
    if name == "torchcodec":
        try:
            from .torch import TorchCodec
            return TorchCodec()
        except ImportError:
            logger.warning("torchcodec not available, falling back to PyAV")
            return PyAVCodec()
    return PyAVCodec()

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
    if name == "torchcodec":
        try:
            from .torch import TorchCodec
            return TorchCodec()
        except ImportError:
            logger.warning("torchcodec not available, falling back to PyAV")
            return PyAVCodec()
    return PyAVCodec()

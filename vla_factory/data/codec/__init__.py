"""Video codec registration and selection."""

from __future__ import annotations

import logging

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
# The reason the last torchcodec probe failed, kept so the fallback warning
# can name the real cause (e.g. a missing CUDA .so) rather than a generic
# "not installed" hint. Empty string on a clean miss / successful probe.
_TORCHCODEC_FAIL_REASON: str = ""

logger = logging.getLogger(__name__)


def resolve_codec(name: str = "auto", format: str | None = None) -> VideoCodec:
    """Construct a registered codec.

    ``auto`` is format-aware:
      - lerobot-v3  -> torchcodec when importable (ABI-safe), else PyAV
      - robotwin    -> hdf5_jpeg
      - other/None  -> PyAV stable default

    Explicit ``name`` always overrides format-based defaults.
    """
    global _TORCHCODEC_AVAILABLE, _TORCHCODEC_FAIL_REASON
    if name.lower() != "auto":
        return CodecRegistry.create(name)

    fmt = (format or "").lower()
    if fmt in ("lerobot-v3", "lerobot_v3", "lerobot"):
        if _TORCHCODEC_AVAILABLE is None:
            try:
                from .torchcodec import _load_torchcodec

                _load_torchcodec()
                _TORCHCODEC_AVAILABLE = True
            except Exception as exc:
                _TORCHCODEC_AVAILABLE = False
                _TORCHCODEC_FAIL_REASON = _torchcodec_fail_reason(exc)
                logger.warning(
                    "torchcodec unavailable for %r video: %s "
                    "Falling back to pyav (frame-accurate, slower than "
                    "torchcodec but no native-dep requirements).",
                    fmt, _TORCHCODEC_FAIL_REASON,
                )
        if _TORCHCODEC_AVAILABLE:
            return CodecRegistry.create("torchcodec")
        return CodecRegistry.create("pyav")
    if fmt in ("robotwin",):
        return CodecRegistry.create("hdf5_jpeg")
    return CodecRegistry.create("pyav")


def _torchcodec_fail_reason(exc: BaseException) -> str:
    """Return the most informative one-line cause from an import/ABI failure.

    ``_load_torchcodec`` wraps the real loader error in an ``ImportError``
    with a generic hint, so the actionable message (e.g. which CUDA ``.so``
    is missing) is on the chained ``__cause__`` / ``__context__``. Walk the
    chain to the bottom, then prefer a concrete linker line over the wrapper
    prose: torchcodec's RuntimeError embeds per-FFmpeg-version tracebacks and
    ends with "[end of libtorchcodec loading traceback]", so the last line is
    useless — we scan for a ``libX.so: cannot open shared object file`` line
    instead, which names the exact missing native dependency.
    """
    import re

    seen: list[BaseException] = []
    cur: BaseException | None = exc
    while cur is not None and cur not in seen:
        seen.append(cur)
        cur = cur.__cause__ or cur.__context__
    root = seen[-1] if seen else exc
    msg = str(root).strip() or type(root).__name__
    # 1. A concrete dynamic-loader line wins: "libfoo.so.N: cannot open ..."
    m = re.search(r"\b(lib[\w./+-]+\.so[\w.]*): cannot open shared object file", msg)
    if m:
        return f"missing native library {m.group(1)!r}"
    # 2. A plain ImportError from a missing module (torchcodec not installed).
    if isinstance(root, ImportError) and root is exc:
        first = msg.splitlines()[0].strip()
        return first if first else type(root).__name__
    # 3. Fallback: first non-empty line of the root exception.
    for line in msg.splitlines():
        line = line.strip()
        if line:
            return line
    return type(root).__name__


__all__ = [
    "CodecRegistry",
    "Hdf5JpegCodec",
    "PyAVCodec",
    "VideoCodec",
    "resolve_codec",
]

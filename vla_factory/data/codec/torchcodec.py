"""TorchCodec-based video decoder — optional VideoCodec implementation.

Uses PyTorch's ``torchcodec`` library (https://github.com/pytorch/torchcodec)
for frame-accurate decoding. ``torchcodec`` is an optional dependency: it is
imported lazily on first decode so the codec registry stays importable without
it (rationale: framework-wide "optional deps defer to call time").

Caching mirrors :class:`PyAVCodec` so both codecs behave identically on the
pipeline: one open decoder handle per video file and an in-memory LRU of
recently decoded frames (``max_cached_per_video``). torchcodec's
``get_frame_at(index=...)`` is frame-accurate random access, so no manual seek
bookkeeping is needed (unlike PyAV's ``_seek_to``).

Frames are requested with ``dimension_order="NHWC"`` so they come out as numpy
HWC uint8 with no transposition, matching the codec contract used by the rest
of the pipeline. Decoding runs on CPU: the contract is numpy arrays, so there
is no point holding frames on a CUDA device.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from ..data_schema import VideoRef
from .base import CachedVideoCodec
from .registry import CodecRegistry

logger = logging.getLogger(__name__)


def _expose_nvidia_cuda_libs() -> None:
    """Preload bundled CUDA ``.so`` files so torchcodec can link them.

    torchcodec's native wheels (``libtorchcodec_core*.so``) link against
    CUDA libraries that the torch wheel ships as separate ``nvidia-*-cu12``
    packages, whose files land under ``site-packages/nvidia/*/lib``. torch
    itself finds its own CUDA deps via RPATH, but torchcodec's ``.so`` has no
    such RPATH and the ``nvidia/*/lib`` dirs are not on the dynamic linker's
    search path, so importing torchcodec fails with e.g.
    ``libnppicc.so.12: cannot open shared object file`` even though the file
    is installed.

    The robust runtime fix is to preload those ``.so`` files into the process
    global symbol table (``RTLD_GLOBAL``) before torchcodec is imported: once
    loaded globally, the dynamic linker satisfies torchcodec's references
    from the already-loaded symbols. ``LD_LIBRARY_PATH`` cannot be set at
    runtime (the linker reads it once at process start), so ``ctypes.CDLL``
    is the only working in-process mechanism.

    Only ``nvidia/*/lib`` is scanned — ``torch/lib`` is left to torch's own
    ``_load_global_deps``. The two directories share no filenames, so there
    is no risk of a preload shadowing a torch-loaded library. Failures
    (non-ELF files, already loaded, unrelated) are silently skipped.
    """
    import ctypes
    import glob
    import sysconfig
    from pathlib import Path

    try:
        rtld_global = ctypes.RTLD_GLOBAL
    except AttributeError:
        return  # non-POSIX; nothing to do

    # site-packages root: resolve via sysconfig (purelib) which is stable
    # across venv layouts.
    purelib = Path(sysconfig.get_paths().get("purelib", ""))
    if not purelib.is_dir():
        return
    for lib_dir in glob.glob(str(purelib / "nvidia" / "*" / "lib")):
        for so in glob.glob(os.path.join(lib_dir, "*.so*")):
            try:
                ctypes.CDLL(so, mode=rtld_global)
            except OSError:
                pass


def _load_torchcodec() -> Any:
    """Import the torchcodec ``VideoDecoder`` lazily with an actionable error.

    A torchcodec wheel built against a different torch version fails to load
    its C++ core at import time; torchcodec surfaces that as ``RuntimeError``
    ("Could not load libtorchcodec"), while a missing install raises
    ``ImportError``. We catch all three so the user gets a hint pointing at
    the fix, not a raw linker traceback.

    Before importing, :func:`_expose_nvidia_cuda_libs` preloads the bundled
    CUDA ``.so`` files (e.g. ``libnppicc.so.12``) into the process symbol
    table so torchcodec can link them — without this, a torchcodec wheel
    whose native libs reference CUDA components shipped as separate
    ``nvidia-*-cu12`` packages fails to load even when those packages are
    installed, because their ``nvidia/*/lib`` dirs are not on the linker
    search path.
    """
    _expose_nvidia_cuda_libs()
    try:
        from torchcodec.decoders import VideoDecoder
    except (ImportError, OSError, RuntimeError) as exc:  # pragma: no cover - exercised only without the extra / with an ABI-mismatched wheel
        raise ImportError(
            "Reading videos with the 'torchcodec' codec requires the "
            "'torchcodec' package built for the installed torch (wheels are "
            "torch-version-locked; a mismatched wheel fails to load). Install "
            "the extra: pip install -e \".[torchcodec]\"."
        ) from exc
    return VideoDecoder


@CodecRegistry.register("torchcodec")
class TorchCodec(CachedVideoCodec):
    """Optional video codec — uses torchcodec to decode frames to numpy.

    Caching mirrors :class:`PyAVCodec`: a per-video-file cache of decoder
    handles plus a decoded-frame LRU.
    The open set is bounded by a file-level LRU (``max_open_videos``):
    every entry holds an fd plus a native decoder context, so an
    unbounded registry eventually exhausts fds. Native decoder resources
    are released when a cache is closed/dropped.
    """

    def __init__(
        self,
        max_cached_per_video: int = 32,
        max_open_videos: int = 32,
    ) -> None:
        super().__init__(max_cached_per_video, max_open_videos)

    @property
    def name(self) -> str:
        return "torchcodec"

    def _open_decoder(self, video_path: Path) -> dict[str, Any]:
        return {"decoder": None}

    def _decode_frame(self, decoder: dict[str, Any], ref: VideoRef) -> NDArray:
        if decoder["decoder"] is None:
            decoder["decoder"] = _load_torchcodec()(
                source=str(ref.video_path), dimension_order="NHWC", device="cpu"
            )
        image = decoder["decoder"].get_frame_at(index=ref.frame_index).data.cpu().numpy()
        if image.shape[:2] != (ref.height, ref.width):
            image = cv2.resize(image, (ref.width, ref.height))
        return image

    def _close_decoder(self, decoder: dict[str, Any]) -> None:
        decoder["decoder"] = None
        super()._close_decoder(decoder)

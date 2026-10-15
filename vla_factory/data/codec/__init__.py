"""Image-decoder registration, storage-aware selection, and validation."""

from __future__ import annotations

import logging
from collections.abc import Mapping

from ..data_schema import ImageRef, ImageStorage
from .base import ImageDecoder
from .registry import CodecRegistry, ImageDecoderRegistry

# Built-ins register on import. HDF5 keeps h5py lazy until its first decode.
from .hdf5_jpeg import Hdf5JpegCodec
from .pyav import PyAVCodec
from .rlds_jpeg import RldsJpegCodec

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


class ImageDecoderRouter:
    """Route heterogeneous image references to storage-compatible decoders."""

    def __init__(self, decoders: Mapping[ImageStorage, ImageDecoder]) -> None:
        self._decoders = dict(decoders)

    @property
    def name(self) -> str:
        names = ",".join(
            f"{storage.value}:{decoder.name}"
            for storage, decoder in sorted(self._decoders.items(), key=lambda item: item[0].value)
        )
        return f"auto[{names}]"

    @property
    def supported_storages(self) -> frozenset[ImageStorage]:
        return frozenset(self._decoders)

    def decode_image(self, ref: ImageRef):
        try:
            return self._decoders[ref.storage].decode_image(ref)
        except KeyError:
            supported = ", ".join(storage.value for storage in self.supported_storages)
            raise ValueError(
                f"Image decoder router does not support storage {ref.storage.value!r}; "
                f"supports: {supported}"
            ) from None

    def close(self) -> None:
        for decoder in self._decoders.values():
            close = getattr(decoder, "close", None)
            if close is not None:
                close()


def resolve_image_decoder(
    name: str = "auto",
    *,
    camera_storages: Mapping[str, ImageStorage | str],
    dataset_format: str | None = None,
) -> ImageDecoder:
    """Select and validate an image decoder for the observed storage types.

    ``auto`` creates a storage router when a dataset has heterogeneous image
    storage. An explicit decoder must support every storage type before a
    dataloader is created; this is intentionally a configuration error rather
    than a late failure in a worker process.
    """
    storages_by_camera = {
        camera: ImageStorage(storage)
        for camera, storage in camera_storages.items()
    }
    storages = frozenset(storages_by_camera.values())
    if not storages:
        raise ValueError(
            f"Dataset format {dataset_format or 'unknown'!r} declares no image storage types; "
            "a Reader must describe every camera's image storage."
        )

    if name.lower() == "auto":
        decoders = {storage: _auto_decoder_for_storage(storage) for storage in storages}
        if len(decoders) == 1:
            return next(iter(decoders.values()))
        return ImageDecoderRouter(decoders)

    decoder = ImageDecoderRegistry.create(name)
    validate_image_decoder(
        decoder,
        camera_storages=storages_by_camera,
        dataset_format=dataset_format,
    )
    return decoder


def validate_image_decoder(
    decoder: ImageDecoder,
    *,
    camera_storages: Mapping[str, ImageStorage | str],
    dataset_format: str | None = None,
) -> None:
    """Fail early when an existing decoder cannot read a schema's cameras."""
    supported_storages = getattr(
        decoder, "supported_storages", frozenset(ImageStorage)
    )
    unsupported = {
        camera: storage
        for camera, storage in (
            (camera, ImageStorage(storage))
            for camera, storage in camera_storages.items()
        )
        if storage not in supported_storages
    }
    if unsupported:
        detail = ", ".join(
            f"{camera}={storage.value}" for camera, storage in sorted(unsupported.items())
        )
        supports = ", ".join(sorted(storage.value for storage in supported_storages))
        raise ValueError(
            f"Image decoder {decoder.name!r} is incompatible with dataset format "
            f"{dataset_format or 'unknown'!r}: {detail}. It supports: {supports}."
        )


def resolve_codec(
    name: str = "auto",
    format: str | None = None,
    *,
    camera_storages: Mapping[str, ImageStorage | str] | None = None,
    dataset_format: str | None = None,
) -> ImageDecoder:
    """Compatibility wrapper around :func:`resolve_image_decoder`.

    New callers must pass ``camera_storages`` from ``DataSchema``. ``format``
    remains a legacy fallback for external users that have not yet adopted
    image-storage declarations.
    """
    if camera_storages is not None:
        return resolve_image_decoder(
            name,
            camera_storages=camera_storages,
            dataset_format=dataset_format or format,
        )

    # Legacy direct construction has no dataset facts to validate. Preserve
    # that API for callers that instantiate a codec in isolation; production
    # train/eval paths always pass ``camera_storages`` and are validated.
    if name.lower() != "auto":
        return ImageDecoderRegistry.create(name)

    storage = _legacy_storage_for_format(format)
    return resolve_image_decoder(
        name,
        camera_storages={"<legacy>": storage},
        dataset_format=dataset_format or format,
    )


def _legacy_storage_for_format(format_name: str | None) -> ImageStorage:
    fmt = (format_name or "").lower()
    if fmt in ("robotwin", "robotwin_hdf5"):
        return ImageStorage.HDF5_JPEG
    if fmt in ("rlds", "rlds_tfrecord"):
        return ImageStorage.ENCODED_BYTES
    return ImageStorage.VIDEO_FILE


def _auto_decoder_for_storage(storage: ImageStorage) -> ImageDecoder:
    global _TORCHCODEC_AVAILABLE, _TORCHCODEC_FAIL_REASON
    if storage is ImageStorage.VIDEO_FILE:
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
                    storage.value, _TORCHCODEC_FAIL_REASON,
                )
        if _TORCHCODEC_AVAILABLE:
            return CodecRegistry.create("torchcodec")
        return CodecRegistry.create("pyav")
    if storage is ImageStorage.HDF5_JPEG:
        return CodecRegistry.create("hdf5_jpeg")
    if storage is ImageStorage.ENCODED_BYTES:
        return CodecRegistry.create("rlds_jpeg")
    raise AssertionError(f"Unhandled image storage {storage!r}")


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
    "ImageDecoder",
    "ImageDecoderRegistry",
    "ImageDecoderRouter",
    "PyAVCodec",
    "RldsJpegCodec",
    "resolve_codec",
    "resolve_image_decoder",
    "validate_image_decoder",
]

"""Image transforms for raw dataset images."""

from __future__ import annotations

import numpy as np

from .base import TransformStep
from .normalize import IMAGENET_MEAN, IMAGENET_STD
from .registry import TransformRegistry


@TransformRegistry.register("image_to_float")
class ImageToFloat(TransformStep):
    """Convert ``images.*`` arrays to float32 in a configured range."""

    def __init__(self, range: tuple[float, float] = (0.0, 1.0)) -> None:  # noqa: A002
        self.range = tuple(range)

    def __call__(self, sample: dict) -> dict:
        lo, hi = self.range
        for key in list(sample.keys()):
            if not key.startswith("images."):
                continue
            img = sample[key]
            if not isinstance(img, np.ndarray):
                continue
            original_dtype = img.dtype
            img = img.astype(np.float32)
            max_value = float(np.max(img)) if img.size else 0.0
            if original_dtype == np.uint8 or max_value > 1.5:
                img = img / 255.0
            if (lo, hi) == (-1.0, 1.0):
                img = img * 2.0 - 1.0
            elif (lo, hi) != (0.0, 1.0):
                img = img * (hi - lo) + lo
            sample[key] = img.astype(np.float32, copy=False)
        return sample

    @classmethod
    def from_config(cls, cfg: dict, ctx=None) -> "ImageToFloat":
        return cls(range=tuple(cfg.get("range", (0.0, 1.0))))


@TransformRegistry.register("image_layout")
class ImageLayout(TransformStep):
    """Convert image arrays between HWC and CHW layout."""

    def __init__(self, to: str = "CHW") -> None:
        self.to = to.upper()
        if self.to not in {"CHW", "HWC"}:
            raise ValueError(f"Unsupported image layout target: {to!r}")

    def __call__(self, sample: dict) -> dict:
        for key in list(sample.keys()):
            if not key.startswith("images."):
                continue
            img = sample[key]
            if not (isinstance(img, np.ndarray) and img.ndim == 3):
                continue
            if self.to == "CHW" and img.shape[-1] in (1, 3, 4):
                sample[key] = np.ascontiguousarray(img.transpose(2, 0, 1))
            elif self.to == "HWC" and img.shape[0] in (1, 3, 4):
                sample[key] = np.ascontiguousarray(img.transpose(1, 2, 0))
        return sample


@TransformRegistry.register("image_normalize")
class ImageNormalize(TransformStep):
    """Normalize images using ImageNet constants or no-op."""

    def __init__(self, mode: str = "imagenet") -> None:
        self.mode = mode

    def __call__(self, sample: dict) -> dict:
        if self.mode in ("none", None):
            return sample
        if self.mode != "imagenet":
            raise ValueError(f"Unsupported image_normalize mode: {self.mode!r}")

        for key in list(sample.keys()):
            if not key.startswith("images."):
                continue
            img = sample[key]
            if not (isinstance(img, np.ndarray) and img.ndim == 3):
                continue
            if img.shape[0] in (1, 3, 4):  # CHW
                sample[key] = (img - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
            elif img.shape[-1] in (1, 3, 4):  # HWC
                sample[key] = (img - IMAGENET_MEAN[None, None, :]) / IMAGENET_STD[None, None, :]
        return sample

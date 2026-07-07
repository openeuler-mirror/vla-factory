"""Image resize transform using OpenCV (numpy-based)."""

from __future__ import annotations

import numpy as np

from .base import TransformStep
from .registry import TransformRegistry


@TransformRegistry.register("resize_images")
class ResizeImages(TransformStep):
    """Resize all ``images.*`` arrays in the flat sample dict.

    ``height`` and ``width`` are the single source of truth for resize target.
    If both are omitted, this transform is a no-op and images keep their
    dataset-native size.

    Expects images as numpy arrays in ``HWC`` or ``CHW`` format.
    """

    def __init__(
        self,
        height: int = 0,
        width: int = 0,
        mode: str = "stretch",
        interpolation: str = "bilinear",
    ) -> None:
        self.height = height
        self.width = width
        self.mode = mode
        self.interpolation = interpolation

    @classmethod
    def from_config(cls, cfg: dict, ctx=None) -> "ResizeImages | None":
        height = cfg.get("height")
        width = cfg.get("width")
        if height is None and width is None:
            return None
        if height is None or width is None:
            raise ValueError("resize_images requires both `height` and `width`.")
        height = int(height)
        width = int(width)
        if height <= 0 or width <= 0:
            raise ValueError("resize_images target height and width must be positive.")
        return cls(
            height=height,
            width=width,
            mode=cfg.get("mode", "stretch"),
            interpolation=cfg.get("interpolation", "bilinear"),
        )

    def __call__(self, sample: dict) -> dict:
        if self.height <= 0 or self.width <= 0:
            return sample

        import cv2

        interpolation = (
            cv2.INTER_AREA
            if self.interpolation == "area"
            else cv2.INTER_NEAREST
            if self.interpolation == "nearest"
            else cv2.INTER_LINEAR
        )

        for key in list(sample.keys()):
            if key.startswith("images."):
                img = sample[key]
                if isinstance(img, np.ndarray) and img.ndim == 3:
                    channels_first = img.shape[0] in (1, 3, 4)
                    hwc = img.transpose(1, 2, 0) if channels_first else img
                    resized = cv2.resize(hwc, (self.width, self.height), interpolation=interpolation)
                    sample[key] = resized.transpose(2, 0, 1) if channels_first else resized

        return sample

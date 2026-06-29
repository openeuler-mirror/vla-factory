"""Image resize transform using OpenCV (numpy-based)."""

from __future__ import annotations

import numpy as np

from .base import TransformStep
from .registry import TransformRegistry


@TransformRegistry.register("resize_images")
class ResizeImages(TransformStep):
    """Resize all ``images.*`` arrays in the flat sample dict.

    When ``height == 0`` and ``width == 0`` (the default), this is a no-op.
    This is the expected behaviour for ACT which uses the original resolution.

    Expects images as numpy arrays in ``[C, H, W]`` float32 format.
    """

    def __init__(self, height: int = 0, width: int = 0) -> None:
        self.height = height
        self.width = width

    def __call__(self, sample: dict) -> dict:
        if self.height <= 0 or self.width <= 0:
            return sample

        import cv2

        for key in list(sample.keys()):
            if key.startswith("images."):
                img = sample[key]
                if isinstance(img, np.ndarray) and img.ndim == 3:
                    # img is CHW → HWC for cv2, then back
                    hwc = img.transpose(1, 2, 0)
                    resized = cv2.resize(hwc, (self.width, self.height))
                    sample[key] = resized.transpose(2, 0, 1)

        return sample

    def state_dict(self) -> dict:
        return {"type": "resize_images", "height": self.height, "width": self.width}

    @classmethod
    def from_state_dict(cls, d: dict, **shared) -> "ResizeImages":
        return cls(height=d["height"], width=d["width"])

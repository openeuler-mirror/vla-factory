"""Image resize transform using OpenCV (numpy-based)."""

from __future__ import annotations

import numpy as np

from .base import TransformStep
from .registry import TransformRegistry


@TransformRegistry.register("resize_images")
class ResizeImages(TransformStep):
    """Resize all ``images.*`` arrays in the flat sample dict.

    The target is compiled from ``ModelIOSpec.camera_shapes``. The declaration
    controls resize policy only; it cannot introduce a second shape source.

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
    def compile_call(cls, cfg: dict, ctx) -> dict | None:
        if "height" in cfg or "width" in cfg:
            raise ValueError(
                "resize_images height/width are interface facts, not transform "
                "arguments. Use ModelMetadata.vision_slots[].resolution for a "
                "fixed model, or model.config.input_image_size for a tunable one."
            )
        targets = dict(ctx.target_camera_shapes or {})
        sources = dict(ctx.source_camera_shapes or {})
        needed = {
            camera: size for camera, size in targets.items()
            if sources.get(camera) != size
        }
        if not needed:
            return None
        target_sizes = set(needed.values())
        if len(target_sizes) != 1:
            raise ValueError(
                "resize_images currently requires one common target size; "
                f"the resolved model interface has {needed}."
            )
        height, width = next(iter(target_sizes))
        return {
            "height": height,
            "width": width,
            "mode": cfg.get("mode", "stretch"),
            "interpolation": cfg.get("interpolation", "bilinear"),
        }

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
                    if self.mode == "stretch":
                        out = cv2.resize(hwc, (self.width, self.height), interpolation=interpolation)
                    elif self.mode in ("pad", "letterbox", "keep_ratio"):
                        out = self._resize_with_pad(hwc, interpolation)
                    else:
                        raise ValueError(f"Unsupported resize_images mode: {self.mode!r}")
                    sample[key] = out.transpose(2, 0, 1) if channels_first else out

        return sample

    def _resize_with_pad(self, hwc: np.ndarray, interpolation) -> np.ndarray:
        """Resize keeping aspect ratio, then center-pad to ``(height, width)``.

        Mirrors openpi's ``resize_with_pad``: scale by ``min(tw/w, th/h)``,
        resize, then zero-pad the shorter side (centered). Avoids the
        aspect-ratio distortion of plain stretch — important for
        SigLIP/PaliGemma image inputs.
        """
        import cv2
        h, w = hwc.shape[:2]
        scale = min(self.width / w, self.height / h)
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))
        resized = cv2.resize(hwc, (new_w, new_h), interpolation=interpolation)
        pad_h = self.height - new_h
        pad_w = self.width - new_w
        top, left = pad_h // 2, pad_w // 2
        bottom, right = pad_h - top, pad_w - left
        canvas = np.zeros((self.height, self.width, hwc.shape[2]), dtype=hwc.dtype)
        canvas[top:top + new_h, left:left + new_w] = resized
        return canvas

"""Checkpoint-owned image transform, executed plan-side.

Some backbones own their image preprocessing end-to-end: OpenVLA's fused
DINOv2+SigLIP vision stack requires per-tower normalization stacked along
the channel dim (float32 [6, 224, 224]) — a contract the generic image
vocabulary (resize / single-stats normalize) cannot express. Models that
declare ``image_normalize_mode="checkpoint_processor"`` get their base
checkpoint's own processor applied here, once, instead of a framework
reimplementation whose drift would be invisible.

The processor is loaded lazily from the base checkpoint (the same
``model_path`` fallback the tokenizers use). It is TF-free to import but
only present in model environments, so the import happens inside model
runs — never at registry load.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .base import PlanContext, TransformStep
from .registry import TransformRegistry


def _registered_image_processor(repo: str):
    """Load the checkpoint's image processor, registering prismatic's HF
    classes first if this process has not already done so (the OpenVLA
    factory performs the same registration; both are idempotent)."""
    from transformers import AutoImageProcessor

    try:
        return AutoImageProcessor.from_pretrained(repo)
    except ValueError:
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor

        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        return AutoImageProcessor.from_pretrained(repo)


@TransformRegistry.register("checkpoint_image_transform")
class CheckpointImageTransform(TransformStep):
    """Apply the base checkpoint's own image processor to the primary camera.

    Raw HWC uint8 in, the checkpoint's ``pixel_values`` contract out (for
    OpenVLA's fused backbone: per-tower normalized, channel-stacked
    float32 — the vision backbone unpacks the towers). Emitted under
    ``pixel_values`` and carried through collate into ``Observation``.
    """

    def __init__(self, source_key: str, repo: str):
        self.source_key = source_key
        self.repo = repo
        self._processor = None

    def __call__(self, sample: dict) -> dict:
        if self._processor is None:
            self._processor = _registered_image_processor(self.repo)
        img = np.asarray(sample[self.source_key]).astype(np.uint8)
        sample["pixel_values"] = self._processor.apply_transform(
            Image.fromarray(img)
        )
        return sample

    @classmethod
    def compile_call(cls, cfg: dict, ctx: PlanContext) -> dict:
        if ctx.tokenizer_repo is None:
            raise ValueError(
                f"{ctx.metadata.name!r} declares the checkpoint image "
                "transform but no model.path — the processor is loaded from "
                "the base checkpoint."
            )
        return {"source_key": cfg["source_key"], "repo": ctx.tokenizer_repo}

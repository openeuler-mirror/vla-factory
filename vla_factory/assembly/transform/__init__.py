"""Transform sub-package.

Importing this package populates the :class:`TransformRegistry` (the
``@register`` decorators on the step classes below execute at import time).

Public surface:
- ``TransformStep``      — forward-only ABC
- ``TransformRegistry``  — name → step class + pairing metadata
- ``TransformPipeline``  — ordered forward-only steps
- ``build_pipeline``     — resolved TransformPipelinePlan → runnable pipeline
- ``Normalize`` / ``ResizeImages`` / ``PadDimensions`` — concrete steps
"""

from .base import TransformStep
from .registry import TransformRegistry
from .pipeline import TransformPipeline, TransformContext, build_pipeline
from .plan import TransformPipelinePlan, TransformStepCall

# Importing the step modules registers them with TransformRegistry.
from .normalize import Normalize, NormalizeVector, UnnormalizeActionStep, IMAGENET_MEAN, IMAGENET_STD
from .resize_images import ResizeImages
from .pad_dimensions import PadDimensions, UnpadAction
from .images import ImageToFloat, ImageLayout, ImageNormalize
from .task_tokenize import TaskTokenize

__all__ = [
    "TransformStep",
    "TransformRegistry",
    "TransformPipeline",
    "TransformContext",
    "build_pipeline",
    "TransformPipelinePlan",
    "TransformStepCall",
    "Normalize",
    "NormalizeVector",
    "UnnormalizeActionStep",
    "ResizeImages",
    "PadDimensions",
    "UnpadAction",
    "ImageToFloat",
    "ImageLayout",
    "ImageNormalize",
    "TaskTokenize",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]

"""Transforms sub-package.

Importing this package populates the :class:`TransformRegistry` (the
``@register`` decorators on the step classes below execute at import time).

Public surface:
- ``TransformStep``      — forward-only ABC
- ``TransformRegistry``  — name → step class + pairing metadata
- ``TransformPipeline``  — ordered forward-only steps
- ``build_preprocessor`` — model-declared transform list → forward pipeline
- ``build_transforms``   — model-declared transform list → (preprocessor, postprocessor)
- ``Normalize`` / ``ResizeImages`` / ``PadDimensions`` — concrete steps
"""

from .base import TransformStep
from .registry import TransformRegistry
from .pipeline import TransformPipeline, BuildContext, build_preprocessor, build_transforms

# Importing the step modules registers them with TransformRegistry.
from .normalize import Normalize, UnnormalizeActionStep, IMAGENET_MEAN, IMAGENET_STD
from .resize_images import ResizeImages
from .pad_dimensions import PadDimensions

__all__ = [
    "TransformStep",
    "TransformRegistry",
    "TransformPipeline",
    "BuildContext",
    "build_preprocessor",
    "build_transforms",
    "Normalize",
    "UnnormalizeActionStep",
    "ResizeImages",
    "PadDimensions",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]

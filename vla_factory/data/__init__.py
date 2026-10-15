"""Data pipeline for VLA-Factory (read-only Canonical IR).

Reads robotics datasets and produces the canonical intermediate
representation (``DataSchema`` / ``Episode`` / ``Frame`` / ``NormStats``).
Sample construction (``VLADataset`` / ``collate_fn`` / ``create_dataloader``)
lives in the training layer (``vla_factory.training``).
"""

from .data_schema import (
    DataSchema,
    Episode,
    FeatureStats,
    Frame,
    ImageStorage,
    ImageRef,
    NormStats,
    describe_dataset,
)
from .reader import FormatReader, ReaderRegistry, get_reader
from .codec import (
    CodecRegistry,
    ImageDecoder,
    ImageDecoderRegistry,
    resolve_codec,
    resolve_image_decoder,
    validate_image_decoder,
)

__all__ = [
    "CodecRegistry",
    "DataSchema",
    "Episode",
    "FeatureStats",
    "FormatReader",
    "Frame",
    "ImageStorage",
    "ImageDecoder",
    "ImageDecoderRegistry",
    "ImageRef",
    "NormStats",
    "ReaderRegistry",
    "describe_dataset",
    "resolve_codec",
    "resolve_image_decoder",
    "validate_image_decoder",
    "get_reader",
]

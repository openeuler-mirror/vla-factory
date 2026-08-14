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
    NormStats,
    VideoRef,
    describe_dataset,
)
from .reader import FormatReader, ReaderRegistry, get_reader
from .codec import CodecRegistry, VideoCodec, resolve_codec

__all__ = [
    "CodecRegistry",
    "DataSchema",
    "Episode",
    "FeatureStats",
    "FormatReader",
    "Frame",
    "NormStats",
    "ReaderRegistry",
    "VideoRef",
    "VideoCodec",
    "describe_dataset",
    "resolve_codec",
    "get_reader",
]

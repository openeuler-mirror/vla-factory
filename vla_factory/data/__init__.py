"""Data pipeline for VLA-Factory (read-only Canonical IR).

Reads robotics datasets and produces the canonical intermediate
representation (``DataSchema`` / ``Episode`` / ``Frame`` / ``NormStats``).
Sample construction (``VLADataset`` / ``collate_fn`` / ``create_dataloaders``)
lives in the training layer (``vla_factory.training``).
"""

from .manifest import DataSchema, DatasetManifest, FeatureStats, NormStats, SampleLocator
from .formats import LeRobotV3Reader, get_reader, Frame, Episode, VideoRef
from .codec import PyAVCodec, VideoCodec, resolve_codec

__all__ = [
    "DataSchema",
    "DatasetManifest",
    "FeatureStats",
    "NormStats",
    "SampleLocator",
    "PyAVCodec",
    "VideoCodec",
    "LeRobotV3Reader",
    "resolve_codec",
    "get_reader",
    "Frame",
    "Episode",
    "VideoRef",
]

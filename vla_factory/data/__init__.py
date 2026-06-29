"""Data pipeline for VLA-Factory.

Reads robotics datasets and produces training samples
in the format ``{"observation": Observation, "actions": Tensor}``.
"""

from .manifest import DataSchema, DatasetManifest, FeatureStats, NormStats, SampleLocator
from .dataset import VLADataset, collate_fn
from .loader import create_dataloaders
from .formats import LeRobotV3Reader, get_reader, Frame, Episode, VideoRef
from .codec import PyAVCodec, VideoCodec, resolve_codec

__all__ = [
    "DataSchema",
    "DatasetManifest",
    "FeatureStats",
    "NormStats",
    "SampleLocator",
    "VLADataset",
    "collate_fn",
    "create_dataloaders",
    "PyAVCodec",
    "VideoCodec",
    "LeRobotV3Reader",
    "resolve_codec",
    "get_reader",
    "Frame",
    "Episode",
    "VideoRef",
]

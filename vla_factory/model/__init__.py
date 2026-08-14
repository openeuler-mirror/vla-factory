"""Public model interface and adapter registry."""

from .model_interface import (
    ModelMetadata,
    Observation,
    VisionSlot,
    VLAModel,
    VLAModelJAX,
    VLAModelPyTorch,
)
from .registry import (
    ModelEntry,
    ModelRegistry,
    RegistryLoadError,
    get_entry,
    list_entries,
    register_vla,
)

__all__ = [
    "ModelEntry",
    "ModelMetadata",
    "ModelRegistry",
    "Observation",
    "RegistryLoadError",
    "VisionSlot",
    "VLAModel",
    "VLAModelJAX",
    "VLAModelPyTorch",
    "get_entry",
    "list_entries",
    "register_vla",
]

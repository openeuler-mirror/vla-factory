"""Inference engine, action execution, and deployment entry points.

Concrete platform and transport implementations remain in their subpackages so
importing :mod:`vla_factory.inference` does not load optional dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_PUBLIC_OBJECTS = {
    "ActionChunk": ("vla_factory.inference.execution", "ActionChunk"),
    "ActionCommand": ("vla_factory.inference.execution", "ActionCommand"),
    "DeploymentConfig": ("vla_factory.inference.deploy", "DeploymentConfig"),
    "InferenceEngine": (
        "vla_factory.inference.inference_engine",
        "InferenceEngine",
    ),
    "ObsDict": ("vla_factory.inference.inference_engine", "ObsDict"),
    "PolicyExecutor": ("vla_factory.inference.execution", "PolicyExecutor"),
    "ReplayPolicy": ("vla_factory.inference.execution", "ReplayPolicy"),
    "deploy": ("vla_factory.inference.deploy", "deploy"),
    "evaluate_dataset": (
        "vla_factory.inference.evaluate_dataset",
        "evaluate_dataset",
    ),
    "infer_dataset_sample": (
        "vla_factory.inference.evaluate_dataset",
        "infer_dataset_sample",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, object_name = _PUBLIC_OBJECTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), object_name)
    globals()[name] = value
    return value


__all__ = sorted(_PUBLIC_OBJECTS)

"""Common protocols for deployment platform adapters."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from vla_factory.inference.inference_engine import ObsDict


@runtime_checkable
class PlatformObservationAdapter(Protocol):
    """Translate a platform observation into VLA Factory's stable contract."""

    def __call__(self, observation: Any, task: str = "") -> ObsDict:
        ...

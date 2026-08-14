"""Adapter for the flat simulator/ZMQ observation format."""

from __future__ import annotations

import numpy as np

from vla_factory.inference.inference_engine import ObsDict


class SimulatorAdapter:
    """Convert ``observation.images.*`` ZMQ keys into ``ObsDict``."""

    def __init__(self, camera_keys: tuple[str, ...]) -> None:
        self.camera_keys = camera_keys

    def __call__(self, observation: dict, task: str = "") -> ObsDict:
        video: dict[str, np.ndarray] = {}
        for cam in self.camera_keys:
            key = f"observation.images.{cam}"
            if key not in observation:
                raise KeyError(
                    f"Expected image key '{key}' in observation, "
                    f"got: {list(observation.keys())}"
                )
            video[cam] = observation[key]

        state = observation.get("observation.state")
        if state is not None:
            state = np.asarray(state, dtype=np.float32)

        language = observation.get("language") or task or None
        return ObsDict(video=video, state=state, language=language)

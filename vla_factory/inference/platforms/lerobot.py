"""LeRobot policy-interface and host wire-format adapters."""

from __future__ import annotations

import base64
import logging
from typing import Any

import cv2
import numpy as np
import torch

from vla_factory.inference.execution import ActionCommand, PolicyExecutor
from vla_factory.inference.inference_engine import ObsDict

logger = logging.getLogger(__name__)


class LeRobotAdapter:
    """Expose an inference engine through LeRobot's ``predict_action`` API."""

    def __init__(self, policy: PolicyExecutor) -> None:
        self.policy = policy

    def predict_action(self, obs_tensor_dict: dict) -> torch.Tensor:
        video: dict[str, np.ndarray] = {}
        for key, value in obs_tensor_dict.items():
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    if "image" in nested_key or "images" in nested_key:
                        video[nested_key] = _as_numpy(nested_value)
            elif "image" in key or "images" in key:
                video[key] = _as_numpy(value)

        state_value = obs_tensor_dict.get("observation.state")
        state = _as_numpy(state_value) if state_value is not None else None
        language = obs_tensor_dict.get("language_instruction")
        if language is not None and hasattr(language, "item"):
            language = language.item()

        command = self.policy.predict(
            ObsDict(video=video, state=state, language=language)
        )
        if not isinstance(command, ActionCommand):
            raise TypeError(
                "PolicyExecutor.predict() must return ActionCommand, got "
                f"{type(command).__name__}."
            )
        return torch.from_numpy(command.single())

    def reset(self) -> None:
        self.policy.reset()


class LerobotHostObsAdapter:
    """Convert a LeRobot host wire observation into ``ObsDict``."""

    def __init__(
        self,
        camera_keys: tuple[str, ...],
        state_keys: tuple[str, ...],
        state_dim: int,
    ) -> None:
        self._camera_keys = tuple(camera_keys)
        self._state_keys = tuple(state_keys)
        # Deployment contract check — a plain raise, not `assert`, so it
        # survives `python -O` (asserts are stripped under optimization).
        if len(self._state_keys) != state_dim:
            raise ValueError(
                f"state_keys count ({len(self._state_keys)}) does not match "
                f"state_dim ({state_dim}); got keys {list(self._state_keys)}"
            )
        logger.info(
            "LerobotHostObsAdapter — state keys: %s, cameras: %s",
            list(self._state_keys),
            list(self._camera_keys),
        )

    def __call__(self, observation: dict[str, Any], task: str = "") -> ObsDict:
        video: dict[str, np.ndarray] = {}
        for camera in self._camera_keys:
            raw = observation.get(camera)
            if raw is None:
                raise KeyError(
                    f"Camera '{camera}' not found in observation. "
                    f"Available: {list(observation.keys())}"
                )
            if isinstance(raw, str):
                video[camera] = _decode_base64_jpeg(raw)
            elif isinstance(raw, np.ndarray):
                video[camera] = raw
            else:
                raise TypeError(
                    f"Unexpected type for camera '{camera}': {type(raw).__name__}"
                )

        state_parts: list[float] = []
        for key in self._state_keys:
            if key not in observation:
                raise KeyError(
                    f"State motor key '{key}' not found in observation. "
                    f"Available: {list(observation.keys())}"
                )
            state_parts.append(float(observation[key]))

        return ObsDict(
            video=video,
            state=np.array(state_parts, dtype=np.float32),
            language=task or None,
        )


class LerobotHostActionAdapter:
    """Map an action vector to ordered LeRobot host motor keys."""

    def __init__(self, action_dim: int, action_keys: tuple[str, ...]) -> None:
        self.action_dim = action_dim
        self._action_keys = tuple(action_keys)
        # Deployment contract check — a plain raise, not `assert`, so it
        # survives `python -O` (asserts are stripped under optimization).
        if len(self._action_keys) != action_dim:
            raise ValueError(
                f"action_keys count ({len(self._action_keys)}) does not match "
                f"action_dim ({action_dim}); got keys {list(self._action_keys)}"
            )
        logger.info(
            "LerobotHostActionAdapter — action mapping: %s",
            list(self._action_keys),
        )

    def __call__(self, action: np.ndarray) -> dict[str, float]:
        flat_action = np.asarray(action, dtype=np.float32)
        if flat_action.shape != (self.action_dim,):
            raise ValueError(
                "LerobotHostActionAdapter requires one action vector with shape "
                f"({self.action_dim},), got {flat_action.shape}."
            )
        return {
            key: float(value)
            for key, value in zip(self._action_keys, flat_action)
        }


def _as_numpy(value: Any) -> np.ndarray:
    return value.cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)


def _decode_base64_jpeg(b64_str: str) -> np.ndarray:
    jpg_data = base64.b64decode(b64_str)
    np_arr = np.frombuffer(jpg_data, dtype=np.uint8)
    bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("Failed to JPEG-decode observation image.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


__all__ = [
    "LeRobotAdapter",
    "LerobotHostObsAdapter",
    "LerobotHostActionAdapter",
]
